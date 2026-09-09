import itertools
import math
from numbers import Number
from typing import Any, Mapping, Sequence

import keras
from keras import ops

from ...utils.image import resample_displacement_field
from ..base import RandomTransform
from ..spatial.affine_utils import spacing_from_affine
from ..tensor_bundle import TensorBundle
from ..utils import (
    ensure_batch_axis_for_layout,
    get_input_layout_info,
    resolve_input_layout,
    restore_from_batch_axis,
    validate_affine_matrix,
    validate_tensor_matches_layout,
)


def _gaussian_kernel_1d(sigma: Any, radius: int, dtype: str = "float32") -> Any:
    offsets = ops.arange(-radius, radius + 1, dtype=dtype)
    kernel = ops.exp(-0.5 * ops.square(offsets / sigma))
    return kernel / ops.sum(kernel)


def _reflect_pad_axis(tensor: Any, axis: int, radius: int) -> Any:
    """Reflect-pad one axis without relying on backend-specific pad modes."""
    if radius == 0:
        return tensor

    size = ops.shape(tensor)[axis]
    left_indices = ops.arange(1, radius + 1, dtype="int32")
    right_indices = ops.arange(size - radius - 1, size - 1, dtype="int32")
    left = ops.flip(ops.take(tensor, left_indices, axis=axis), axis=axis)
    right = ops.flip(ops.take(tensor, right_indices, axis=axis), axis=axis)
    return ops.concatenate([left, tensor, right], axis=axis)


def _smooth_along_axis(
    tensor: Any,
    kernel_1d: Any,
    axis: int,
    radius: int,
    spatial_rank: int,
) -> Any:
    kernel_shape = [1] * spatial_rank
    kernel_shape[axis] = 2 * radius + 1
    kernel = ops.reshape(kernel_1d, kernel_shape + [1, 1])
    padded = _reflect_pad_axis(tensor, axis + 1, radius)
    return ops.conv(padded, kernel, padding="valid")


def _sigma_for_axis(sigma: Any, axis: int, spatial_rank: int) -> Any:
    """Return the per-sample smoothing width for one spatial axis."""
    if isinstance(sigma, Number) or len(sigma.shape) == 0:
        return sigma
    if len(sigma.shape) == 1:
        return sigma
    if len(sigma.shape) == 2 and sigma.shape[1] == spatial_rank:
        return ops.take(sigma, axis, axis=1)
    raise ValueError("sigma must be scalar, rank 1, or rank 2.")


def _gaussian_weights_broadcast(sigma: Any, radius: int) -> Any:
    """Create scalar or per-sample Gaussian weights for fixed tap offsets."""
    offsets = ops.arange(-radius, radius + 1, dtype="float32")
    if isinstance(sigma, Number) or len(sigma.shape) == 0:
        weights = ops.exp(-0.5 * ops.square(offsets / sigma))
    else:
        weights = ops.exp(
            -0.5
            * ops.square(
                ops.expand_dims(offsets, axis=0)
                / ops.expand_dims(sigma, axis=1)
            )
        )
    return weights / ops.sum(weights, axis=-1, keepdims=True)


def _smooth_axis_broadcast(
    tensor: Any,
    sigma: Any,
    axis: int,
    radius: int,
    spatial_shape: Sequence[int],
) -> Any:
    """Smooth one axis using broadcasted per-sample Gaussian weights."""
    padded_axis = axis + 1
    padded = _reflect_pad_axis(tensor, padded_axis, radius)
    weights = _gaussian_weights_broadcast(sigma, radius)
    output = ops.zeros_like(tensor)
    size = spatial_shape[axis]

    for tap in range(2 * radius + 1):
        indices = ops.arange(tap, tap + size, dtype="int32")
        shifted = ops.take(padded, indices, axis=padded_axis)
        if len(weights.shape) == 1:
            weight = weights[tap]
        else:
            weight = ops.reshape(
                weights[:, tap],
                [weights.shape[0]] + [1] * (len(spatial_shape) + 1),
            )
        output = output + shifted * weight
    return output


def _gaussian_smooth_broadcast_nd(
    field: Any,
    sigma: Any,
    spatial_rank: int,
    *,
    max_sigma: float | Sequence[float] | None = None,
) -> Any:
    """Smooth a field with per-sample sigma without vectorized mapping."""
    if max_sigma is None:
        radius_sigma = (sigma,) * spatial_rank
    elif isinstance(max_sigma, Number):
        radius_sigma = (max_sigma,) * spatial_rank
    else:
        if len(max_sigma) != spatial_rank:
            raise ValueError("`max_sigma` must contain one value per spatial axis.")
        radius_sigma = tuple(max_sigma)
    if any(not isinstance(value, Number) for value in radius_sigma):
        raise ValueError("A static maximum sigma is required for Gaussian smoothing.")

    spatial_shape = tuple(field.shape[1 : spatial_rank + 1])
    if any(size is None for size in spatial_shape):
        raise ValueError("Field spatial dimensions must be statically known.")
    radii = tuple(
        min(max(1, int(round(3.0 * float(value)))), int(size) - 1)
        for value, size in zip(radius_sigma, spatial_shape, strict=True)
    )
    if any(radius <= 0 for radius in radii):
        return field

    result = field
    for axis, radius in enumerate(radii):
        result = _smooth_axis_broadcast(
            result,
            _sigma_for_axis(sigma, axis, spatial_rank),
            axis,
            radius,
            spatial_shape,
        )
    return result


def _gaussian_smooth_nd(
    field: Any,
    sigma: Any,
    spatial_rank: int,
    *,
    max_sigma: float | Sequence[float] | None = None,
) -> Any:
    """Smooth each displacement channel with separable Gaussian kernels."""
    if max_sigma is None:
        radius_sigma = (sigma,) * spatial_rank
    elif isinstance(max_sigma, Number):
        radius_sigma = (max_sigma,) * spatial_rank
    else:
        if len(max_sigma) != spatial_rank:
            raise ValueError("`max_sigma` must contain one value per spatial axis.")
        radius_sigma = tuple(max_sigma)

    if any(not isinstance(value, Number) for value in radius_sigma):
        raise ValueError("A static maximum sigma is required for Gaussian smoothing.")
    static_spatial_shape = tuple(field.shape[1 : spatial_rank + 1])
    radii = tuple(max(1, int(round(3.0 * float(value)))) for value in radius_sigma)
    if all(size is not None for size in static_spatial_shape):
        radii = tuple(
            min(radius, int(size) - 1)
            for radius, size in zip(radii, static_spatial_shape, strict=True)
        )
        if any(radius <= 0 for radius in radii):
            return field
    shape = ops.shape(field)
    spatial_shape = [shape[index + 1] for index in range(spatial_rank)]
    channels = shape[-1]

    # Fold displacement channels into the batch dimension for independent
    # convolution, then restore the original channel-last field layout.
    permutation = [0, spatial_rank + 1] + list(range(1, spatial_rank + 1))
    folded = ops.transpose(field, permutation)
    folded = ops.reshape(folded, [-1] + spatial_shape + [1])
    for axis in range(spatial_rank):
        if isinstance(sigma, Number) or len(sigma.shape) == 0:
            sigma_axis = sigma
        else:
            sigma_axis = ops.take(sigma, axis, axis=0)
        kernel = _gaussian_kernel_1d(sigma_axis, radii[axis], dtype="float32")
        folded = _smooth_along_axis(
            folded,
            kernel,
            axis,
            radii[axis],
            spatial_rank,
        )

    folded = ops.reshape(folded, [shape[0], channels] + spatial_shape)
    inverse_permutation = [0] + list(range(2, spatial_rank + 2)) + [1]
    return ops.transpose(folded, inverse_permutation)


def _flatten_gather(volume: Any, batch_indices: Any, spatial_indices: Sequence[Any]) -> Any:
    """Gather channel vectors using flat indices for a channel-last volume."""
    shape = ops.shape(volume)
    spatial_sizes = shape[1:-1]
    flat_index = spatial_indices[0]
    for index in range(1, len(spatial_indices)):
        flat_index = flat_index * spatial_sizes[index] + spatial_indices[index]

    total_spatial = 1
    for size in spatial_sizes:
        total_spatial = total_spatial * size
    flat_index = batch_indices * total_spatial + flat_index

    flattened = ops.reshape(volume, [-1, shape[-1]])
    gathered = ops.take(flattened, ops.reshape(flat_index, [-1]), axis=0)
    output_shape = list(ops.shape(flat_index)) + [shape[-1]]
    return ops.reshape(gathered, output_shape)


def _normalize_coordinates(
    volume: Any,
    coordinates: Any,
    fill_mode: str,
) -> tuple[Any, Any]:
    """Map sampling coordinates to the requested boundary behavior."""
    spatial_rank = coordinates.shape[-1]
    shape = ops.shape(volume)
    valid = ops.ones_like(coordinates[..., 0], dtype="bool")
    normalized = []
    for axis in range(spatial_rank):
        size = shape[axis + 1]
        coordinate = coordinates[..., axis]
        if fill_mode == "constant":
            size_value = ops.cast(size, coordinate.dtype)
            valid = ops.logical_and(
                valid,
                ops.logical_and(coordinate >= 0.0, coordinate <= size_value - 1.0),
            )
            normalized.append(ops.clip(coordinate, 0.0, size_value - 1.0))
        elif fill_mode == "reflect":
            static_size = volume.shape[axis + 1]
            if static_size == 1:
                normalized.append(ops.zeros_like(coordinate))
            else:
                period = ops.cast(2 * (int(static_size) - 1), coordinate.dtype)
                reflected = ops.mod(ops.abs(coordinate), period)
                edge = ops.cast(int(static_size) - 1, coordinate.dtype)
                normalized.append(ops.where(reflected <= edge, reflected, period - reflected))
        elif fill_mode == "wrap":
            normalized.append(ops.mod(coordinate, ops.cast(size, coordinate.dtype)))
        else:  # nearest
            normalized.append(ops.clip(coordinate, 0.0, ops.cast(size, coordinate.dtype) - 1.0))
    return ops.stack(normalized, axis=-1), valid


def _linear_sample(
    volume: Any,
    coordinates: Any,
    fill_mode: str = "nearest",
    fill_value: float = 0.0,
) -> Any:
    """Sample a 2D or 3D channel-last volume with linear interpolation."""
    spatial_rank = coordinates.shape[-1]
    coordinates, valid = _normalize_coordinates(volume, coordinates, fill_mode)
    shape = ops.shape(volume)
    spatial_sizes = [ops.cast(shape[index + 1], volume.dtype) for index in range(spatial_rank)]
    floors = [ops.floor(coordinates[..., index]) for index in range(spatial_rank)]
    fractions = [
        (coordinates[..., index] - floors[index])[..., None] for index in range(spatial_rank)
    ]

    batch_indices = ops.arange(shape[0], dtype="int32")
    batch_indices = ops.reshape(batch_indices, [shape[0]] + [1] * spatial_rank)
    batch_indices = ops.broadcast_to(batch_indices, ops.shape(floors[0]))
    output = ops.zeros(list(ops.shape(floors[0])) + [shape[-1]], dtype=volume.dtype)

    for corner in itertools.product((0, 1), repeat=spatial_rank):
        indices = []
        weight = ops.ones_like(fractions[0])
        for axis, bit in enumerate(corner):
            coordinate = ops.clip(
                floors[axis] + bit,
                0.0,
                spatial_sizes[axis] - 1.0,
            )
            indices.append(ops.cast(coordinate, "int32"))
            weight = weight * (fractions[axis] if bit else (1.0 - fractions[axis]))
        output = output + _flatten_gather(volume, batch_indices, indices) * weight
    if fill_mode == "constant":
        fill = ops.cast(fill_value, output.dtype)
        output = ops.where(valid[..., None], output, fill)
    return output


def _nearest_sample(
    volume: Any,
    coordinates: Any,
    fill_mode: str = "nearest",
    fill_value: float = 0.0,
) -> Any:
    """Sample a channel-last volume with nearest-neighbor interpolation."""
    spatial_rank = coordinates.shape[-1]
    coordinates, valid = _normalize_coordinates(volume, coordinates, fill_mode)
    shape = ops.shape(volume)
    spatial_sizes = [ops.cast(shape[index + 1], coordinates.dtype) for index in range(spatial_rank)]
    indices = []
    for axis in range(spatial_rank):
        coordinate = ops.clip(
            ops.round(coordinates[..., axis]),
            0.0,
            spatial_sizes[axis] - 1.0,
        )
        indices.append(ops.cast(coordinate, "int32"))

    batch_indices = ops.arange(shape[0], dtype="int32")
    batch_indices = ops.reshape(batch_indices, [shape[0]] + [1] * spatial_rank)
    batch_indices = ops.broadcast_to(batch_indices, ops.shape(indices[0]))
    output = _flatten_gather(volume, batch_indices, indices)
    if fill_mode == "constant":
        fill = ops.cast(fill_value, output.dtype)
        output = ops.where(valid[..., None], output, fill)
    return output


def _lock_field_borders(field: Any, locked_borders: int, spatial_rank: int) -> Any:
    """Zero displacement on the outer control-grid layers."""
    if locked_borders == 0:
        return field

    shape = ops.shape(field)
    interior = ops.ones_like(field[..., 0], dtype="bool")
    for axis in range(spatial_rank):
        size = shape[axis + 1]
        coordinates = ops.arange(size, dtype="int32")
        axis_interior = ops.logical_and(
            coordinates >= locked_borders,
            coordinates < size - locked_borders,
        )
        reshape = [1] * (spatial_rank + 1)
        reshape[axis + 1] = size
        axis_interior = ops.reshape(axis_interior, reshape)
        interior = ops.logical_and(interior, axis_interior)
    return ops.where(interior[..., None], field, ops.zeros_like(field))


class RandomElasticTransform(RandomTransform):
    """Apply random smooth elastic deformation to 2D or 3D tensors.

    One displacement field is sampled per batch item and shared by all
    selected keys, keeping aligned images and masks geometrically consistent.
    For batch layouts, ``prob``, ``alpha``, and ``sigma`` are sampled
    independently for each batch item. For sample layouts, they are sampled
    once for the single sample.
    Two- and three-dimensional fields can optionally be sampled on a coarse
    grid and expanded to the input resolution. ``control_grid_spacing=None``
    keeps the full-resolution path for both ranks.

    Args:
        keys: Keys of aligned tensors to deform.
        alpha: Maximum displacement magnitude in the units selected by
            ``displacement_units``. A scalar uses the same value for every
            batch item; a ``(min, max)`` range samples one value per batch
            item.
        sigma: Gaussian smoothing width in the units selected by
            ``displacement_units``. A scalar uses the same value for every
            batch item; a ``(min, max)`` range samples one value per batch
            item.
        interpolation: Optional interpolation mode, a sequence aligned with
            ``keys``, or a mapping from key to mode. When omitted, the first
            key uses ``"bilinear"`` for 2D or ``"trilinear"`` for 3D, and
            later keys use ``"nearest"``. Explicit mappings are recommended
            for image and label pipelines.
        fill_mode: Boundary behavior for out-of-bounds coordinates. Supported
            values are ``"nearest"``, ``"constant"``, ``"reflect"``, and
            ``"wrap"``. The default ``"nearest"`` preserves border values.
        fill_value: Value used outside the input when ``fill_mode="constant"``.
        prob: Per-sample probability of applying the deformation. For a batch,
            each item receives an independent application decision.
        input_layout: One of ``"HWC"``, ``"DHWC"``, ``"BHWC"``, or
            ``"BDHWC"``.
        control_grid_spacing: Optional spacing between coarse field samples,
            in pixel/voxel units. A scalar applies to every spatial axis, while
            a sequence must contain exactly two values for 2D or three values
            for 3D. If ``None``, the field is sampled at full resolution.
        displacement_units: Units for ``alpha`` and the sampled displacement
            field. ``"voxel"`` is the default for both ranks; for 2D it means
            pixel units, and for 3D it means voxel units. ``"mm"`` samples
            physical displacement magnitudes and converts each tensor spatial
            axis using the corresponding spacing extracted from
            ``bundle.meta["affine"]``. The affine must describe the same
            spatial-axis order as the input tensor.
        minimum_physical_spacing: Static lower bound for the physical spacing
            of each tensor spatial axis, in millimeters. Required when
            ``displacement_units="mm"`` so the Gaussian kernel radius can be
            bounded safely during graph execution. A scalar is broadcast to
            every spatial axis.
        field_interpolation: Interpolation used to expand a coarse field. If
            ``None``, 2D fields use ``"bilinear"`` and 3D fields use
            ``"trilinear"``.
            ``"bspline"`` is supported for both ranks and treats coarse values
            as control-point coefficients.
        locked_borders: Number of outer coarse-grid layers with zero
            displacement. This applies to the control grid for both 2D and 3D
            fields. For example, ``locked_borders=1`` keeps the outermost
            control-point layer fixed while allowing the interior to move.
        seed: Optional integer or Keras ``SeedGenerator``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Validation rules:
        ``input_layout`` determines the spatial rank: ``HWC``/``BHWC`` are
        2D and ``DHWC``/``BDHWC`` are 3D. ``control_grid_spacing`` must have
        exactly one positive value per spatial axis when given. The default
        field interpolation is ``"bilinear"`` for 2D and ``"trilinear"`` for
        3D; ``"bspline"`` is valid for either rank. A nonzero
        ``locked_borders`` value is measured in coarse-grid layers. When
        ``displacement_units="mm"``, every call must provide a validated 4x4
        affine in ``bundle.meta["affine"]`` and a valid
        ``minimum_physical_spacing`` bound must have been configured.

    Examples:
        A 2D sample-level TensorFlow pipeline.

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import RandomElasticTransform

            image = tf.random.normal((224, 224, 3), seed=101)
            transform = RandomElasticTransform(
                keys=["image"],
                input_layout="HWC",
                alpha=4.0,
                sigma=6.0,
                control_grid_spacing=(16, 16),
                field_interpolation="bilinear",
                seed=101,
            )
            result = transform({"image": image})

        A 2D batch-level Torch pipeline using a coarse B-spline field.

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import RandomElasticTransform

            images = torch.randn((8, 224, 224, 3))
            transform = RandomElasticTransform(
                keys=["image"],
                input_layout="BHWC",
                alpha=(2.0, 5.0),
                sigma=(4.0, 8.0),
                control_grid_spacing=(16, 16),
                field_interpolation="bspline",
                prob=0.8,
                seed=102,
            )
            result = transform({"image": images})

        A 3D sample-level JAX pipeline with trilinear coarse-field expansion.

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            from medicai.transforms import RandomElasticTransform

            volume = jax.random.normal(jax.random.PRNGKey(103), (96, 128, 128, 1))
            transform = RandomElasticTransform(
                keys=["image"],
                input_layout="DHWC",
                alpha=3.0,
                sigma=5.0,
                control_grid_spacing=(8, 8, 8),
                field_interpolation="trilinear",
                seed=103,
            )
            result = transform({"image": volume})

        A 3D batch-level TensorFlow segmentation pipeline. The same sampled
        field is shared by the image and label keys, while their value
        interpolation remains different.

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import RandomElasticTransform

            image = tf.random.normal((2, 64, 96, 96, 1), seed=104)
            label = tf.random.uniform(
                (2, 64, 96, 96, 1), maxval=2, dtype=tf.int32, seed=104
            )
            transform = RandomElasticTransform(
                keys=["image", "label"],
                input_layout="BDHWC",
                interpolation={"image": "trilinear", "label": "nearest"},
                control_grid_spacing=(8, 8, 8),
                locked_borders=1,
                seed=104,
            )
            result = transform({"image": image, "label": label})

        A Torch 3D pipeline with constant image-domain padding and a B-spline
        deformation field.

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import RandomElasticTransform

            volume = torch.randn((1, 160, 256, 256, 1))
            transform = RandomElasticTransform(
                keys=["image"],
                input_layout="BDHWC",
                alpha=6.0,
                sigma=8.0,
                control_grid_spacing=(16, 16, 16),
                field_interpolation="bspline",
                fill_mode="constant",
                fill_value=0.0,
                seed=105,
            )
            result = transform({"image": volume})

        A JAX 2D pipeline using the full-resolution default field. No coarse
        grid is created when ``control_grid_spacing`` is ``None``.

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            from medicai.transforms import RandomElasticTransform

            images = jax.random.normal(jax.random.PRNGKey(106), (4, 512, 512, 1))
            transform = RandomElasticTransform(
                keys=["image"],
                input_layout="BHWC",
                alpha=2.0,
                sigma=3.0,
                field_interpolation="bspline",
                control_grid_spacing=None,
                prob=0.5,
                seed=106,
            )
            result = transform({"image": images})

        A 3D physical-unit pipeline. The affine supplies the voxel spacing;
        ``alpha`` and ``sigma`` are interpreted in millimeters.

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import RandomElasticTransform

            volume = tf.random.normal((64, 96, 96, 1), seed=107)
            affine = tf.constant(
                [
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.8, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=tf.float32,
            )
            transform = RandomElasticTransform(
                keys=["image"],
                input_layout="DHWC",
                alpha=(1.0, 3.0),
                sigma=4.0,
                displacement_units="mm",
                minimum_physical_spacing=(2.0, 1.0, 0.8),
                control_grid_spacing=(8, 8, 8),
                locked_borders=1,
                seed=107,
            )
            result = transform({"image": volume}, {"affine": affine})

        A 2D physical-unit Torch pipeline. For ``HWC`` and ``BHWC``, the
        first two affine spacing entries correspond to the tensor's two
        spatial axes.

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import RandomElasticTransform

            images = torch.randn((4, 224, 224, 3))
            affine = torch.diag(torch.tensor([0.7, 0.7, 1.0, 1.0]))
            transform = RandomElasticTransform(
                keys=["image"],
                input_layout="BHWC",
                alpha=2.0,
                sigma=3.0,
                displacement_units="mm",
                minimum_physical_spacing=0.7,
                field_interpolation="bspline",
                seed=108,
            )
            result = transform({"image": images}, {"affine": affine})

    """

    def __init__(
        self,
        keys: Sequence[str],
        alpha: float | Sequence[float] = 20.0,
        sigma: float | Sequence[float] = 4.0,
        interpolation: str | Sequence[str] | Mapping[str, str] | None = None,
        prob: float = 0.1,
        *,
        input_layout: str,
        control_grid_spacing: int | Sequence[int] | None = None,
        displacement_units: str = "voxel",
        minimum_physical_spacing: float | Sequence[float] | None = None,
        field_interpolation: str | None = None,
        fill_mode: str = "nearest",
        fill_value: float = 0.0,
        locked_borders: int = 0,
        seed: int | keras.random.SeedGenerator | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob, seed=seed)
        if not keys:
            raise ValueError("`keys` must contain at least one tensor key.")
        self.keys = tuple(keys)
        self.alpha = self._normalize_parameter_range(alpha, "alpha", 0.0)
        self.sigma = self._normalize_parameter_range(sigma, "sigma", 1e-6)
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            transform_name=type(self).__name__,
        )
        self.layout_info = get_input_layout_info(self.input_layout)
        self.control_grid_spacing = self._normalize_control_grid_spacing(control_grid_spacing)
        if displacement_units not in {"voxel", "mm"}:
            raise ValueError("`displacement_units` must be either 'voxel' or 'mm'.")
        self.displacement_units = displacement_units
        self.minimum_physical_spacing = self._normalize_physical_spacing(minimum_physical_spacing)
        if displacement_units == "mm" and self.minimum_physical_spacing is None:
            raise ValueError(
                "`minimum_physical_spacing` is required when " "displacement_units='mm'."
            )
        if field_interpolation is None:
            field_interpolation = "bilinear" if self.layout_info.spatial_rank == 2 else "trilinear"
        allowed_field_interpolations = (
            {"bilinear", "bspline"}
            if self.layout_info.spatial_rank == 2
            else {"trilinear", "bspline"}
        )
        if field_interpolation not in allowed_field_interpolations:
            raise ValueError(
                f"`field_interpolation`={field_interpolation!r} is invalid for "
                f"{self.layout_info.spatial_rank}D input. Allowed values are "
                f"{sorted(allowed_field_interpolations)}."
            )
        self.field_interpolation = field_interpolation
        if not isinstance(locked_borders, int) or locked_borders < 0:
            raise ValueError("`locked_borders` must be a non-negative integer.")
        self.locked_borders = locked_borders
        if fill_mode not in {"nearest", "constant", "reflect", "wrap"}:
            raise ValueError(
                "`fill_mode` must be one of 'nearest', 'constant', 'reflect', or 'wrap'."
            )
        if not isinstance(fill_value, Number):
            raise TypeError("`fill_value` must be numeric.")
        self.fill_mode = fill_mode
        self.fill_value = float(fill_value)
        self.interpolation = self._normalize_interpolation(interpolation)
        self.allow_missing_keys = allow_missing_keys

    def _normalize_control_grid_spacing(
        self,
        spacing: int | Sequence[int] | None,
    ) -> tuple[int, ...] | None:
        if spacing is None:
            return None
        if isinstance(spacing, int):
            values = (spacing,) * self.layout_info.spatial_rank
        elif isinstance(spacing, (tuple, list)):
            if len(spacing) != self.layout_info.spatial_rank:
                raise ValueError("`control_grid_spacing` must contain one value per spatial axis.")
            values = tuple(spacing)
        else:
            raise TypeError("`control_grid_spacing` must be an int, sequence, or None.")
        if any(not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("`control_grid_spacing` values must be positive integers.")
        return values

    def _normalize_physical_spacing(
        self,
        spacing: float | Sequence[float] | None,
    ) -> tuple[float, ...] | None:
        if spacing is None:
            return None
        if isinstance(spacing, Number):
            values = (float(spacing),) * self.layout_info.spatial_rank
        elif isinstance(spacing, (tuple, list)):
            if len(spacing) != self.layout_info.spatial_rank:
                raise ValueError(
                    "`minimum_physical_spacing` must contain one value per " "spatial axis."
                )
            values = tuple(float(value) for value in spacing)
        else:
            raise TypeError("`minimum_physical_spacing` must be a number, sequence, or None.")
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("`minimum_physical_spacing` values must be finite and positive.")
        return values

    def _normalize_parameter_range(
        self,
        value: float | Sequence[float],
        name: str,
        minimum: float,
    ) -> tuple[float, float]:
        if isinstance(value, Number):
            bounds = (float(value), float(value))
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            bounds = (float(value[0]), float(value[1]))
        else:
            raise TypeError(f"`{name}` must be a number or a two-value range.")
        if bounds[0] < minimum or bounds[1] < minimum or bounds[0] > bounds[1]:
            raise ValueError(
                f"`{name}` must satisfy {name}[0] <= {name}[1] and both values "
                f"must be >= {minimum}. Received {value!r}."
            )
        return bounds

    def _sample_parameter(
        self,
        bounds: tuple[float, float],
        batch_size: Any | None = None,
    ) -> Any:
        shape = () if batch_size is None else (batch_size,)
        if bounds[0] == bounds[1]:
            if batch_size is None:
                return bounds[0]
            return ops.full(shape, bounds[0], dtype="float32")
        return self.random_uniform(
            shape=shape,
            minval=bounds[0],
            maxval=bounds[1],
            dtype="float32",
        )

    def _sample_apply_mask(self, batch_size: Any) -> Any:
        """Sample one independent application decision for each batch item."""
        return (
            self.random_uniform(
                shape=(batch_size,),
                minval=0.0,
                maxval=1.0,
                dtype="float32",
            )
            < self.prob
        )

    def _normalize_interpolation(
        self,
        interpolation: str | Sequence[str] | Mapping[str, str] | None,
    ) -> dict[str, str]:
        if interpolation is None:
            linear_mode = "bilinear" if self.layout_info.spatial_rank == 2 else "trilinear"
            result = {
                key: linear_mode if index == 0 else "nearest" for index, key in enumerate(self.keys)
            }
        elif isinstance(interpolation, str):
            result = {key: interpolation for key in self.keys}
        elif isinstance(interpolation, Mapping):
            missing = set(self.keys) - set(interpolation)
            if missing:
                raise ValueError(f"Missing interpolation mode for keys: {sorted(missing)}")
            result = {key: interpolation[key] for key in self.keys}
        elif isinstance(interpolation, (tuple, list)):
            if len(interpolation) != len(self.keys):
                raise ValueError("Length of `interpolation` must match length of `keys`.")
            result = dict(zip(self.keys, interpolation, strict=True))
        else:
            raise TypeError("`interpolation` must be a string, sequence, or mapping.")

        valid = (
            {"nearest", "bilinear"}
            if self.layout_info.spatial_rank == 2
            else {
                "nearest",
                "trilinear",
            }
        )
        for key, mode in result.items():
            if mode not in valid:
                raise ValueError(
                    f"Invalid interpolation {mode!r} for {self.layout_info.spatial_rank}D "
                    f"input (key={key!r}). Allowed: {sorted(valid)}."
                )
        return result

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        affine = bundle.meta.get("affine")
        if self.displacement_units == "mm":
            if affine is None:
                raise ValueError(
                    "RandomElasticTransform with displacement_units='mm' "
                    "requires bundle.meta['affine'] containing a 4x4 affine matrix."
                )
            affine = validate_affine_matrix(affine)
        missing_keys = [key for key in self.keys if key not in bundle.data]
        if missing_keys and not self.allow_missing_keys:
            raise KeyError(f"Key {missing_keys[0]!r} not found in input data.")
        present_keys = [key for key in self.keys if key in bundle.data]
        reference = bundle.data[present_keys[0]] if present_keys else None
        batch_size = ops.shape(reference)[0] if present_keys and self.layout_info.batched else 1
        should_apply = self._sample_apply_mask(batch_size)
        params = {
            "keys": list(present_keys),
            "alpha": self.alpha,
            "sigma": self.sigma,
            "control_grid_spacing": self.control_grid_spacing,
            "displacement_units": self.displacement_units,
            "minimum_physical_spacing": self.minimum_physical_spacing,
            "field_interpolation": self.field_interpolation,
            "locked_borders": self.locked_borders,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": dict(self.interpolation),
            "input_layout": self.input_layout,
            "should_apply": should_apply,
        }
        if not present_keys:
            params["should_apply"] = ops.zeros((batch_size,), dtype="bool")
            self.record_random_transform(bundle, params=params, applied=False)
            return bundle

        reference = bundle.data[present_keys[0]]
        validate_tensor_matches_layout(
            reference,
            self.input_layout,
            transform_name=type(self).__name__,
        )
        batched, _ = ensure_batch_axis_for_layout(
            reference,
            input_layout=self.input_layout,
        )
        field = self._sample_or_zero_field(
            batched,
            params["should_apply"],
            affine=affine,
        )

        for key in present_keys:
            tensor = bundle.data[key]
            batched_tensor, added_batch_axis = ensure_batch_axis_for_layout(
                tensor,
                input_layout=self.input_layout,
            )
            transformed = self._warp_tensor(
                batched_tensor,
                field,
                self.interpolation[key],
            )
            apply_shape = [ops.shape(batched_tensor)[0]] + [1] * (self.layout_info.spatial_rank + 1)
            transformed = ops.where(
                ops.reshape(ops.cast(params["should_apply"], "bool"), apply_shape),
                transformed,
                batched_tensor,
            )
            bundle.data[key] = restore_from_batch_axis(transformed, added_batch_axis)

        self.record_random_transform(
            bundle,
            params=params,
            applied=ops.any(params["should_apply"]),
            kernel=type(self).__name__,
        )
        return bundle

    def _sample_or_zero_field(
        self,
        tensor: Any,
        should_apply: Any,
        *,
        affine: Any | None = None,
    ) -> Any:
        shape = ops.shape(tensor)
        spatial_rank = self.layout_info.spatial_rank
        spatial_shape = self._static_spatial_shape(tensor)
        spacing = self.control_grid_spacing or (1,) * spatial_rank
        coarse_shape = tuple(
            max(1, (size + step - 1) // step)
            for size, step in zip(spatial_shape, spacing, strict=True)
        )
        coarse_field_shape = [shape[0]] + list(coarse_shape) + [spatial_rank]
        output_field_shape = [shape[0]] + list(spatial_shape) + [spatial_rank]

        def sample_field():
            noise = self.random_normal(shape=coarse_field_shape, dtype="float32")
            alpha = self._sample_parameter(self.alpha, shape[0])
            sigma = self._sample_parameter(self.sigma, shape[0])
            if self.displacement_units == "mm":
                physical_spacing = self._physical_spacing(affine)
                coarse_physical_spacing = physical_spacing * ops.cast(
                    spacing,
                    physical_spacing.dtype,
                )
                smooth_sigma = ops.reshape(sigma, [shape[0], 1]) / coarse_physical_spacing
                # The runtime affine controls the actual smoothing widths. The
                # configured per-axis lower bounds provide static radii for graphs.
                min_coarse_spacing = tuple(
                    bound * step
                    for bound, step in zip(self.minimum_physical_spacing, spacing, strict=True)
                )
                max_smooth_sigma = tuple(
                    self.sigma[1] / axis_spacing for axis_spacing in min_coarse_spacing
                )
            else:
                smooth_sigma = sigma / min(spacing)
                max_smooth_sigma = self.sigma[1] / min(spacing)
            if noise.shape[0] == 1:
                field = _gaussian_smooth_nd(
                    noise,
                    ops.maximum(smooth_sigma[0], 1e-3),
                    spatial_rank,
                    max_sigma=max_smooth_sigma,
                )
            else:

                field = _gaussian_smooth_broadcast_nd(
                    noise,
                    ops.maximum(smooth_sigma, 1e-3),
                    spatial_rank,
                    max_sigma=max_smooth_sigma,
                )
            field = _lock_field_borders(field, self.locked_borders, spatial_rank)
            if spacing != (1,) * spatial_rank:
                field = resample_displacement_field(
                    field,
                    target_shape=spatial_shape,
                    interpolation=self.field_interpolation,
                    align_corners=False,
                )
            reduction_axes = tuple(range(1, spatial_rank + 2))
            peak = ops.max(ops.abs(field), axis=reduction_axes, keepdims=True)
            safe_peak = ops.where(peak > 1e-6, peak, ops.ones_like(peak))
            alpha_shape = [shape[0]] + [1] * (spatial_rank + 1)
            alpha = ops.reshape(alpha, alpha_shape)
            field = (field / safe_peak) * alpha
            if self.displacement_units == "mm":
                voxel_spacing = self._physical_spacing(affine)
                field = field / voxel_spacing[None, ...]
                return ops.clip(
                    field,
                    -alpha / voxel_spacing[None, ...],
                    alpha / voxel_spacing[None, ...],
                )
            return ops.clip(field, -alpha, alpha)

        # Keep random sampling outside conditional control flow. JAX does not
        # allow the stateful Keras SeedGenerator to escape an ``ops.cond``
        # branch while tracing a data-loader or compiled training function.
        sampled_field = sample_field()
        zero_field = ops.zeros(output_field_shape, dtype="float32")
        apply_mask = ops.cast(should_apply, "bool")
        if getattr(apply_mask, "shape", None) is not None and len(apply_mask.shape) == 0:
            apply_mask = ops.broadcast_to(
                ops.reshape(apply_mask, (1,)),
                (shape[0],),
            )
        apply_shape = [shape[0]] + [1] * (spatial_rank + 1)
        return ops.where(
            ops.reshape(apply_mask, apply_shape),
            sampled_field,
            zero_field,
        )

    def _physical_spacing(self, affine: Any | None) -> Any:
        """Return physical spacing aligned with the tensor spatial axes."""
        if affine is None:
            raise ValueError("An affine matrix is required for displacement_units='mm'.")
        spacing = spacing_from_affine(affine)
        return spacing[: self.layout_info.spatial_rank]

    def _static_spatial_shape(self, tensor: Any) -> tuple[int, ...]:
        spatial_rank = self.layout_info.spatial_rank
        spatial_shape = tuple(tensor.shape[1 : spatial_rank + 1])
        if any(size is None for size in spatial_shape):
            raise ValueError(
                "RandomElasticTransform requires statically known spatial dimensions "
                "to construct its deformation field."
            )
        return tuple(int(size) for size in spatial_shape)

    def _warp_tensor(self, tensor: Any, field: Any, interpolation: str) -> Any:
        shape = ops.shape(tensor)
        spatial_rank = self.layout_info.spatial_rank
        spatial_shape = [shape[index + 1] for index in range(spatial_rank)]
        ranges = [ops.arange(size, dtype="float32") for size in spatial_shape]
        mesh = ops.meshgrid(*ranges, indexing="ij")
        grid = ops.cast(ops.stack(mesh, axis=-1), field.dtype)
        coordinates = grid[None, ...] + field
        if interpolation == "nearest":
            return _nearest_sample(
                tensor,
                coordinates,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
            )
        return _linear_sample(
            tensor,
            coordinates,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )

"""Backend-neutral batched random rotation kernels.

The implementation has two 3D paths. Single-axis rotations fold the batch and
the untouched spatial axis into one 2D affine-transform batch. Multi-axis
rotations build a batched 3x3 matrix and use coordinate sampling for genuine
3D linear interpolation. Both paths preserve channel-last layouts and use
Keras random streams so they remain dispatchable across supported backends.
"""

from typing import Any, Sequence

import keras
from keras import ops

from ..base import RandomTransform, _normalize_keys, _pop_last_transform_trace
from ..tensor_bundle import TensorBundle
from ..utils import (
    ensure_batch_axis_for_layout,
    get_input_layout_info,
    resolve_input_layout,
    restore_from_batch_axis,
    validate_tensor_matches_layout,
)

AXES = ("D", "H", "W")
FILL_MODES = ("constant", "nearest", "wrap", "mirror", "reflect")


def _as_range(value: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Each rotation range must contain exactly two values.")
        return float(value[0]), float(value[1])
    if float(value) < 0:
        raise ValueError(f"`factor` must be non-negative. Received {value}.")
    return -float(value), float(value)


def _resolve_axis_ranges(factor: float | Sequence[float] | dict[str, Any]):
    if isinstance(factor, dict):
        ranges = {}
        for axis, value in factor.items():
            axis = str(axis).upper()
            if axis not in AXES:
                raise ValueError(f"Rotation axes must be drawn from {AXES}. Received {axis!r}.")
            ranges[axis] = _as_range(value)
        return ranges
    return {"D": _as_range(factor)}


def _apply_anisotropy_policy(ranges, spacing, threshold):
    """Restrict 3D rotations to the coarsest axis for highly anisotropic data."""
    if spacing is None or max(spacing) / min(spacing) <= threshold:
        return dict(ranges)
    coarse_axis = AXES[list(spacing).index(max(spacing))]
    retained = {axis: value for axis, value in ranges.items() if axis == coarse_axis}
    if not retained:
        raise ValueError(
            f"`spacing` is anisotropic beyond `anisotropy_threshold`, so rotation is "
            f"restricted to axis {coarse_axis!r}, but `factor` configures "
            f"{sorted(ranges)}. Configure a range for {coarse_axis!r}."
        )
    return retained


def _resolve_per_key(keys, value, default_fn, name):
    if value is None:
        return {key: default_fn(key, index) for index, key in enumerate(keys)}
    if isinstance(value, dict):
        missing = [key for key in keys if key not in value]
        if missing:
            raise ValueError(f"`{name}` is missing entries for keys: {missing}.")
        return {key: value[key] for key in keys}
    if isinstance(value, (tuple, list)):
        if len(value) != len(keys):
            raise ValueError(f"`{name}` must have one value per key.")
        return dict(zip(keys, value))
    return {key: value for key in keys}


def _rotation_matrix_2d(angle: Any, height: Any, width: Any) -> Any:
    height = ops.cast(height - 1, angle.dtype)
    width = ops.cast(width - 1, angle.dtype)
    y0, x0 = height / 2.0, width / 2.0
    cos_a, sin_a = ops.cos(angle), ops.sin(angle)
    zeros = ops.zeros_like(angle)
    return ops.stack(
        [
            cos_a,
            sin_a,
            x0 - x0 * cos_a - y0 * sin_a,
            -sin_a,
            cos_a,
            y0 - y0 * cos_a + x0 * sin_a,
            zeros,
            zeros,
        ],
        axis=-1,
    )


def rotate_2d(
    images: Any,
    angles: Any,
    interpolation: str = "bilinear",
    fill_mode: str = "constant",
    fill_value: float = 0.0,
) -> Any:
    """Rotate a ``(B, H, W, C)`` batch with Keras' affine image kernel.

    The angle is a ``(B,)`` tensor, allowing each batch item to receive its
    own sampled rotation while preserving one shared kernel invocation.
    """
    original_dtype = images.dtype
    images = ops.cast(images, "float32")
    shape = ops.shape(images)
    matrix = _rotation_matrix_2d(angles, shape[1], shape[2])
    rotated = ops.image.affine_transform(
        images,
        matrix,
        interpolation=interpolation.lower(),
        fill_mode=fill_mode,
        fill_value=fill_value,
    )
    return ops.cast(rotated, original_dtype)


def _plane_dims(shape, axis):
    _, depth, height, width, _ = shape
    if axis == "D":
        return (height, width), depth
    if axis == "H":
        return (depth, width), height
    if axis == "W":
        return (depth, height), width
    raise ValueError(axis)


def rotate_single_axis(
    volumes: Any,
    angles: Any,
    axis: str,
    interpolation: str = "bilinear",
    fill_mode: str = "constant",
    fill_value: float = 0.0,
) -> Any:
    """Rotate a ``(B, D, H, W, C)`` batch about one 3D axis.

    The untouched spatial axis is folded into the affine kernel's batch axis,
    so every 2D plane for one volume receives the same angle.
    """
    original_dtype = volumes.dtype
    volumes = ops.cast(volumes, "float32")
    batch, depth, height, width, channels = volumes.shape
    if channels is None:
        raise ValueError("RandomRotate requires a statically known channel dimension.")
    (dim0, dim1), folded = _plane_dims((batch, depth, height, width, channels), axis)

    if axis == "D":
        transposed = volumes
    elif axis == "H":
        transposed = ops.transpose(volumes, (0, 2, 1, 3, 4))
    else:
        transposed = ops.transpose(volumes, (0, 3, 1, 2, 4))

    merged = ops.reshape(transposed, (-1, dim0, dim1, channels))
    repeated_angles = ops.repeat(angles, folded, axis=0)
    matrices = _rotation_matrix_2d(repeated_angles, dim0, dim1)
    rotated = ops.image.affine_transform(
        merged,
        matrices,
        interpolation=interpolation.lower(),
        fill_mode=fill_mode,
        fill_value=fill_value,
    )
    rotated = ops.reshape(rotated, (-1, folded, dim0, dim1, channels))

    if axis == "H":
        rotated = ops.transpose(rotated, (0, 2, 1, 3, 4))
    elif axis == "W":
        rotated = ops.transpose(rotated, (0, 2, 3, 1, 4))
    return ops.cast(rotated, original_dtype)


def _rotation_matrix_3d(angle_d, angle_h, angle_w):
    """Compose batched D, H, and W rotations in voxel-axis order."""
    one, zero = ops.ones_like(angle_d), ops.zeros_like(angle_d)
    cd, sd = ops.cos(angle_d), ops.sin(angle_d)
    ch, sh = ops.cos(angle_h), ops.sin(angle_h)
    cw, sw = ops.cos(angle_w), ops.sin(angle_w)
    rotation_d = ops.stack(
        [
            ops.stack([one, zero, zero], axis=-1),
            ops.stack([zero, cd, sd], axis=-1),
            ops.stack([zero, -sd, cd], axis=-1),
        ],
        axis=-2,
    )
    rotation_h = ops.stack(
        [
            ops.stack([ch, zero, sh], axis=-1),
            ops.stack([zero, one, zero], axis=-1),
            ops.stack([-sh, zero, ch], axis=-1),
        ],
        axis=-2,
    )
    rotation_w = ops.stack(
        [
            ops.stack([cw, sw, zero], axis=-1),
            ops.stack([-sw, cw, zero], axis=-1),
            ops.stack([zero, zero, one], axis=-1),
        ],
        axis=-2,
    )
    return rotation_w @ rotation_h @ rotation_d


def _spacing_scale_matrix(spacing, dtype):
    """Build the voxel-to-physical spacing correction for 3D rotations."""
    spacing = ops.convert_to_tensor(spacing, dtype=dtype)
    return ops.outer(1.0 / spacing, spacing)


def _rotate_one_volume(volume, inverse_matrix, interpolation, fill_mode, fill_value):
    """Sample one volume using an output-to-input 3D coordinate matrix.

    ``interpolation="bilinear"`` is the public Keras image vocabulary; with
    three coordinate axes this corresponds to order-1, commonly called
    trilinear interpolation. ``nearest`` uses order 0 for discrete labels.
    """
    depth, height, width, channels = volume.shape
    if channels is None:
        raise ValueError("RandomRotate requires a statically known channel dimension.")
    z, y, x = ops.meshgrid(ops.arange(depth), ops.arange(height), ops.arange(width), indexing="ij")
    coordinates = ops.stack(
        [
            ops.cast(z, inverse_matrix.dtype),
            ops.cast(y, inverse_matrix.dtype),
            ops.cast(x, inverse_matrix.dtype),
        ],
        axis=0,
    )
    center = (
        ops.cast(ops.convert_to_tensor([depth - 1, height - 1, width - 1]), inverse_matrix.dtype)
        / 2.0
    )
    centered = coordinates - ops.reshape(center, (3, 1, 1, 1))
    input_coordinates = ops.einsum("ij,jdhw->idhw", inverse_matrix, centered)
    input_coordinates = input_coordinates + ops.reshape(center, (3, 1, 1, 1))
    channels_out = []
    order = 1 if interpolation.lower() == "bilinear" else 0
    for channel in range(channels):
        channels_out.append(
            ops.image.map_coordinates(
                volume[..., channel],
                input_coordinates,
                order=order,
                fill_mode=fill_mode,
                fill_value=fill_value,
            )
        )
    return ops.stack(channels_out, axis=-1)


def rotate_multi_axis(
    volumes,
    angle_d,
    angle_h,
    angle_w,
    spacing=None,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0.0,
    precomputed_matrix=None,
):
    """Rotate a ``(B, D, H, W, C)`` batch with batched 3D sampling.

    ``precomputed_matrix`` is used by inverse paths so the exact sampled
    forward matrix is reused instead of reconstructing it from angles.
    """
    original_dtype = volumes.dtype
    volumes = ops.cast(volumes, "float32")
    matrix = (
        precomputed_matrix
        if precomputed_matrix is not None
        else _rotation_matrix_3d(angle_d, angle_h, angle_w)
    )
    inverse_matrix = ops.transpose(matrix, (0, 2, 1))
    if spacing is not None:
        inverse_matrix = (
            inverse_matrix * _spacing_scale_matrix(spacing, inverse_matrix.dtype)[None, :, :]
        )

    def rotate_one(args):
        volume, matrix_one = args
        return _rotate_one_volume(volume, matrix_one, interpolation, fill_mode, fill_value)

    rotated = ops.vectorized_map(rotate_one, (volumes, inverse_matrix))
    return ops.cast(rotated, original_dtype)


class RandomRotate(RandomTransform):
    """Apply random 2D or 3D rotations to channel-last sample or batch tensors.

    The transform accepts ``HWC``, ``DHWC``, ``BHWC``, and ``BDHWC`` layouts.
    A scalar or two-value ``factor`` rotates around the depth axis (the H-W
    plane). A dictionary can specify independent ranges for ``D``, ``H``, and
    ``W`` axes. In 2D, only the ``D`` rotation axis is valid.

    ``prob`` is sampled independently for each batch item. A skipped item gets
    zero angles and is therefore an exact identity. Selected keys share the
    same sampled angles so images, masks, and labels remain aligned.

    The forward operation is a resampling transform. ``inverse()`` reuses the
    recorded angles and reverses the geometry, but arbitrary-angle image
    interpolation can introduce small numerical differences; it is not an
    exact voxel-for-voxel round trip. ``fill_mode="constant"`` is typically
    the most predictable choice for medical images.

    .. note::

        On the TensorFlow backend, the affine image kernel used by
        ``RandomRotate`` is not currently XLA-compatible. Regular eager and
        ``tf.data`` graph execution remain supported, including GPU execution
        when TensorFlow places the kernel there. Use ``jit_compile=False`` for
        this transform rather than treating it as CPU-only.

    Args:
        keys: Tensor keys to rotate together.
        factor: A non-negative maximum angle, a ``(min, max)`` range, or a
            mapping from ``D``, ``H``, and ``W`` to either form. Angles are in
            radians.
        prob: Per-sample probability of applying the rotation.
        spacing: Optional 3D voxel spacing used for physical-space correction.
        anisotropy_threshold: If the largest spacing divided by the smallest
            spacing exceeds this value, restrict rotation to the coarsest
            physical axis. This applies only when ``spacing`` is provided.
        interpolation: One mode, one mode per key, or a key-to-mode mapping.
            Supported modes are ``"bilinear"`` and ``"nearest"``.
        fill_mode: One Keras image fill mode: ``"constant"``, ``"nearest"`,
            ``"wrap"``, ``"mirror"``, or ``"reflect"``.
        fill_value: One value, one value per key, or a key-to-value mapping.
        input_layout: One of ``HWC``, ``DHWC``, ``BHWC``, or ``BDHWC``.
        seed: Optional integer or ``keras.random.SeedGenerator``.
        allow_missing_keys: If ``True``, missing requested keys are skipped.

    Example:

        TensorFlow backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import RandomRotate

            transform = RandomRotate(
                keys=["image"],
                factor=0.1,
                prob=0.5,
                input_layout="HWC",
            )
            image = tf.random.normal((64, 64, 1), seed=7)
            result = transform({"image": image})
            print(result["image"].shape)

        JAX backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            from medicai.transforms import RandomRotate

            transform = RandomRotate(
                keys=["image"],
                factor={"H": 0.1, "W": 0.1},
                prob=0.5,
                input_layout="DHWC",
            )
            image = jax.random.normal(
                jax.random.PRNGKey(7), shape=(32, 64, 64, 1)
            )
            result = transform({"image": image})
            print(result["image"].shape)

        Torch backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import RandomRotate

            transform = RandomRotate(
                keys=["image"],
                factor=0.1,
                prob=0.5,
                input_layout="BHWC",
            )
            torch.manual_seed(7)
            batch = torch.randn((2, 64, 64, 1))
            result = transform({"image": batch})
            print(result["image"].shape)

    Notes:
        The trace stores sampled per-item angles in bundle metadata, so inverse
        execution can work after the model replaces transformed tensors with
        predictions. Reuse the returned ``TensorBundle`` when calling
        ``inverse()``.
    """

    def __init__(
        self,
        keys: Sequence[str],
        factor: float | Sequence[float] | dict[str, Any] = 0.1,
        prob: float = 0.8,
        spacing: Sequence[float] | None = None,
        anisotropy_threshold: float = 3.0,
        interpolation=None,
        fill_mode="constant",
        fill_value=None,
        *,
        input_layout: str,
        seed: int | keras.random.SeedGenerator | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob, seed=seed)
        self.keys = _normalize_keys(keys)
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            allowed_layouts=("HWC", "DHWC", "BHWC", "BDHWC"),
            transform_name=type(self).__name__,
        )
        self.layout_info = get_input_layout_info(self.input_layout)
        self.allow_missing_keys = allow_missing_keys
        self.ranges = _resolve_axis_ranges(factor)
        if any(low > high for low, high in self.ranges.values()):
            raise ValueError("Each rotation range must have lower bound <= upper bound.")
        if self.layout_info.spatial_rank == 2 and set(self.ranges) != {"D"}:
            raise ValueError("2D RandomRotate supports only the `D` rotation axis.")
        if spacing is not None:
            if self.layout_info.spatial_rank != 3 or len(spacing) != 3:
                raise ValueError(
                    "`spacing` is supported only for 3D layouts and must have length 3."
                )
            spacing = tuple(float(value) for value in spacing)
            if any(value <= 0 for value in spacing):
                raise ValueError("`spacing` values must be positive.")
        self.spacing = spacing
        self.anisotropy_threshold = float(anisotropy_threshold)
        self.ranges = _apply_anisotropy_policy(self.ranges, self.spacing, self.anisotropy_threshold)
        self.interpolation = _resolve_per_key(
            self.keys,
            interpolation,
            lambda _, index: "bilinear" if index == 0 else "nearest",
            "interpolation",
        )
        self.fill_mode = _resolve_per_key(self.keys, fill_mode, lambda *_: "constant", "fill_mode")
        self.fill_value = _resolve_per_key(self.keys, fill_value, lambda *_: 0.0, "fill_value")
        for key in self.keys:
            mode = str(self.interpolation[key]).lower()
            if mode not in ("bilinear", "nearest"):
                raise ValueError(f"Unsupported interpolation for key {key!r}.")
            fill_mode_key = str(self.fill_mode[key]).lower()
            if fill_mode_key not in FILL_MODES:
                raise ValueError(
                    f"Unsupported fill_mode {fill_mode_key!r}; use one of {FILL_MODES}."
                )
            self.interpolation[key] = mode
            self.fill_mode[key] = fill_mode_key

    @property
    def invertible(self) -> bool:
        return True

    def _sample_angles(self, batch_size, dtype="float32"):
        apply_mask = ops.cast(
            self.random_uniform(shape=(batch_size,), minval=0.0, maxval=1.0, dtype="float32")
            < self.prob,
            dtype,
        )
        angles = {}
        for axis, (low, high) in self.ranges.items():
            sampled = self.random_uniform(shape=(batch_size,), minval=low, maxval=high, dtype=dtype)
            angles[axis] = sampled * apply_mask
        return angles, ops.any(apply_mask > 0)

    def _apply_tensor(self, tensor, key, angles):
        if not angles:
            return tensor
        batched, added_batch = ensure_batch_axis_for_layout(
            tensor,
            input_layout=self.input_layout,
            allowed_spatial_ranks=(2, 3),
        )
        validate_tensor_matches_layout(
            batched,
            "BHWC" if self.layout_info.spatial_rank == 2 else "BDHWC",
            transform_name=type(self).__name__,
        )
        batch_size = ops.shape(batched)[0]
        angle_d = angles.get("D")
        if angle_d is None:
            angle_d = ops.zeros((batch_size,), dtype="float32")
        zero = ops.zeros_like(angle_d)
        if self.layout_info.spatial_rank == 2:
            rotated = rotate_2d(
                batched,
                angle_d,
                interpolation=self.interpolation[key],
                fill_mode=self.fill_mode[key],
                fill_value=self.fill_value[key],
            )
        else:
            active = [axis for axis in AXES if axis in angles]
            # Torch uses coordinate sampling for single-axis 3D rotations
            # because its affine kernel does not cover this case.
            if len(active) == 1 and keras.config.backend() != "torch":
                rotated = rotate_single_axis(
                    batched,
                    angles[active[0]],
                    active[0],
                    interpolation=self.interpolation[key],
                    fill_mode=self.fill_mode[key],
                    fill_value=self.fill_value[key],
                )
            else:
                rotated = rotate_multi_axis(
                    batched,
                    angle_d,
                    angles.get("H", zero),
                    angles.get("W", zero),
                    spacing=self.spacing,
                    interpolation=self.interpolation[key],
                    fill_mode=self.fill_mode[key],
                    fill_value=self.fill_value[key],
                )
        return restore_from_batch_axis(rotated, added_batch)

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present = []
        for key in self.keys:
            if key in bundle.data:
                present.append(key)
            elif not self.allow_missing_keys:
                raise KeyError(f"Key {key!r} not found in input data.")
        if not present:
            return bundle

        reference = bundle.data[present[0]]
        validate_tensor_matches_layout(
            reference, self.input_layout, transform_name=type(self).__name__
        )
        batch_size = ops.shape(reference)[0] if self.layout_info.batched else 1
        angles, applied = self._sample_angles(batch_size, dtype="float32")
        for key in present:
            bundle.data[key] = self._apply_tensor(bundle.data[key], key, angles)
        self.record_random_transform(
            bundle,
            params={
                "keys": present,
                "angles": angles,
                "input_layout": self.input_layout,
                "interpolation": dict(self.interpolation),
                "fill_mode": dict(self.fill_mode),
                "fill_value": dict(self.fill_value),
            },
            applied=applied,
            kernel="rotate_volume",
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = _pop_last_transform_trace(bundle, type(self).__name__)
        if trace is None:
            return bundle
        angles = trace["params"].get("angles", {})
        for key in trace["params"].get("keys", []):
            if key not in bundle.data:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"Key {key!r} not found in input data.")
            active = [axis for axis in AXES if axis in angles]
            if len(active) > 1 and self.layout_info.spatial_rank == 3:
                reference = bundle.data[key]
                batched, added_batch = ensure_batch_axis_for_layout(
                    reference,
                    input_layout=self.input_layout,
                    allowed_spatial_ranks=(3,),
                )
                batch_size = ops.shape(batched)[0]
                zero = ops.zeros((batch_size,), dtype="float32")
                # Skipped batch items are represented by zero angles in the trace;
                # rebuilding the matrix therefore preserves identity for them.
                forward_matrix = _rotation_matrix_3d(
                    angles.get("D", zero),
                    angles.get("H", zero),
                    angles.get("W", zero),
                )
                inverse_matrix = ops.transpose(forward_matrix, (0, 2, 1))
                restored = rotate_multi_axis(
                    batched,
                    zero,
                    zero,
                    zero,
                    spacing=self.spacing,
                    interpolation=self.interpolation[key],
                    fill_mode=self.fill_mode[key],
                    fill_value=self.fill_value[key],
                    precomputed_matrix=inverse_matrix,
                )
                bundle.data[key] = restore_from_batch_axis(restored, added_batch)
            else:
                inverse_angles = {axis: -value for axis, value in angles.items()}
                bundle.data[key] = self._apply_tensor(bundle.data[key], key, inverse_angles)
        return bundle

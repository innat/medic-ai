import itertools
from numbers import Number
from typing import Any, Mapping, Sequence

import keras
from keras import ops

from ..base import RandomTransform, _apply_if_applied
from ..tensor_bundle import TensorBundle
from ..utils import (
    ensure_batch_axis_for_layout,
    get_input_layout_info,
    resolve_input_layout,
    restore_from_batch_axis,
    validate_tensor_matches_layout,
)


def _gaussian_kernel_1d(sigma: float, radius: int, dtype: str = "float32") -> Any:
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


def _gaussian_smooth_nd(field: Any, sigma: float, spatial_rank: int) -> Any:
    """Smooth each displacement channel with separable Gaussian kernels."""
    radius = max(1, int(round(3.0 * sigma)))
    static_spatial_shape = tuple(field.shape[1 : spatial_rank + 1])
    if all(size is not None for size in static_spatial_shape):
        max_radius = min(int(size) - 1 for size in static_spatial_shape)
        if max_radius <= 0:
            return field
        radius = min(radius, max_radius)
    kernel = _gaussian_kernel_1d(sigma, radius, dtype="float32")
    shape = ops.shape(field)
    spatial_shape = [shape[index + 1] for index in range(spatial_rank)]
    channels = shape[-1]

    # Fold displacement channels into the batch dimension for independent
    # convolution, then restore the original channel-last field layout.
    permutation = [0, spatial_rank + 1] + list(range(1, spatial_rank + 1))
    folded = ops.transpose(field, permutation)
    folded = ops.reshape(folded, [-1] + spatial_shape + [1])
    for axis in range(spatial_rank):
        folded = _smooth_along_axis(folded, kernel, axis, radius, spatial_rank)

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


def _linear_sample(volume: Any, coordinates: Any) -> Any:
    """Sample a 2D or 3D channel-last volume with linear interpolation."""
    spatial_rank = coordinates.shape[-1]
    shape = ops.shape(volume)
    spatial_sizes = [ops.cast(shape[index + 1], volume.dtype) for index in range(spatial_rank)]
    floors = [ops.floor(coordinates[..., index]) for index in range(spatial_rank)]
    fractions = [
        (coordinates[..., index] - floors[index])[..., None]
        for index in range(spatial_rank)
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
    return output


def _nearest_sample(volume: Any, coordinates: Any) -> Any:
    """Sample a channel-last volume with nearest-neighbor interpolation."""
    spatial_rank = coordinates.shape[-1]
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
    return _flatten_gather(volume, batch_indices, indices)


class RandomElasticTransform(RandomTransform):
    """Apply random smooth elastic deformation to 2D or 3D tensors.

    One displacement field is sampled per batch item and shared by all
    selected keys, keeping aligned images and masks geometrically consistent.
    The initial implementation uses a full-resolution smoothed field; a
    coarse-grid 3D implementation is planned for large volumes.

    Args:
        keys: Keys of aligned tensors to deform.
        alpha: Maximum displacement magnitude in pixels or voxels.
        sigma: Gaussian smoothing width in pixels or voxels.
        interpolation: One interpolation mode, a sequence aligned with
            ``keys``, or a mapping from key to mode. Use ``"bilinear"`` for
            images and ``"nearest"`` for labels or masks. For 3D inputs,
            ``"trilinear"`` is the linear mode.
        prob: Probability of applying the deformation.
        input_layout: One of ``"HWC"``, ``"DHWC"``, ``"BHWC"``, or
            ``"BDHWC"``.
        seed: Optional integer or Keras ``SeedGenerator``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Note:
        The transform is not invertible. Elastic deformation does not have a
        generally exact inverse, so the sampled field is not used for inverse
        reconstruction.
    """

    def __init__(
        self,
        keys: Sequence[str],
        alpha: float = 20.0,
        sigma: float = 4.0,
        interpolation: str | Sequence[str] | Mapping[str, str] = "bilinear",
        prob: float = 0.1,
        *,
        input_layout: str,
        seed: int | keras.random.SeedGenerator | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob, seed=seed)
        if not keys:
            raise ValueError("`keys` must contain at least one tensor key.")
        if not isinstance(alpha, Number) or alpha < 0:
            raise ValueError(f"`alpha` must be a non-negative number. Received {alpha!r}.")
        if not isinstance(sigma, Number) or sigma <= 0:
            raise ValueError(f"`sigma` must be a positive number. Received {sigma!r}.")

        self.keys = tuple(keys)
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            transform_name=type(self).__name__,
        )
        self.layout_info = get_input_layout_info(self.input_layout)
        self.interpolation = self._normalize_interpolation(interpolation)
        self.allow_missing_keys = allow_missing_keys

    def _normalize_interpolation(
        self,
        interpolation: str | Sequence[str] | Mapping[str, str],
    ) -> dict[str, str]:
        if isinstance(interpolation, str):
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

        valid = {"nearest", "bilinear"} if self.layout_info.spatial_rank == 2 else {
            "nearest",
            "trilinear",
        }
        for key, mode in result.items():
            if mode not in valid:
                raise ValueError(
                    f"Invalid interpolation {mode!r} for {self.layout_info.spatial_rank}D "
                    f"input (key={key!r}). Allowed: {sorted(valid)}."
                )
        return result

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        missing_keys = [key for key in self.keys if key not in bundle.data]
        if missing_keys and not self.allow_missing_keys:
            raise KeyError(f"Key {missing_keys[0]!r} not found in input data.")
        present_keys = [key for key in self.keys if key in bundle.data]
        params = {
            "keys": list(present_keys),
            "alpha": self.alpha,
            "sigma": self.sigma,
            "interpolation": dict(self.interpolation),
            "input_layout": self.input_layout,
            "should_apply": self.sample_should_apply(),
        }
        if not present_keys:
            params["should_apply"] = False
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
        field = self._sample_or_zero_field(batched, params["should_apply"])

        for key in present_keys:
            tensor = bundle.data[key]
            batched_tensor, added_batch_axis = ensure_batch_axis_for_layout(
                tensor,
                input_layout=self.input_layout,
            )
            transformed = _apply_if_applied(
                params["should_apply"],
                lambda tensor=batched_tensor, key=key: self._warp_tensor(
                    tensor, field, self.interpolation[key]
                ),
                lambda tensor=batched_tensor: tensor,
            )
            bundle.data[key] = restore_from_batch_axis(transformed, added_batch_axis)

        self.record_random_transform(
            bundle,
            params=params,
            applied=params["should_apply"],
            kernel=type(self).__name__,
        )
        return bundle

    def _sample_or_zero_field(self, tensor: Any, should_apply: Any) -> Any:
        shape = ops.shape(tensor)
        spatial_rank = self.layout_info.spatial_rank
        field_shape = [shape[0]] + [shape[index + 1] for index in range(spatial_rank)]
        field_shape.append(spatial_rank)

        def sample_field():
            noise = self.random_normal(shape=field_shape, dtype="float32")
            return _gaussian_smooth_nd(noise, self.sigma, spatial_rank) * self.alpha

        return ops.cond(
            ops.cast(should_apply, "bool"),
            sample_field,
            lambda: ops.zeros(field_shape, dtype="float32"),
        )

    def _warp_tensor(self, tensor: Any, field: Any, interpolation: str) -> Any:
        shape = ops.shape(tensor)
        spatial_rank = self.layout_info.spatial_rank
        spatial_shape = [shape[index + 1] for index in range(spatial_rank)]
        ranges = [ops.arange(size, dtype="float32") for size in spatial_shape]
        mesh = ops.meshgrid(*ranges, indexing="ij")
        grid = ops.cast(ops.stack(mesh, axis=-1), field.dtype)
        coordinates = grid[None, ...] + field
        if interpolation == "nearest":
            return _nearest_sample(tensor, coordinates)
        return _linear_sample(tensor, coordinates)

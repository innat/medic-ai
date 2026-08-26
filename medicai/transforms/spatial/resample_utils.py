"""Affine-aware spatial resampling helpers for medical volumes."""

from typing import Any, Mapping

from keras import ops

from medicai.transforms.spatial.affine_utils import (
    affine_apply,
    build_affine,
    direction_from_affine,
    invert_affine,
    origin_from_affine,
)
from medicai.transforms.spatial.samplers import sample_volume
from medicai.transforms.utils import _get_static_shape_tuple, get_tensor_rank

# Resampling Utility


def _concrete_int_tuple(value: Any) -> tuple[int, ...] | None:
    """Return concrete integer values when a runtime shape is available."""
    try:
        return tuple(int(item) for item in ops.convert_to_numpy(value).reshape(-1))
    except (TypeError, ValueError, AttributeError):
        return None


def round_half_up(values: Any) -> Any:
    """Round floating-point values with half-up semantics."""
    return ops.floor(values + 0.5)


def compute_destination_affine(
    src_affine: Any,
    pixdim: Any,
    diagonal: bool = False,
) -> Any:
    """Compute a destination affine for resampling."""
    src_affine = ops.cast(src_affine, "float32")
    pixdim = ops.cast(pixdim, "float32")
    origin = origin_from_affine(src_affine)
    direction = ops.eye(3, dtype="float32") if diagonal else direction_from_affine(src_affine)
    return build_affine(pixdim, direction, origin)


def compute_output_shape(
    input_shape: Any,
    src_affine: Any,
    dst_affine: Any,
    align_corners: bool = False,
) -> Any:
    """Compute an output shape from source and destination geometry.

    The output extent is derived from source-volume corner coordinates mapped
    into the destination index space, so it remains correct for both
    axis-aligned and permuted affines.
    """
    input_shape = ops.cast(input_shape, "float32")
    src_affine = ops.cast(src_affine, "float32")
    dst_affine = ops.cast(dst_affine, "float32")

    if align_corners:
        max_corner = ops.maximum(input_shape - 1.0, 0.0)
        src_corners = ops.stack(
            ops.meshgrid(
                ops.stack([0.0, max_corner[0]]),
                ops.stack([0.0, max_corner[1]]),
                ops.stack([0.0, max_corner[2]]),
                indexing="ij",
            ),
            axis=-1,
        )
        src_corners = ops.reshape(src_corners, [-1, 3])
        dst_corners = affine_apply(ops.matmul(invert_affine(dst_affine), src_affine), src_corners)
        min_corner = ops.min(dst_corners, axis=0)
        max_corner = ops.max(dst_corners, axis=0)
        output_shape = round_half_up(ops.maximum(max_corner - min_corner, 0.0)) + 1.0
    else:
        max_corner = ops.maximum(input_shape, 0.0)
        src_corners = ops.stack(
            ops.meshgrid(
                ops.stack([0.0, max_corner[0]]),
                ops.stack([0.0, max_corner[1]]),
                ops.stack([0.0, max_corner[2]]),
                indexing="ij",
            ),
            axis=-1,
        )
        src_corners = ops.reshape(src_corners, [-1, 3])
        dst_corners = affine_apply(ops.matmul(invert_affine(dst_affine), src_affine), src_corners)
        min_corner = ops.min(dst_corners, axis=0)
        max_corner = ops.max(dst_corners, axis=0)
        output_shape = round_half_up(ops.maximum(max_corner - min_corner, 0.0))

    return ops.maximum(ops.cast(output_shape, "int32"), 1)


def make_output_grid(output_shape: Any) -> Any:
    """Create an output index grid shaped ``(N, 3)`` in ``(D, H, W)`` order."""
    output_shape = ops.cast(output_shape, "int32")
    d = ops.arange(output_shape[0], dtype="float32")
    h = ops.arange(output_shape[1], dtype="float32")
    w = ops.arange(output_shape[2], dtype="float32")
    dd, hh, ww = ops.meshgrid(d, h, w, indexing="ij")
    grid = ops.stack([dd, hh, ww], axis=-1)
    return ops.reshape(grid, [-1, 3])


def make_output_grid_chunk(
    output_shape: Any,
    start: Any,
    size: Any,
) -> Any:
    """Create one chunk of an output index grid from flat voxel indices.

    Args:
        output_shape: Spatial shape tensor shaped ``(3,)`` in ``(D, H, W)``
            order.
        start: Scalar flat-index offset into the full output grid.
        size: Number of grid points to generate.

    Returns:
        Tensor-like: Float32 grid chunk shaped ``(size, 3)`` in ``(D, H, W)``
        order.
    """
    output_shape = ops.cast(output_shape, "int32")
    start = ops.cast(start, "int32")
    size = ops.cast(size, "int32")

    flat = ops.arange(0, size, dtype="int32") + start
    hw = output_shape[1] * output_shape[2]
    d = ops.floor_divide(flat, hw)
    rem = ops.mod(flat, hw)
    h = ops.floor_divide(rem, output_shape[2])
    w = ops.mod(rem, output_shape[2])
    return ops.cast(ops.stack([d, h, w], axis=1), "float32")


class SpatialResample:
    """Internal affine-aware 3D resampling primitive."""

    def __init__(self, max_points_per_chunk: int = 65536):
        if max_points_per_chunk < 1:
            raise ValueError("`max_points_per_chunk` must be a positive integer.")
        self.max_points_per_chunk = int(max_points_per_chunk)

    def __call__(
        self,
        tensor: Any,
        src_affine: Any,
        dst_affine: Any,
        output_shape: Any,
        interpolation: str,
        padding_mode: str = "constant",
        fill_value: float = 0.0,
    ) -> Any:
        tensor = ops.convert_to_tensor(tensor)
        src_affine = ops.cast(src_affine, "float32")
        dst_affine = ops.cast(dst_affine, "float32")
        output_shape = ops.cast(output_shape, "int32")

        if get_tensor_rank(tensor) != 4:
            raise ValueError(
                f"Expected a 4D channel-last tensor shaped (D, H, W, C), got {tensor.shape}."
            )
        output_shape_static = _get_static_shape_tuple(output_shape)
        if len(output_shape_static) != 1 or output_shape_static[0] != 3:
            raise ValueError(f"Expected output_shape shaped (3,), got {output_shape.shape}.")

        index_mapping_affine = ops.matmul(invert_affine(src_affine), dst_affine)
        return self._resample_from_mapping(
            tensor=tensor,
            index_mapping_affine=index_mapping_affine,
            output_shape=output_shape,
            interpolation=interpolation,
            padding_mode=padding_mode,
            fill_value=fill_value,
        )

    def resample_many(
        self,
        tensors: Mapping[str, Any],
        src_affine: Any,
        dst_affine: Any,
        output_shape: Any,
        interpolation: Mapping[str, str],
        padding_mode: str = "constant",
        fill_value: float = 0.0,
    ) -> dict[str, Any]:
        """Resample multiple volumes while sharing the same coordinate mapping.

        This is useful for image-label pairs that live in the same physical
        space. The affine mapping is computed once and per-chunk coordinates are
        shared across tensors, while each key still uses its own interpolation
        mode.
        """
        src_affine = ops.cast(src_affine, "float32")
        dst_affine = ops.cast(dst_affine, "float32")
        output_shape = ops.cast(output_shape, "int32")
        index_mapping_affine = ops.matmul(invert_affine(src_affine), dst_affine)
        tensors = {key: ops.convert_to_tensor(tensor) for key, tensor in tensors.items()}
        return self._resample_many_from_mapping(
            tensors=tensors,
            index_mapping_affine=index_mapping_affine,
            output_shape=output_shape,
            interpolation=interpolation,
            padding_mode=padding_mode,
            fill_value=fill_value,
        )

    def _resample_from_mapping(
        self,
        tensor: Any,
        index_mapping_affine: Any,
        output_shape: Any,
        interpolation: str,
        padding_mode: str = "constant",
        fill_value: float = 0.0,
    ) -> Any:
        """Resample one tensor using a precomputed output-to-source mapping."""
        tensor = ops.convert_to_tensor(tensor)
        index_mapping_affine = ops.cast(index_mapping_affine, "float32")
        num_points = ops.prod(output_shape)
        chunk_size = ops.convert_to_tensor(self.max_points_per_chunk, dtype="int32")
        num_chunks = ops.floor_divide(num_points + chunk_size - 1, chunk_size)
        chunk_starts = ops.arange(num_chunks, dtype="int32") * chunk_size

        def sample_chunk(start: Any) -> Any:
            grid_chunk = make_output_grid_chunk(output_shape, start, chunk_size)
            src_coords = affine_apply(index_mapping_affine, grid_chunk)
            return sample_volume(
                tensor,
                src_coords,
                interpolation=interpolation,
                padding_mode=padding_mode,
                fill_value=fill_value,
            )

        sampled = ops.map(sample_chunk, chunk_starts)
        channels = _get_static_shape_tuple(tensor)[-1]
        concrete_output_shape = _concrete_int_tuple(output_shape)
        if channels is not None and concrete_output_shape is not None:
            num_points_value = 1
            for dimension in concrete_output_shape:
                num_points_value *= dimension
            sampled = ops.reshape(sampled, (-1, channels))
            sampled = ops.slice(sampled, [0, 0], [num_points_value, channels])
            return ops.reshape(sampled, (*concrete_output_shape, channels))

        channels_tensor = ops.shape(tensor)[-1]
        sampled = ops.reshape(sampled, ops.stack([-1, channels_tensor]))
        sampled = ops.slice(sampled, [0, 0], ops.stack([num_points, channels_tensor]))
        return ops.reshape(
            sampled,
            ops.concatenate([output_shape, ops.reshape(channels_tensor, (1,))], axis=0),
        )

    def _resample_many_from_mapping(
        self,
        tensors: Mapping[str, Any],
        index_mapping_affine: Any,
        output_shape: Any,
        interpolation: Mapping[str, str],
        padding_mode: str = "constant",
        fill_value: float = 0.0,
    ) -> dict[str, Any]:
        """Resample multiple tensors while sharing per-chunk coordinates."""
        if not tensors:
            return {}

        tensor_items = list(tensors.items())
        index_mapping_affine = ops.cast(index_mapping_affine, "float32")
        num_points = ops.prod(output_shape)
        chunk_size = ops.convert_to_tensor(self.max_points_per_chunk, dtype="int32")
        num_chunks = ops.floor_divide(num_points + chunk_size - 1, chunk_size)
        chunk_starts = ops.arange(num_chunks, dtype="int32") * chunk_size
        grid_chunks = ops.map(
            lambda start: make_output_grid_chunk(output_shape, start, chunk_size),
            chunk_starts,
        )
        coordinate_chunks = ops.map(
            lambda grid: affine_apply(index_mapping_affine, grid),
            grid_chunks,
        )

        outputs = {}
        for key, tensor in tensor_items:
            sampled_chunks = ops.map(
                lambda coords: sample_volume(
                    tensor,
                    coords,
                    interpolation=interpolation[key],
                    padding_mode=padding_mode,
                    fill_value=fill_value,
                ),
                coordinate_chunks,
            )
            channels = _get_static_shape_tuple(tensor)[-1]
            concrete_output_shape = _concrete_int_tuple(output_shape)
            if channels is not None and concrete_output_shape is not None:
                num_points_value = 1
                for dimension in concrete_output_shape:
                    num_points_value *= dimension
                sampled = ops.reshape(sampled_chunks, (-1, channels))
                sampled = ops.slice(sampled, [0, 0], [num_points_value, channels])
                outputs[key] = ops.reshape(sampled, (*concrete_output_shape, channels))
                continue

            channels_tensor = ops.shape(tensor)[-1]
            sampled = ops.reshape(sampled_chunks, ops.stack([-1, channels_tensor]))
            sampled = ops.slice(sampled, [0, 0], ops.stack([num_points, channels_tensor]))
            outputs[key] = ops.reshape(
                sampled,
                ops.concatenate([output_shape, ops.reshape(channels_tensor, (1,))], axis=0),
            )
        return outputs

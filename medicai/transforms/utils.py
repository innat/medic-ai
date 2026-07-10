from typing import Mapping, Sequence
import numpy as np
from scipy import ndimage
import tensorflow as tf


def largest_component_mask(mask: np.ndarray, closing_iterations: int = 2) -> np.ndarray:
    """Keep only the largest connected foreground component of a binary mask.

    Dilates the mask by `closing_iterations` steps before labeling to bridge
    thin/faint gaps (e.g. soft tissue edges, skin folds) that could otherwise
    fragment a single region into multiple components, then intersects the
    result back with the original mask so no pixels are added beyond the
    true foreground boundary.

    Args:
        mask: Boolean-like array, (H, W) for 2D or (D, H, W) for 3D.
        closing_iterations: Dilation steps used to bridge gaps before
            labeling. Increase if a real region is getting fragmented/
            truncated; decrease if disconnected artifacts (e.g. text)
            start merging into the kept region. Set to 0 to disable bridging.

    Returns:
        Boolean array, same shape as `mask`, True only where the largest
        connected component is.
    """
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask
    structure = np.ones((3,) * mask.ndim, dtype=bool)
    bridged = (
        ndimage.binary_dilation(mask, structure=structure, iterations=closing_iterations)
        if closing_iterations > 0
        else mask
    )
    labeled, num_components = ndimage.label(bridged, structure=structure)
    if num_components <= 1:
        return mask
    sizes = ndimage.sum(bridged, labeled, index=range(1, num_components + 1))
    largest_label = 1 + int(np.argmax(sizes))
    return np.logical_and(mask, labeled == largest_label)


def validate_affine_matrix(affine: tf.Tensor) -> tf.Tensor:
    """Validate and normalize an affine matrix to float32 4x4 form.

    Args:
        affine: Candidate affine matrix.

    Returns:
        tf.Tensor: The affine cast to ``tf.float32``.

    Raises:
        ValueError: If ``affine`` is not statically shaped ``(4, 4)``.
    """
    affine = tf.cast(affine, tf.float32)
    if affine.shape.rank != 2 or affine.shape[0] != 4 or affine.shape[1] != 4:
        raise ValueError(f"Expected a 4x4 affine matrix, got shape {affine.shape}.")
    return affine


def get_tensor_rank(tensor: tf.Tensor) -> int:
    """Return the static rank of a channel-last sample tensor.

    Args:
        tensor: Input tensor with channel-last layout.

    Returns:
        int: Static tensor rank.

    Raises:
        ValueError: If the tensor rank is unknown.
    """
    rank = tensor.shape.rank
    if rank is None:
        raise ValueError("Tensor rank must be statically known.")
    return rank


def get_spatial_rank(tensor: tf.Tensor) -> int:
    """Return the number of spatial dimensions in a channel-last sample tensor."""
    rank = get_tensor_rank(tensor)
    if rank < 2:
        raise ValueError(
            "Expected a channel-last sample tensor with at least one spatial axis and one "
            f"channel axis. Received rank {rank}."
        )
    return rank - 1


def validate_spatial_rank(
    tensor: tf.Tensor,
    allowed_ranks: Sequence[int] = (2, 3),
) -> int:
    """Validate the spatial rank of a channel-last sample tensor."""
    spatial_rank = get_spatial_rank(tensor)
    if spatial_rank not in allowed_ranks:
        allowed = ", ".join(str(rank) for rank in allowed_ranks)
        raise ValueError(
            f"Expected spatial rank in ({allowed}), received {spatial_rank} "
            f"for shape {tensor.shape}."
        )
    return spatial_rank


def get_spatial_shape(tensor: tf.Tensor) -> tf.Tensor:
    """Return the dynamic spatial shape of a channel-last sample tensor."""
    spatial_rank = get_spatial_rank(tensor)
    return tf.shape(tensor)[:spatial_rank]


def ensure_spatial_tuple(
    value: int | Sequence[int],
    spatial_rank: int,
    name: str,
) -> tuple[int, ...]:
    """Normalize an integer or sequence to a spatial-rank-sized tuple."""
    if isinstance(value, int):
        return (value,) * spatial_rank

    value = tuple(value)
    if len(value) != spatial_rank:
        raise ValueError(f"`{name}` must have length {spatial_rank}, got {len(value)}.")
    return value


def normalize_axes(axes: Sequence[int], rank: int, name: str = "axes") -> tuple[int, ...]:
    """Normalize possibly-negative axes against a tensor rank."""
    if len(axes) == 0:
        raise ValueError(f"`{name}` cannot be empty.")

    normalized = []
    for axis in axes:
        normalized_axis = axis if axis >= 0 else rank + axis
        if normalized_axis < 0 or normalized_axis >= rank:
            raise ValueError(f"Axis {axis} is out of bounds for rank {rank}.")
        normalized.append(normalized_axis)

    if len(set(normalized)) != len(normalized):
        raise ValueError(f"`{name}` must contain unique axes. Received {axes}.")

    return tuple(normalized)


def normalize_spatial_axes(
    axes: Sequence[int],
    spatial_rank: int,
    name: str = "spatial_axes",
) -> tuple[int, ...]:
    """Normalize axes expressed relative to spatial dimensions only."""
    return normalize_axes(axes, spatial_rank, name=name)


# Affine Utility


def spacing_from_affine(affine: tf.Tensor) -> tf.Tensor:
    """Extract voxel spacing magnitudes from a 4x4 affine matrix.

    The spacing is computed as the Euclidean norm of each spatial column in the
    affine's upper-left ``3x3`` linear block.
    """
    affine = validate_affine_matrix(affine)
    linear = affine[:3, :3]
    return tf.norm(linear, axis=0)


def direction_from_affine(affine: tf.Tensor) -> tf.Tensor:
    """Extract normalized direction columns from a 4x4 affine matrix.

    This returns the orientation component of the affine after removing voxel
    spacing magnitudes from the upper-left ``3x3`` linear block.
    """
    affine = validate_affine_matrix(affine)
    linear = affine[:3, :3]
    spacing = spacing_from_affine(affine)
    safe_spacing = tf.where(spacing > 0.0, spacing, tf.ones_like(spacing))
    return linear / safe_spacing[tf.newaxis, :]


def is_axis_aligned_affine(
    affine: tf.Tensor,
    atol: float = 1e-5,
) -> tf.Tensor:
    """Return whether an affine preserves tensor axis order without rotation.

    This allows sign flips but rejects axis permutation and general rotation.
    It is useful for deciding when shape-based resize can safely replace full
    affine-aware resampling.
    """
    direction = direction_from_affine(affine)
    off_diagonal = direction - tf.linalg.diag(tf.linalg.diag_part(direction))
    return tf.reduce_all(tf.abs(off_diagonal) <= tf.cast(atol, direction.dtype))


def origin_from_affine(affine: tf.Tensor) -> tf.Tensor:
    """Extract the world-space origin from a 4x4 affine matrix."""
    affine = validate_affine_matrix(affine)
    return affine[:3, 3]


def invert_affine(affine: tf.Tensor) -> tf.Tensor:
    """Invert a 4x4 affine matrix."""
    affine = validate_affine_matrix(affine)
    return tf.linalg.inv(affine)


def build_affine(
    spacing: tf.Tensor,
    direction: tf.Tensor,
    origin: tf.Tensor,
) -> tf.Tensor:
    """Build a 4x4 affine matrix from spacing, direction, and origin."""
    spacing = tf.cast(spacing, tf.float32)
    direction = tf.cast(direction, tf.float32)
    origin = tf.cast(origin, tf.float32)

    linear = direction * spacing[tf.newaxis, :]
    bottom_row = tf.constant([[0.0, 0.0, 0.0, 1.0]], dtype=tf.float32)
    top = tf.concat([linear, origin[:, tf.newaxis]], axis=1)
    return tf.concat([top, bottom_row], axis=0)


def affine_apply(affine: tf.Tensor, points: tf.Tensor) -> tf.Tensor:
    """Apply a 4x4 affine matrix to points shaped ``(..., 3)``."""
    affine = validate_affine_matrix(affine)
    points = tf.cast(points, tf.float32)
    ones = tf.ones_like(points[..., :1])
    homogeneous = tf.concat([points, ones], axis=-1)
    transformed = tf.linalg.matvec(affine, homogeneous)
    return transformed[..., :3]


# Resampling Utility


def round_half_up(values: tf.Tensor) -> tf.Tensor:
    """Round floating-point values with half-up semantics."""
    return tf.floor(values + 0.5)


def compute_destination_affine(
    src_affine: tf.Tensor,
    pixdim: tf.Tensor,
    diagonal: bool = False,
) -> tf.Tensor:
    """Compute a destination affine for resampling."""
    src_affine = tf.cast(src_affine, tf.float32)
    pixdim = tf.cast(pixdim, tf.float32)
    origin = origin_from_affine(src_affine)
    direction = tf.eye(3, dtype=tf.float32) if diagonal else direction_from_affine(src_affine)
    return build_affine(pixdim, direction, origin)


def compute_output_shape(
    input_shape: tf.Tensor,
    src_affine: tf.Tensor,
    dst_affine: tf.Tensor,
    align_corners: bool = False,
) -> tf.Tensor:
    """Compute an output shape from source and destination geometry.

    The output extent is derived from source-volume corner coordinates mapped
    into the destination index space, so it remains correct for both
    axis-aligned and permuted affines.
    """
    input_shape = tf.cast(input_shape, tf.float32)
    src_affine = tf.cast(src_affine, tf.float32)
    dst_affine = tf.cast(dst_affine, tf.float32)

    if align_corners:
        max_corner = tf.maximum(input_shape - 1.0, 0.0)
        src_corners = tf.stack(
            tf.meshgrid(
                tf.stack([0.0, max_corner[0]]),
                tf.stack([0.0, max_corner[1]]),
                tf.stack([0.0, max_corner[2]]),
                indexing="ij",
            ),
            axis=-1,
        )
        src_corners = tf.reshape(src_corners, [-1, 3])
        dst_corners = affine_apply(
            tf.linalg.matmul(invert_affine(dst_affine), src_affine), src_corners
        )
        min_corner = tf.reduce_min(dst_corners, axis=0)
        max_corner = tf.reduce_max(dst_corners, axis=0)
        output_shape = round_half_up(tf.maximum(max_corner - min_corner, 0.0)) + 1.0
    else:
        max_corner = tf.maximum(input_shape, 0.0)
        src_corners = tf.stack(
            tf.meshgrid(
                tf.stack([0.0, max_corner[0]]),
                tf.stack([0.0, max_corner[1]]),
                tf.stack([0.0, max_corner[2]]),
                indexing="ij",
            ),
            axis=-1,
        )
        src_corners = tf.reshape(src_corners, [-1, 3])
        dst_corners = affine_apply(
            tf.linalg.matmul(invert_affine(dst_affine), src_affine), src_corners
        )
        min_corner = tf.reduce_min(dst_corners, axis=0)
        max_corner = tf.reduce_max(dst_corners, axis=0)
        output_shape = round_half_up(tf.maximum(max_corner - min_corner, 0.0))

    return tf.maximum(tf.cast(output_shape, tf.int32), 1)


def make_output_grid(output_shape: tf.Tensor) -> tf.Tensor:
    """Create an output index grid shaped ``(N, 3)`` in ``(D, H, W)`` order."""
    output_shape = tf.cast(output_shape, tf.int32)
    d = tf.range(output_shape[0], dtype=tf.float32)
    h = tf.range(output_shape[1], dtype=tf.float32)
    w = tf.range(output_shape[2], dtype=tf.float32)
    dd, hh, ww = tf.meshgrid(d, h, w, indexing="ij")
    grid = tf.stack([dd, hh, ww], axis=-1)
    return tf.reshape(grid, [-1, 3])


def make_output_grid_chunk(
    output_shape: tf.Tensor,
    start: tf.Tensor,
    size: tf.Tensor,
) -> tf.Tensor:
    """Create one chunk of an output index grid from flat voxel indices.

    Args:
        output_shape: Spatial shape tensor shaped ``(3,)`` in ``(D, H, W)``
            order.
        start: Scalar flat-index offset into the full output grid.
        size: Number of grid points to generate.

    Returns:
        tf.Tensor: Float32 grid chunk shaped ``(size, 3)`` in ``(D, H, W)``
        order.
    """
    output_shape = tf.cast(output_shape, tf.int32)
    start = tf.cast(start, tf.int32)
    size = tf.cast(size, tf.int32)

    flat = tf.range(start, start + size, dtype=tf.int32)
    hw = output_shape[1] * output_shape[2]
    d = tf.math.floordiv(flat, hw)
    rem = tf.math.floormod(flat, hw)
    h = tf.math.floordiv(rem, output_shape[2])
    w = tf.math.floormod(rem, output_shape[2])
    return tf.cast(tf.stack([d, h, w], axis=1), tf.float32)


# Spatial Utility


def _validate_volume_and_coords(
    volume: tf.Tensor, coords: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    volume = tf.convert_to_tensor(volume)
    coords = tf.cast(tf.convert_to_tensor(coords), tf.float32)

    if volume.shape.rank != 4:
        raise ValueError(
            f"Expected a 4D channel-last volume shaped (D, H, W, C), got {volume.shape}."
        )
    if coords.shape.rank != 2 or coords.shape[-1] != 3:
        raise ValueError(f"Expected coords shaped (N, 3), got {coords.shape}.")
    return volume, coords


def _gather_with_fill(
    volume: tf.Tensor,
    indices: tf.Tensor,
    valid: tf.Tensor,
    fill_value: float,
    output_dtype: tf.DType,
) -> tf.Tensor:
    """Gather volume values and replace out-of-bounds samples with a fill value."""
    safe_indices = tf.where(valid[:, tf.newaxis], indices, tf.zeros_like(indices))
    gathered = tf.gather_nd(volume, safe_indices)
    return tf.where(valid[:, tf.newaxis], gathered, tf.cast(fill_value, output_dtype))


def sample_nearest(
    volume: tf.Tensor,
    coords: tf.Tensor,
    padding_mode: str = "constant",
    fill_value: float = 0.0,
) -> tf.Tensor:
    """Sample a 3D volume at arbitrary coordinates using nearest neighbors."""
    if padding_mode != "constant":
        raise ValueError(
            f"Unsupported padding_mode '{padding_mode}'. Only 'constant' is supported."
        )

    volume, coords = _validate_volume_and_coords(volume, coords)
    indices = tf.cast(tf.round(coords), tf.int32)
    shape = tf.shape(volume)[:3]
    valid = tf.reduce_all((indices >= 0) & (indices < shape), axis=1)
    return _gather_with_fill(volume, indices, valid, fill_value, volume.dtype)


def sample_trilinear(
    volume: tf.Tensor,
    coords: tf.Tensor,
    padding_mode: str = "constant",
    fill_value: float = 0.0,
) -> tf.Tensor:
    """Sample a 3D volume at arbitrary coordinates using trilinear interpolation.

    The eight neighboring voxel corners are gathered in one batched operation
    before interpolation weights are applied, which keeps the implementation
    graph-friendly while reducing repeated gather overhead.
    """
    if padding_mode != "constant":
        raise ValueError(
            f"Unsupported padding_mode '{padding_mode}'. Only 'constant' is supported."
        )

    volume, coords = _validate_volume_and_coords(volume, coords)
    original_dtype = volume.dtype
    output_dtype = original_dtype if original_dtype.is_floating else tf.float32
    volume = tf.cast(volume, output_dtype)

    lower = tf.floor(coords)
    upper = lower + 1.0
    frac = coords - lower

    d0 = tf.cast(lower[:, 0], tf.int32)
    h0 = tf.cast(lower[:, 1], tf.int32)
    w0 = tf.cast(lower[:, 2], tf.int32)
    d1 = tf.cast(upper[:, 0], tf.int32)
    h1 = tf.cast(upper[:, 1], tf.int32)
    w1 = tf.cast(upper[:, 2], tf.int32)

    wd = tf.cast(frac[:, 0:1], output_dtype)
    wh = tf.cast(frac[:, 1:2], output_dtype)
    ww = tf.cast(frac[:, 2:3], output_dtype)

    one = tf.cast(1.0, output_dtype)
    shape = tf.shape(volume)[:3]

    all_d = tf.concat([d0, d0, d0, d0, d1, d1, d1, d1], axis=0)
    all_h = tf.concat([h0, h0, h1, h1, h0, h0, h1, h1], axis=0)
    all_w = tf.concat([w0, w1, w0, w1, w0, w1, w0, w1], axis=0)
    all_indices = tf.stack([all_d, all_h, all_w], axis=1)
    all_valid = tf.reduce_all((all_indices >= 0) & (all_indices < shape), axis=1)
    all_gathered = _gather_with_fill(volume, all_indices, all_valid, fill_value, output_dtype)
    c000, c001, c010, c011, c100, c101, c110, c111 = tf.split(all_gathered, 8, axis=0)

    out = (
        c000 * (one - wd) * (one - wh) * (one - ww)
        + c001 * (one - wd) * (one - wh) * ww
        + c010 * (one - wd) * wh * (one - ww)
        + c011 * (one - wd) * wh * ww
        + c100 * wd * (one - wh) * (one - ww)
        + c101 * wd * (one - wh) * ww
        + c110 * wd * wh * (one - ww)
        + c111 * wd * wh * ww
    )

    return tf.cast(out, original_dtype) if original_dtype != output_dtype else out


def sample_volume(
    volume: tf.Tensor,
    coords: tf.Tensor,
    interpolation: str,
    padding_mode: str = "constant",
    fill_value: float = 0.0,
) -> tf.Tensor:
    """Dispatch 3D volume sampling to the requested interpolation kernel."""
    if interpolation == "nearest":
        return sample_nearest(
            volume,
            coords,
            padding_mode=padding_mode,
            fill_value=fill_value,
        )
    if interpolation == "trilinear":
        return sample_trilinear(
            volume,
            coords,
            padding_mode=padding_mode,
            fill_value=fill_value,
        )
    raise ValueError(
        f"Unsupported interpolation '{interpolation}'. Allowed values are 'nearest' and 'trilinear'."
    )


class SpatialResample:
    """Internal affine-aware 3D resampling primitive."""

    def __init__(self, max_points_per_chunk: int = 65536):
        if max_points_per_chunk < 1:
            raise ValueError("`max_points_per_chunk` must be a positive integer.")
        self.max_points_per_chunk = int(max_points_per_chunk)

    def __call__(
        self,
        tensor: tf.Tensor,
        src_affine: tf.Tensor,
        dst_affine: tf.Tensor,
        output_shape: tf.Tensor,
        interpolation: str,
        padding_mode: str = "constant",
        fill_value: float = 0.0,
    ) -> tf.Tensor:
        tensor = tf.convert_to_tensor(tensor)
        src_affine = tf.cast(src_affine, tf.float32)
        dst_affine = tf.cast(dst_affine, tf.float32)
        output_shape = tf.cast(output_shape, tf.int32)

        if tensor.shape.rank != 4:
            raise ValueError(
                f"Expected a 4D channel-last tensor shaped (D, H, W, C), got {tensor.shape}."
            )
        if output_shape.shape.rank != 1 or output_shape.shape[0] != 3:
            raise ValueError(f"Expected output_shape shaped (3,), got {output_shape.shape}.")

        index_mapping_affine = tf.linalg.matmul(invert_affine(src_affine), dst_affine)
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
        tensors: Mapping[str, tf.Tensor],
        src_affine: tf.Tensor,
        dst_affine: tf.Tensor,
        output_shape: tf.Tensor,
        interpolation: Mapping[str, str],
        padding_mode: str = "constant",
        fill_value: float = 0.0,
    ) -> dict[str, tf.Tensor]:
        """Resample multiple volumes while sharing the same coordinate mapping.

        This is useful for image-label pairs that live in the same physical
        space. The affine mapping is computed once and per-chunk coordinates are
        shared across tensors, while each key still uses its own interpolation
        mode.
        """
        src_affine = tf.cast(src_affine, tf.float32)
        dst_affine = tf.cast(dst_affine, tf.float32)
        output_shape = tf.cast(output_shape, tf.int32)
        index_mapping_affine = tf.linalg.matmul(invert_affine(src_affine), dst_affine)
        tensors = {key: tf.convert_to_tensor(tensor) for key, tensor in tensors.items()}
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
        tensor: tf.Tensor,
        index_mapping_affine: tf.Tensor,
        output_shape: tf.Tensor,
        interpolation: str,
        padding_mode: str = "constant",
        fill_value: float = 0.0,
    ) -> tf.Tensor:
        """Resample one tensor using a precomputed output-to-source mapping."""
        tensor = tf.convert_to_tensor(tensor)
        index_mapping_affine = tf.cast(index_mapping_affine, tf.float32)
        num_points = tf.reduce_prod(output_shape)
        chunk_size = tf.constant(self.max_points_per_chunk, dtype=tf.int32)
        num_chunks = tf.cast(tf.math.floordiv(num_points + chunk_size - 1, chunk_size), tf.int32)
        sampled_chunks = tf.TensorArray(dtype=tensor.dtype, size=num_chunks, infer_shape=False)

        def loop_body(index: tf.Tensor, chunks: tf.TensorArray) -> tuple[tf.Tensor, tf.TensorArray]:
            start = index * chunk_size
            size = tf.minimum(chunk_size, num_points - start)
            grid_chunk = make_output_grid_chunk(output_shape, start, size)
            src_coords = affine_apply(index_mapping_affine, grid_chunk)
            sampled_chunk = sample_volume(
                tensor,
                src_coords,
                interpolation=interpolation,
                padding_mode=padding_mode,
                fill_value=fill_value,
            )
            return index + 1, chunks.write(index, sampled_chunk)

        _, sampled_chunks = tf.while_loop(
            lambda index, _: index < num_chunks,
            loop_body,
            (tf.constant(0, dtype=tf.int32), sampled_chunks),
            parallel_iterations=1,
        )

        sampled = sampled_chunks.concat()
        channels = tf.shape(tensor)[-1]
        return tf.reshape(sampled, tf.concat([output_shape, [channels]], axis=0))

    def _resample_many_from_mapping(
        self,
        tensors: Mapping[str, tf.Tensor],
        index_mapping_affine: tf.Tensor,
        output_shape: tf.Tensor,
        interpolation: Mapping[str, str],
        padding_mode: str = "constant",
        fill_value: float = 0.0,
    ) -> dict[str, tf.Tensor]:
        """Resample multiple tensors while sharing per-chunk coordinates."""
        if not tensors:
            return {}

        tensor_items = list(tensors.items())
        index_mapping_affine = tf.cast(index_mapping_affine, tf.float32)
        num_points = tf.reduce_prod(output_shape)
        chunk_size = tf.constant(self.max_points_per_chunk, dtype=tf.int32)
        num_chunks = tf.cast(tf.math.floordiv(num_points + chunk_size - 1, chunk_size), tf.int32)

        chunk_arrays = {
            key: tf.TensorArray(dtype=tensor.dtype, size=num_chunks, infer_shape=False)
            for key, tensor in tensor_items
        }

        def loop_body(
            index: tf.Tensor,
            *arrays: tf.TensorArray,
        ):
            start = index * chunk_size
            size = tf.minimum(chunk_size, num_points - start)
            grid_chunk = make_output_grid_chunk(output_shape, start, size)
            src_coords = affine_apply(index_mapping_affine, grid_chunk)

            updated_arrays = []
            for array, (key, tensor) in zip(arrays, tensor_items):
                sampled_chunk = sample_volume(
                    tensor,
                    src_coords,
                    interpolation=interpolation[key],
                    padding_mode=padding_mode,
                    fill_value=fill_value,
                )
                updated_arrays.append(array.write(index, sampled_chunk))
            return (index + 1, *updated_arrays)

        _, *chunk_arrays_out = tf.while_loop(
            lambda index, *_: index < num_chunks,
            loop_body,
            (tf.constant(0, dtype=tf.int32), *chunk_arrays.values()),
            parallel_iterations=1,
        )

        outputs = {}
        for array, (key, tensor) in zip(chunk_arrays_out, tensor_items):
            sampled = array.concat()
            channels = tf.shape(tensor)[-1]
            outputs[key] = tf.reshape(sampled, tf.concat([output_shape, [channels]], axis=0))
        return outputs

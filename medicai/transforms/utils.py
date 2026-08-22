from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from keras import ops
import tensorflow as tf


# Backend-neutral layout and shape utilities


@dataclass(frozen=True)
class LayoutInfo:
    """Static layout description for a channel-last transform tensor.

    Args:
        input_layout: Canonical layout string such as ``"HWC"`` or ``"BDHWC"``.
        tensor_rank: Total tensor rank.
        spatial_rank: Number of spatial axes.
        batched: Whether the leading axis is interpreted as batch.
        batch_axis: Batch-axis index when ``batched=True``; otherwise ``None``.
        channel_axis: Channel-axis index. In Medic-AI transforms this is
            always the last axis.
        spatial_axes: Spatial-axis indices in tensor order.
    """

    input_layout: str
    tensor_rank: int
    spatial_rank: int
    batched: bool
    batch_axis: int | None
    channel_axis: int
    spatial_axes: tuple[int, ...]


_INPUT_LAYOUT_TO_INFO: Mapping[str, dict[str, int | bool | tuple[int, ...] | None]] = {
    "HWC": {
        "tensor_rank": 3,
        "spatial_rank": 2,
        "batched": False,
        "batch_axis": None,
        "spatial_axes": (0, 1),
    },
    "DHWC": {
        "tensor_rank": 4,
        "spatial_rank": 3,
        "batched": False,
        "batch_axis": None,
        "spatial_axes": (0, 1, 2),
    },
    "BHWC": {
        "tensor_rank": 4,
        "spatial_rank": 2,
        "batched": True,
        "batch_axis": 0,
        "spatial_axes": (1, 2),
    },
    "BDHWC": {
        "tensor_rank": 5,
        "spatial_rank": 3,
        "batched": True,
        "batch_axis": 0,
        "spatial_axes": (1, 2, 3),
    },
}
_SUPPORTED_INPUT_LAYOUTS: tuple[str, ...] = tuple(_INPUT_LAYOUT_TO_INFO)


def _get_static_shape_tuple(tensor: Any) -> tuple[int | None, ...]:
    """Return a backend-neutral static shape tuple when available."""
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise ValueError("Tensor must expose a `shape` attribute.")

    if hasattr(shape, "as_list"):
        try:
            return tuple(shape.as_list())
        except (TypeError, ValueError):
            pass

    try:
        return tuple(int(dim) if dim is not None else None for dim in shape)
    except TypeError as err:
        raise ValueError("Tensor shape must be tuple-like.") from err


def validate_affine_matrix(affine: Any) -> Any:
    """Validate and normalize an affine matrix to float32 4x4 form.

    Args:
        affine: Candidate affine matrix.

    Returns:
        Tensor-like object: The affine cast to ``float32``.

    Raises:
        ValueError: If ``affine`` is not statically shaped ``(4, 4)``.
    """
    affine = ops.cast(ops.convert_to_tensor(affine), "float32")
    shape = _get_static_shape_tuple(affine)
    if len(shape) != 2 or shape[0] != 4 or shape[1] != 4:
        raise ValueError(f"Expected a 4x4 affine matrix, got shape {affine.shape}.")
    return affine


def get_tensor_rank(tensor: Any) -> int:
    """Return the static rank of a channel-last sample tensor.

    Args:
        tensor: Input tensor with channel-last layout.

    Returns:
        int: Static tensor rank.

    Raises:
        ValueError: If the tensor rank is unknown.
    """
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise ValueError("Tensor must expose a `shape` attribute.")

    rank = getattr(shape, "rank", None)
    if rank is not None:
        return int(rank)

    try:
        return len(shape)
    except TypeError:
        pass

    ndim = getattr(tensor, "ndim", None)
    if ndim is not None:
        return int(ndim)

    rank = ops.ndim(tensor)
    if rank is None:
        raise ValueError("Tensor rank must be statically known.")
    return int(rank)


def normalize_input_layout(input_layout: str) -> str:
    """Normalize a public transform ``input_layout`` string.

    Args:
        input_layout: Layout string such as ``"hwc"`` or ``"BDHWC"``.

    Returns:
        str: Uppercase layout string.

    Raises:
        TypeError: If ``input_layout`` is not a string.
        ValueError: If the normalized layout is empty.
    """
    if not isinstance(input_layout, str):
        raise TypeError("`input_layout` must be a string.")
    normalized = input_layout.strip().upper()
    if not normalized:
        raise ValueError("`input_layout` cannot be empty.")
    return normalized


def validate_input_layout(
    input_layout: str,
    *,
    allowed_layouts: Sequence[str] = _SUPPORTED_INPUT_LAYOUTS,
    transform_name: str | None = None,
) -> str:
    """Validate and normalize a transform ``input_layout`` string.

    Args:
        input_layout: Candidate public layout string.
        allowed_layouts: Layouts accepted by the calling transform.
        transform_name: Optional transform name for clearer error messages.

    Returns:
        str: Canonical uppercase input layout.

    Raises:
        ValueError: If the layout is unsupported.
    """
    normalized = normalize_input_layout(input_layout)
    normalized_allowed = tuple(normalize_input_layout(layout) for layout in allowed_layouts)
    if normalized not in normalized_allowed:
        label = transform_name or "Transform"
        supported = ", ".join(repr(layout) for layout in normalized_allowed)
        raise ValueError(
            f"{label} supports only input_layout values ({supported}). Received {normalized!r}."
        )
    return normalized


def get_input_layout_info(input_layout: str) -> LayoutInfo:
    """Return canonical static metadata for a public ``input_layout`` string.

    Args:
        input_layout: Public layout string such as ``"HWC"`` or ``"BDHWC"``.

    Returns:
        LayoutInfo: Canonical layout metadata derived from ``input_layout``.
    """
    normalized = validate_input_layout(input_layout)
    info = _INPUT_LAYOUT_TO_INFO[normalized]
    tensor_rank = int(info["tensor_rank"])
    return LayoutInfo(
        input_layout=normalized,
        tensor_rank=tensor_rank,
        spatial_rank=int(info["spatial_rank"]),
        batched=bool(info["batched"]),
        batch_axis=info["batch_axis"],
        channel_axis=tensor_rank - 1,
        spatial_axes=tuple(info["spatial_axes"]),
    )


def get_batched_input_layout(input_layout: str) -> str:
    """Return the batched variant of a supported channel-last ``input_layout``."""
    normalized = validate_input_layout(input_layout)
    if normalized == "HWC":
        return "BHWC"
    if normalized == "DHWC":
        return "BDHWC"
    return normalized

def resolve_input_layout(
    *,
    input_layout: str,
    allowed_layouts: Sequence[str] = _SUPPORTED_INPUT_LAYOUTS,
    transform_name: str | None = None,
) -> str:
    """Resolve one canonical public ``input_layout``.

    Args:
        input_layout: Explicit public layout string.
        allowed_layouts: Layouts accepted by the calling transform.
        transform_name: Optional transform name for clearer error messages.

    Returns:
        str: Canonical uppercase input layout.
    """
    return validate_input_layout(
        input_layout,
        allowed_layouts=allowed_layouts,
        transform_name=transform_name,
    )


def validate_tensor_matches_layout(
    tensor: Any,
    input_layout: str,
    *,
    transform_name: str | None = None,
) -> LayoutInfo:
    """Validate that a tensor matches a declared public ``input_layout``.

    Args:
        tensor: Tensor to validate.
        input_layout: Public layout string such as ``"HWC"`` or ``"BDHWC"``.
        transform_name: Optional transform name for clearer error messages.

    Returns:
        LayoutInfo: Layout metadata for the validated ``input_layout``.

    Raises:
        ValueError: If the tensor rank does not match the declared layout.
    """
    layout = get_input_layout_info(input_layout)
    tensor_rank = get_tensor_rank(tensor)
    if tensor_rank != layout.tensor_rank:
        label = transform_name or "Transform"
        raise ValueError(
            f"{label} expects input_layout={layout.input_layout!r} with rank "
            f"{layout.tensor_rank}, but received rank {tensor_rank} for shape {tensor.shape}."
        )
    return layout

def get_spatial_rank(tensor: Any) -> int:
    """Return the number of spatial dimensions in a channel-last sample tensor."""
    rank = get_tensor_rank(tensor)
    if rank == 3:
        return 2
    if rank == 4:
        return 3
    raise ValueError(
        "Expected a channel-last sample tensor shaped like (H, W, C) or (D, H, W, C). "
        f"Received rank {rank} with shape {tensor.shape}."
    )


def validate_spatial_rank(
    tensor: Any,
    allowed_ranks: Sequence[int] = (2, 3),
) -> int:
    """Validate the spatial rank of a channel-last sample tensor."""
    spatial_rank = get_spatial_rank(tensor)
    if spatial_rank not in allowed_ranks:
        allowed = ", ".join(str(rank) for rank in allowed_ranks)
        raise ValueError(
            f"Expected spatial rank in ({allowed}), received {spatial_rank} for shape "
            f"{tensor.shape}."
        )
    return spatial_rank


def get_spatial_shape_for_layout(tensor: Any, *, input_layout: str) -> Any:
    """Return the dynamic spatial shape of a tensor validated by ``input_layout``.

    Args:
        tensor: Input tensor in Medic-AI channel-last layout.
        input_layout: Canonical public layout string such as ``"HWC"`` or
            ``"BDHWC"``.

    Returns:
        Tensor-like object: Dynamic spatial shape.
    """
    layout = validate_tensor_matches_layout(tensor, input_layout)
    return ops.take(ops.shape(tensor), layout.spatial_axes)

def ensure_batch_axis_for_layout(
    tensor: Any,
    *,
    input_layout: str,
    allowed_spatial_ranks: Sequence[int] = (2, 3),
) -> tuple[Any, bool]:
    """Normalize a tensor declared by ``input_layout`` to a batched view.

    Args:
        tensor: Input tensor in Medic-AI channel-last layout.
        input_layout: Canonical public layout string such as ``"HWC"`` or
            ``"BDHWC"``.
        allowed_spatial_ranks: Accepted spatial ranks for validation.

    Returns:
        tuple[Any, bool]: A pair ``(batched_tensor, added_batch_axis)``
        where ``added_batch_axis`` is ``True`` only when a leading singleton
        batch axis was inserted for sample-layout input.
    """
    layout = validate_tensor_matches_layout(tensor, input_layout)
    if layout.spatial_rank not in allowed_spatial_ranks:
        allowed = ", ".join(str(rank) for rank in allowed_spatial_ranks)
        raise ValueError(
            f"Expected spatial rank in ({allowed}) for input_layout={layout.input_layout!r}, "
            f"received {layout.spatial_rank} for shape {tensor.shape}."
        )
    if layout.batched:
        return tensor, False
    return ops.expand_dims(tensor, axis=0), True


def restore_from_batch_axis(tensor: Any, added_batch_axis: bool) -> Any:
    """Remove a temporary singleton batch axis added by :func:`ensure_batch_axis`."""
    if added_batch_axis:
        return ops.squeeze(tensor, axis=0)
    return tensor


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


def resolve_input_layout_axes(
    tensor: Any,
    axes: Sequence[int],
    *,
    input_layout: str,
    name: str = "spatial_axis",
) -> tuple[int, ...]:
    """Resolve actual tensor axes against a declared public ``input_layout``.

    Args:
        tensor: Input tensor expected to match ``input_layout``.
        axes: Axis indices expressed in real tensor-axis coordinates.
        input_layout: Canonical public layout string.
        name: Axis-group name used in error messages.

    Returns:
        tuple[int, ...]: Normalized tensor-axis indices.

    Raises:
        ValueError: If any resolved axis falls outside the spatial axes of the
            declared layout.
    """
    layout = validate_tensor_matches_layout(tensor, input_layout)
    normalized = normalize_axes(tuple(axes), layout.tensor_rank, name=name)
    invalid = tuple(axis for axis in normalized if axis not in layout.spatial_axes)
    if invalid:
        raise ValueError(
            f"`{name}` must refer only to spatial axes for input_layout="
            f"{layout.input_layout!r}. Received {normalized}, spatial axes are "
            f"{layout.spatial_axes}."
        )
    return normalized


# TensorFlow-native affine and resampling utilities
#
# These helpers still intentionally use TensorFlow directly. They sit below the
# backend-neutral layout/shape layer above, and power the remaining transform
# paths that have not yet been migrated fully to ``keras.ops``.


# Affine Utility


def spacing_from_affine(affine: Any) -> Any:
    """Extract voxel spacing magnitudes from a 4x4 affine matrix.

    The spacing is computed as the Euclidean norm of each spatial column in the
    affine's upper-left ``3x3`` linear block.
    """
    affine = validate_affine_matrix(affine)
    linear = affine[:3, :3]
    return ops.norm(linear, axis=0)


def direction_from_affine(affine: Any) -> Any:
    """Extract normalized direction columns from a 4x4 affine matrix.

    This returns the orientation component of the affine after removing voxel
    spacing magnitudes from the upper-left ``3x3`` linear block.
    """
    affine = validate_affine_matrix(affine)
    linear = affine[:3, :3]
    spacing = spacing_from_affine(affine)
    safe_spacing = ops.where(spacing > 0.0, spacing, ops.ones_like(spacing))
    return linear / safe_spacing[None, :]


def is_axis_aligned_affine(
    affine: Any,
    atol: float = 1e-5,
) -> Any:
    """Return whether an affine preserves tensor axis order without rotation.

    This allows sign flips but rejects axis permutation and general rotation.
    It is useful for deciding when shape-based resize can safely replace full
    affine-aware resampling.
    """
    direction = direction_from_affine(affine)
    off_diagonal = direction - ops.diag(ops.diagonal(direction))
    return ops.all(ops.abs(off_diagonal) <= ops.cast(atol, direction.dtype))


def origin_from_affine(affine: Any) -> Any:
    """Extract the world-space origin from a 4x4 affine matrix."""
    affine = validate_affine_matrix(affine)
    return affine[:3, 3]


def invert_affine(affine: Any) -> Any:
    """Invert a 4x4 affine matrix."""
    affine = validate_affine_matrix(affine)
    return ops.linalg.inv(affine)


def build_affine(
    spacing: Any,
    direction: Any,
    origin: Any,
) -> Any:
    """Build a 4x4 affine matrix from spacing, direction, and origin."""
    spacing = ops.cast(spacing, "float32")
    direction = ops.cast(direction, "float32")
    origin = ops.cast(origin, "float32")

    linear = direction * spacing[None, :]
    bottom_row = ops.convert_to_tensor([[0.0, 0.0, 0.0, 1.0]], dtype="float32")
    top = ops.concatenate([linear, origin[:, None]], axis=1)
    return ops.concatenate([top, bottom_row], axis=0)


def affine_apply(affine: Any, points: Any) -> Any:
    """Apply a 4x4 affine matrix to points shaped ``(..., 3)``."""
    affine = validate_affine_matrix(affine)
    points = ops.cast(points, "float32")
    ones = ops.ones_like(points[..., :1])
    homogeneous = ops.concatenate([points, ones], axis=-1)
    transformed = ops.matmul(homogeneous, ops.transpose(affine))
    return transformed[..., :3]


def orientation_from_affine(affine: tf.Tensor) -> tf.Tensor:
    """Infer a three-letter orientation code from a 4x4 affine matrix."""
    matrix = tf.cast(validate_affine_matrix(affine)[:3, :3], tf.float32)
    current_axes = tf.argmax(tf.abs(matrix), axis=0, output_type=tf.int32)
    unique_axes, _, counts = tf.unique_with_counts(current_axes)
    del unique_axes
    if tf.reduce_any(counts > 1):
        raise ValueError("Affine orientation is invalid: multiple output axes map to the same world axis.")
    gather_indices = tf.stack([current_axes, tf.range(3, dtype=tf.int32)], axis=1)
    signs = tf.gather_nd(matrix, gather_indices) >= 0

    def axis_code(axis_index: tf.Tensor, sign: tf.Tensor) -> tf.Tensor:
        return tf.case(
            [
                (tf.equal(axis_index, 0), lambda: tf.where(sign, "R", "L")),
                (tf.equal(axis_index, 1), lambda: tf.where(sign, "A", "P")),
            ],
            default=lambda: tf.where(sign, "S", "I"),
            exclusive=True,
        )

    codes = tf.map_fn(
        lambda pair: axis_code(pair[0], tf.cast(pair[1], tf.bool)),
        (current_axes, tf.cast(signs, tf.int32)),
        fn_output_signature=tf.string,
    )
    return tf.strings.reduce_join(codes)


def compute_orientation_transform(
    affine: tf.Tensor,
    target_tensor_axcodes: str,
) -> dict[str, tf.Tensor]:
    """Compute spatial permutation and flips for a target tensor orientation."""
    axis_to_world = {"R": 0, "L": 0, "A": 1, "P": 1, "S": 2, "I": 2}
    axis_to_sign = {"R": 1, "L": -1, "A": 1, "P": -1, "S": 1, "I": -1}

    matrix = tf.cast(validate_affine_matrix(affine)[:3, :3], tf.float32)
    current_axes = tf.argmax(tf.abs(matrix), axis=0, output_type=tf.int32)
    unique_axes, _, counts = tf.unique_with_counts(current_axes)
    del unique_axes
    if tf.reduce_any(counts > 1):
        raise ValueError("Affine orientation is invalid: multiple output axes map to the same world axis.")
    gather_indices = tf.stack([current_axes, tf.range(3, dtype=tf.int32)], axis=1)
    current_signs = tf.sign(tf.gather_nd(matrix, gather_indices))
    current_signs = tf.where(current_signs == 0, tf.ones_like(current_signs), current_signs)

    target_axes = [axis_to_world[code] for code in target_tensor_axcodes]
    perm_spatial = tf.stack(
        [
            tf.argmax(
                tf.cast(tf.equal(current_axes, target_axis), tf.int32),
                output_type=tf.int32,
            )
            for target_axis in target_axes
        ]
    )
    current_signs_for_output = tf.gather(current_signs, perm_spatial)
    target_signs = tf.constant(
        [axis_to_sign[code] for code in target_tensor_axcodes],
        dtype=tf.float32,
    )
    flip_axes = tf.reshape(
        tf.where(tf.not_equal(current_signs_for_output, target_signs)),
        [-1],
    )
    return {"perm_spatial": perm_spatial, "flip_axes": flip_axes}


def reoriented_affine(
    affine: tf.Tensor,
    input_spatial_shape: tf.Tensor,
    perm_spatial: tuple[int, int, int] | tf.Tensor,
    flip_axes: tuple[int, ...] | tf.Tensor,
) -> tf.Tensor:
    """Update affine metadata for a spatial permutation and flips."""
    affine = tf.cast(validate_affine_matrix(affine), tf.float32)
    input_spatial_shape = tf.cast(input_spatial_shape, tf.float32)
    perm_spatial = tf.cast(tf.convert_to_tensor(perm_spatial), tf.int32)
    flip_axes = tf.cast(tf.convert_to_tensor(flip_axes), tf.int32)

    flipped_output_mask = tf.scatter_nd(
        indices=tf.expand_dims(flip_axes, axis=1),
        updates=tf.ones_like(flip_axes, dtype=tf.float32),
        shape=(3,),
    )
    signs = 1.0 - 2.0 * flipped_output_mask

    transform = tf.eye(4, dtype=tf.float32)
    spatial_block = tf.transpose(
        tf.one_hot(perm_spatial, depth=3, dtype=tf.float32) * signs[:, None]
    )
    transform = tf.tensor_scatter_nd_update(
        transform,
        indices=[
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 0],
            [2, 1],
            [2, 2],
        ],
        updates=tf.reshape(spatial_block, [-1]),
    )

    flipped_input_mask = tf.scatter_nd(
        indices=tf.expand_dims(perm_spatial, axis=1),
        updates=flipped_output_mask,
        shape=(3,),
    )
    translations = flipped_input_mask * (input_spatial_shape - 1.0)
    transform = tf.tensor_scatter_nd_update(
        transform,
        indices=[[0, 3], [1, 3], [2, 3]],
        updates=translations,
    )
    return tf.linalg.matmul(affine, transform)


# Resampling Utility


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
        dst_corners = affine_apply(
            ops.matmul(invert_affine(dst_affine), src_affine), src_corners
        )
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
        dst_corners = affine_apply(
            ops.matmul(invert_affine(dst_affine), src_affine), src_corners
        )
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

    flat = ops.arange(start, start + size, dtype="int32")
    hw = output_shape[1] * output_shape[2]
    d = ops.floor_divide(flat, hw)
    rem = ops.mod(flat, hw)
    h = ops.floor_divide(rem, output_shape[2])
    w = ops.mod(rem, output_shape[2])
    return ops.cast(ops.stack([d, h, w], axis=1), "float32")


# Spatial Utility


def _validate_volume_and_coords(
    volume: Any, coords: Any
) -> tuple[Any, Any]:
    volume = ops.convert_to_tensor(volume)
    coords = ops.cast(ops.convert_to_tensor(coords), "float32")

    if get_tensor_rank(volume) != 4:
        raise ValueError(
            f"Expected a 4D channel-last volume shaped (D, H, W, C), got {volume.shape}."
        )
    coords_shape = _get_static_shape_tuple(coords)
    if len(coords_shape) != 2 or coords_shape[-1] != 3:
        raise ValueError(f"Expected coords shaped (N, 3), got {coords.shape}.")
    return volume, coords


def _gather_with_fill(
    volume: Any,
    indices: Any,
    valid: Any,
    fill_value: float,
    output_dtype: Any,
) -> Any:
    """Gather volume values and replace out-of-bounds samples with a fill value."""
    safe_indices = ops.where(valid[:, None], indices, ops.zeros_like(indices))
    gathered = tf.gather_nd(volume, safe_indices)
    return ops.where(valid[:, None], gathered, ops.cast(fill_value, output_dtype))


def sample_nearest(
    volume: Any,
    coords: Any,
    padding_mode: str = "constant",
    fill_value: float = 0.0,
) -> Any:
    """Sample a 3D volume at arbitrary coordinates using nearest neighbors."""
    if padding_mode != "constant":
        raise ValueError(
            f"Unsupported padding_mode '{padding_mode}'. Only 'constant' is supported."
        )

    volume, coords = _validate_volume_and_coords(volume, coords)
    indices = ops.cast(ops.round(coords), "int32")
    shape = ops.shape(volume)[:3]
    valid = ops.all((indices >= 0) & (indices < shape), axis=1)
    return _gather_with_fill(volume, indices, valid, fill_value, volume.dtype)


def sample_trilinear(
    volume: Any,
    coords: Any,
    padding_mode: str = "constant",
    fill_value: float = 0.0,
) -> Any:
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
    output_dtype = original_dtype if original_dtype.is_floating else "float32"
    volume = ops.cast(volume, output_dtype)

    lower = ops.floor(coords)
    upper = lower + 1.0
    frac = coords - lower

    d0 = ops.cast(lower[:, 0], "int32")
    h0 = ops.cast(lower[:, 1], "int32")
    w0 = ops.cast(lower[:, 2], "int32")
    d1 = ops.cast(upper[:, 0], "int32")
    h1 = ops.cast(upper[:, 1], "int32")
    w1 = ops.cast(upper[:, 2], "int32")

    wd = ops.cast(frac[:, 0:1], output_dtype)
    wh = ops.cast(frac[:, 1:2], output_dtype)
    ww = ops.cast(frac[:, 2:3], output_dtype)

    one = ops.cast(1.0, output_dtype)
    shape = ops.shape(volume)[:3]

    all_d = ops.concatenate([d0, d0, d0, d0, d1, d1, d1, d1], axis=0)
    all_h = ops.concatenate([h0, h0, h1, h1, h0, h0, h1, h1], axis=0)
    all_w = ops.concatenate([w0, w1, w0, w1, w0, w1, w0, w1], axis=0)
    all_indices = ops.stack([all_d, all_h, all_w], axis=1)
    all_valid = ops.all((all_indices >= 0) & (all_indices < shape), axis=1)
    all_gathered = _gather_with_fill(volume, all_indices, all_valid, fill_value, output_dtype)
    c000, c001, c010, c011, c100, c101, c110, c111 = ops.split(all_gathered, 8, axis=0)

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

    return ops.cast(out, original_dtype) if original_dtype != output_dtype else out


def sample_volume(
    volume: Any,
    coords: Any,
    interpolation: str,
    padding_mode: str = "constant",
    fill_value: float = 0.0,
) -> Any:
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
        tensor = tf.convert_to_tensor(tensor)
        index_mapping_affine = ops.cast(index_mapping_affine, "float32")
        num_points = ops.prod(output_shape)
        chunk_size = tf.constant(self.max_points_per_chunk, dtype=tf.int32)
        num_chunks = ops.cast(tf.math.floordiv(num_points + chunk_size - 1, chunk_size), "int32")
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
        channels = ops.shape(tensor)[-1]
        return ops.reshape(
            sampled,
            ops.concatenate([output_shape, ops.reshape(channels, (1,))], axis=0),
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
        chunk_size = tf.constant(self.max_points_per_chunk, dtype=tf.int32)
        num_chunks = ops.cast(tf.math.floordiv(num_points + chunk_size - 1, chunk_size), "int32")

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
            channels = ops.shape(tensor)[-1]
            outputs[key] = ops.reshape(
                sampled,
                ops.concatenate([output_shape, ops.reshape(channels, (1,))], axis=0),
            )
        return outputs

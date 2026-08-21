from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from keras import ops
import tensorflow as tf


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

def custom_tf_boolean_mask(
    tensor: Any,
    mask: Any,
    mode: str = "extract",
    fill_value: float | int = 0,
) -> tf.Tensor:
    """Apply a TensorFlow-native boolean mask without calling ``tf.boolean_mask``.

    Args:
        tensor: Input tensor.
        mask: Boolean-like mask tensor broadcastable to ``tensor``.
        mode: One of ``"extract"``, ``"where"``, or ``"multiply"``.
        fill_value: Fill value used when ``mode="where"``.

    Returns:
        tf.Tensor: Masked tensor according to ``mode``.
    """
    bool_mask = ops.cast(mask, "bool") if mask.dtype != tf.bool else mask

    if mode == "extract":
        indices = tf.where(bool_mask)
        return tf.gather_nd(tensor, indices)
    if mode == "where":
        fill_tensor = ops.cast(fill_value, tensor.dtype)
        return ops.where(bool_mask, tensor, fill_tensor)
    if mode == "multiply":
        numeric_mask = ops.cast(bool_mask, tensor.dtype)
        return ops.multiply(tensor, numeric_mask)
    raise ValueError(
        f"Unsupported mode '{mode}'. Choose from 'extract', 'where', or 'multiply'."
    )

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
        tf.Tensor: Dynamic spatial shape.
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
        tuple[tf.Tensor, bool]: A pair ``(batched_tensor, added_batch_axis)``
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
    return tensor[None, ...], True


def restore_from_batch_axis(tensor: Any, added_batch_axis: bool) -> Any:
    """Remove a temporary singleton batch axis added by :func:`ensure_batch_axis`."""
    if added_batch_axis:
        return tensor[0]
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
    tensor: tf.Tensor,
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

    if get_tensor_rank(volume) != 4:
        raise ValueError(
            f"Expected a 4D channel-last volume shaped (D, H, W, C), got {volume.shape}."
        )
    coords_shape = _get_static_shape_tuple(coords)
    if len(coords_shape) != 2 or coords_shape[-1] != 3:
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


def resize_volumes(
    volumes: tf.Tensor,
    depth: int | tf.Tensor,
    height: int | tf.Tensor,
    width: int | tf.Tensor,
    method: str = "trilinear",
    align_corners: bool = False,
) -> tf.Tensor:
    """Resize batched 3D channel-last volumes with TensorFlow-native kernels.

    Args:
        volumes: 5D tensor shaped ``(B, D, H, W, C)``.
        depth: Output depth.
        height: Output height.
        width: Output width.
        method: Either ``"trilinear"`` or ``"nearest"``.
        align_corners: Whether to align the extreme input/output corners for
            trilinear interpolation.

    Returns:
        tf.Tensor: Resized tensor shaped ``(B, depth, height, width, C)``.
    """
    # Keep this TensorFlow-native fallback localized here until Keras exposes
    # an equivalent backend-agnostic 3D resize primitive.

    def trilinear_resize(
        volumes: tf.Tensor,
        depth: int | tf.Tensor,
        height: int | tf.Tensor,
        width: int | tf.Tensor,
        align_corners: bool,
    ) -> tf.Tensor:
        original_dtype = volumes.dtype
        volumes = tf.cast(volumes, "float32")
        in_d = tf.shape(volumes)[1]
        in_h = tf.shape(volumes)[2]
        in_w = tf.shape(volumes)[3]

        if align_corners:
            z_coords = tf.linspace(0.0, tf.cast(in_d - 1, "float32"), depth)
            y_coords = tf.linspace(0.0, tf.cast(in_h - 1, "float32"), height)
            x_coords = tf.linspace(0.0, tf.cast(in_w - 1, "float32"), width)
        else:
            scale_d = tf.cast(in_d, "float32") / tf.cast(depth, "float32")
            scale_h = tf.cast(in_h, "float32") / tf.cast(height, "float32")
            scale_w = tf.cast(in_w, "float32") / tf.cast(width, "float32")

            z_coords = (tf.range(depth, dtype="float32") + 0.5) * scale_d - 0.5
            y_coords = (tf.range(height, dtype="float32") + 0.5) * scale_h - 0.5
            x_coords = (tf.range(width, dtype="float32") + 0.5) * scale_w - 0.5

            z_coords = tf.clip_by_value(z_coords, 0.0, tf.cast(in_d - 1, "float32"))
            y_coords = tf.clip_by_value(y_coords, 0.0, tf.cast(in_h - 1, "float32"))
            x_coords = tf.clip_by_value(x_coords, 0.0, tf.cast(in_w - 1, "float32"))

        def interpolate_1d(input_vol: tf.Tensor, coords: tf.Tensor, axis: int) -> tf.Tensor:
            idx0 = tf.cast(tf.floor(coords), "int32")
            idx1 = tf.minimum(idx0 + 1, tf.shape(input_vol)[axis] - 1)

            values0 = tf.gather(input_vol, idx0, axis=axis)
            values1 = tf.gather(input_vol, idx1, axis=axis)

            weight1 = coords - tf.cast(idx0, "float32")
            weight0 = 1.0 - weight1

            new_shape = [1] * 5
            new_shape[axis] = tf.shape(coords)[0]
            weight0 = tf.reshape(weight0, new_shape)
            weight1 = tf.reshape(weight1, new_shape)
            return weight0 * values0 + weight1 * values1

        interp_d = interpolate_1d(volumes, z_coords, axis=1)
        interp_h = interpolate_1d(interp_d, y_coords, axis=2)
        interp_w = interpolate_1d(interp_h, x_coords, axis=3)
        return tf.cast(interp_w, original_dtype)

    def nearest(
        volumes: tf.Tensor,
        depth: int | tf.Tensor,
        height: int | tf.Tensor,
        width: int | tf.Tensor,
    ) -> tf.Tensor:
        shape = tf.shape(volumes)
        bs, d, h, w, c = shape[0], shape[1], shape[2], shape[3], shape[4]

        z = tf.linspace(0.0, tf.cast(d - 1, "float32"), depth)
        z = tf.cast(tf.round(z), "int32")
        z = tf.clip_by_value(z, 0, d - 1)

        y = tf.linspace(0.0, tf.cast(h - 1, "float32"), height)
        y = tf.cast(tf.round(y), "int32")
        y = tf.clip_by_value(y, 0, h - 1)

        x = tf.linspace(0.0, tf.cast(w - 1, "float32"), width)
        x = tf.cast(tf.round(x), "int32")
        x = tf.clip_by_value(x, 0, w - 1)

        z_grid, y_grid, x_grid = tf.meshgrid(z, y, x, indexing="ij")
        z_grid = tf.reshape(z_grid, (-1,))
        y_grid = tf.reshape(y_grid, (-1,))
        x_grid = tf.reshape(x_grid, (-1,))

        batch_idx = tf.repeat(tf.range(bs), tf.shape(z_grid)[0])
        z_grid = tf.tile(z_grid, [bs])
        y_grid = tf.tile(y_grid, [bs])
        x_grid = tf.tile(x_grid, [bs])

        flat = tf.reshape(volumes, (bs * d * h * w, c))
        indices = (batch_idx * d * h * w) + (z_grid * h * w) + (y_grid * w) + x_grid
        result = tf.gather(flat, indices, axis=0)
        return tf.reshape(result, (bs, depth, height, width, c))

    if method == "trilinear":
        return trilinear_resize(volumes, depth, height, width, align_corners)
    if method == "nearest":
        return nearest(volumes, depth, height, width)
    raise ValueError(f"Unsupported resize method: {method}")


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

        if get_tensor_rank(tensor) != 4:
            raise ValueError(
                f"Expected a 4D channel-last tensor shaped (D, H, W, C), got {tensor.shape}."
            )
        output_shape_static = _get_static_shape_tuple(output_shape)
        if len(output_shape_static) != 1 or output_shape_static[0] != 3:
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

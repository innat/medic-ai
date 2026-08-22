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




# Compatibility exports
#
# Spatial helpers moved to focused modules, but these lazy exports preserve the
# established medicai.transforms.utils import path for existing users.

_SPATIAL_HELPERS = {
    "spacing_from_affine": ("medicai.transforms.spatial.affine_utils", "spacing_from_affine"),
    "direction_from_affine": ("medicai.transforms.spatial.affine_utils", "direction_from_affine"),
    "is_axis_aligned_affine": ("medicai.transforms.spatial.affine_utils", "is_axis_aligned_affine"),
    "origin_from_affine": ("medicai.transforms.spatial.affine_utils", "origin_from_affine"),
    "invert_affine": ("medicai.transforms.spatial.affine_utils", "invert_affine"),
    "build_affine": ("medicai.transforms.spatial.affine_utils", "build_affine"),
    "affine_apply": ("medicai.transforms.spatial.affine_utils", "affine_apply"),
    "orientation_from_affine": ("medicai.transforms.spatial.affine_utils", "orientation_from_affine"),
    "compute_orientation_transform": ("medicai.transforms.spatial.affine_utils", "compute_orientation_transform"),
    "reoriented_affine": ("medicai.transforms.spatial.affine_utils", "reoriented_affine"),
    "round_half_up": ("medicai.transforms.spatial.resample_utils", "round_half_up"),
    "compute_destination_affine": ("medicai.transforms.spatial.resample_utils", "compute_destination_affine"),
    "compute_output_shape": ("medicai.transforms.spatial.resample_utils", "compute_output_shape"),
    "make_output_grid": ("medicai.transforms.spatial.resample_utils", "make_output_grid"),
    "make_output_grid_chunk": ("medicai.transforms.spatial.resample_utils", "make_output_grid_chunk"),
    "sample_nearest": ("medicai.transforms.spatial.samplers", "sample_nearest"),
    "sample_trilinear": ("medicai.transforms.spatial.samplers", "sample_trilinear"),
    "sample_volume": ("medicai.transforms.spatial.samplers", "sample_volume"),
    "SpatialResample": ("medicai.transforms.spatial.resample_utils", "SpatialResample"),
}


def __getattr__(name: str) -> Any:
    """Load moved spatial helpers without creating import cycles."""
    target = _SPATIAL_HELPERS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    import importlib

    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value

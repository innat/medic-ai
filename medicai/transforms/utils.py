from typing import Sequence

import tensorflow as tf


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
) -> tuple[int, ...]:
    """Normalize axes expressed relative to spatial dimensions only."""
    return normalize_axes(axes, spatial_rank, name="spatial_axes")

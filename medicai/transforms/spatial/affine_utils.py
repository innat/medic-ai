"""Focused affine, orientation, and metadata helpers for spatial transforms.

The public compatibility exports remain available from
medicai.transforms.utils.
"""

from typing import Any

from keras import ops
import numpy as np

from medicai.transforms.utils import validate_affine_matrix


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


def _raise_for_duplicate_axes(current_axes: Any) -> None:
    """Raise for duplicate affine axes when the result is concrete."""
    duplicate = ops.any(
        ops.stack(
            [
                ops.equal(current_axes[0], current_axes[1]),
                ops.equal(current_axes[0], current_axes[2]),
                ops.equal(current_axes[1], current_axes[2]),
            ]
        )
    )
    try:
        duplicate_value = bool(ops.convert_to_numpy(duplicate))
    except (TypeError, ValueError):
        return
    if duplicate_value:
        raise ValueError(
            "Affine orientation is invalid: multiple output axes map to the same world axis."
        )


def _selected_matrix_entries(matrix: Any, row_indices: Any) -> Any:
    """Select one row from each matrix column without N-D gather operations."""
    selectors = ops.one_hot(row_indices, num_classes=3, dtype=matrix.dtype)
    return ops.sum(matrix * selectors, axis=0)


def orientation_from_affine(affine: Any) -> str:
    """Infer a three-letter orientation code from a concrete affine matrix."""
    matrix = ops.cast(validate_affine_matrix(affine)[:3, :3], "float32")
    current_axes = ops.argmax(ops.abs(matrix), axis=0)
    _raise_for_duplicate_axes(current_axes)
    signs = _selected_matrix_entries(matrix, current_axes) >= 0
    axes = np.asarray(ops.convert_to_numpy(current_axes), dtype=np.int32)
    signs = np.asarray(ops.convert_to_numpy(signs), dtype=bool)
    codes = np.where(
        axes == 0,
        np.where(signs, "R", "L"),
        np.where(axes == 1, np.where(signs, "A", "P"), np.where(signs, "S", "I")),
    )
    return "".join(codes.tolist())


def compute_orientation_transform(
    affine: Any,
    target_tensor_axcodes: str,
) -> dict[str, Any]:
    """Compute spatial permutation and flips for a target tensor orientation."""
    axis_to_world = {"R": 0, "L": 0, "A": 1, "P": 1, "S": 2, "I": 2}
    axis_to_sign = {"R": 1, "L": -1, "A": 1, "P": -1, "S": 1, "I": -1}

    matrix = ops.cast(validate_affine_matrix(affine)[:3, :3], "float32")
    current_axes = ops.argmax(ops.abs(matrix), axis=0)
    _raise_for_duplicate_axes(current_axes)
    current_signs = ops.sign(_selected_matrix_entries(matrix, current_axes))
    current_signs = ops.where(current_signs == 0, ops.ones_like(current_signs), current_signs)

    target_axes = [axis_to_world[code] for code in target_tensor_axcodes]
    perm_spatial = ops.stack(
        [
            ops.argmax(ops.cast(ops.equal(current_axes, target_axis), "int32"), axis=0)
            for target_axis in target_axes
        ]
    )
    current_signs_for_output = ops.take(current_signs, perm_spatial, axis=0)
    target_signs = ops.convert_to_tensor(
        [axis_to_sign[code] for code in target_tensor_axcodes],
        dtype="float32",
    )
    flip_axes = ops.reshape(
        ops.where(ops.not_equal(current_signs_for_output, target_signs)),
        [-1],
    )
    return {"perm_spatial": perm_spatial, "flip_axes": flip_axes}


def reoriented_affine(
    affine: Any,
    input_spatial_shape: Any,
    perm_spatial: tuple[int, int, int] | Any,
    flip_axes: tuple[int, ...] | Any,
) -> Any:
    """Update affine metadata for a spatial permutation and flips."""
    affine = ops.cast(validate_affine_matrix(affine), "float32")
    input_spatial_shape = ops.cast(input_spatial_shape, "float32")
    perm_spatial = ops.cast(ops.convert_to_tensor(perm_spatial), "int32")
    flip_axes = ops.cast(ops.convert_to_tensor(flip_axes), "int32")

    flipped_output_mask = ops.scatter(
        indices=ops.expand_dims(flip_axes, axis=1),
        values=ops.ones_like(flip_axes, dtype="float32"),
        shape=(3,),
    )
    signs = 1.0 - 2.0 * flipped_output_mask

    spatial_block = ops.transpose(
        ops.one_hot(perm_spatial, num_classes=3, dtype="float32") * signs[:, None]
    )

    flipped_input_mask = ops.scatter(
        indices=ops.expand_dims(perm_spatial, axis=1),
        values=flipped_output_mask,
        shape=(3,),
    )
    translations = flipped_input_mask * (input_spatial_shape - 1.0)
    top = ops.concatenate([spatial_block, translations[:, None]], axis=1)
    transform = ops.concatenate(
        [top, ops.convert_to_tensor([[0.0, 0.0, 0.0, 1.0]], dtype="float32")],
        axis=0,
    )
    return ops.matmul(affine, transform)

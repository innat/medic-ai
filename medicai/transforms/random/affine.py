"""Backend-neutral helpers for centered affine sampling matrices.

The public random affine transforms use Cartesian parameter names while the
medical tensors remain channel-last: ``x -> W``, ``y -> H``, and ``z -> D``.
Matrices represent forward geometry. Resampling code must use their inverse as
the output-to-input sampling matrix.
"""

import itertools
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

from keras import ops

AFFINE_AXES_2D = ("x", "y")
AFFINE_AXES_3D = ("z", "x", "y")


def compose_affine_matrices(*matrices: Any) -> Any:
    """Compose homogeneous affine matrices from left to right.

    ``compose_affine_matrices(a, b, c)`` returns ``a @ b @ c``. Matrices may
    be unbatched or share a leading batch shape.

    Args:
        *matrices: Square homogeneous matrices with matching dimensions.

    Returns:
        The composed affine matrix.

    Raises:
        ValueError: If no matrices are provided.
    """
    if not matrices:
        raise ValueError("At least one affine matrix is required.")

    result = matrices[0]
    for matrix in matrices[1:]:
        result = ops.matmul(result, matrix)
    return result


def invert_affine_matrix(matrix: Any) -> Any:
    """Return the inverse of one or more homogeneous affine matrices."""
    return ops.linalg.inv(matrix)


def resample_affine_keys(
    data: MutableMapping[str, Any],
    keys: Sequence[str],
    matrix: Any,
    resample: Callable[[Any, Any, str, str, float], Any],
    interpolation: Mapping[str, str],
    fill_mode: Mapping[str, str],
    fill_value: Mapping[str, float],
) -> MutableMapping[str, Any]:
    """Resample selected keys with one shared affine matrix.

    The helper deliberately does not implement interpolation. The caller
    supplies the backend-neutral sampler so 2D and 3D kernels can retain their
    existing behavior while sharing key alignment and option dispatch.

    Args:
        data: Mutable tensor mapping to update in place.
        keys: Keys that should receive the shared geometry.
        matrix: Forward or backward matrix expected by ``resample``.
        resample: Callback with arguments ``(tensor, matrix, interpolation,
            fill_mode, fill_value)``.
        interpolation: Per-key interpolation modes.
        fill_mode: Per-key boundary modes.
        fill_value: Per-key constant boundary values.

    Returns:
        The same mapping passed through ``data``.

    Raises:
        KeyError: If a present key is missing a per-key option.
    """
    for key in keys:
        if key not in data:
            continue
        data[key] = resample(
            data[key],
            matrix,
            interpolation[key],
            fill_mode[key],
            fill_value[key],
        )
    return data


def apply_plane_affine_3d(
    volumes: Any,
    matrices: Any,
    plane_axes: tuple[str, str],
    interpolation: str,
    fill_mode: str,
    fill_value: float,
) -> Any:
    """Apply batched 2D affine matrices to planes of 3D volumes.

    The untouched spatial axis is folded into the batch dimension. This keeps
    separable 3D affine operations on Keras' optimized 2D image kernel.
    """
    axis_indices = {"z": 1, "y": 2, "x": 3}
    if len(plane_axes) != 2 or len(set(plane_axes)) != 2:
        raise ValueError("`plane_axes` must contain two distinct spatial axes.")
    if any(axis not in axis_indices for axis in plane_axes):
        raise ValueError("`plane_axes` must use only 'z', 'y', and 'x'.")

    untouched = next(axis for axis in axis_indices if axis not in plane_axes)
    permutation = (
        0,
        axis_indices[untouched],
        axis_indices[plane_axes[0]],
        axis_indices[plane_axes[1]],
        4,
    )
    transposed = ops.transpose(volumes, permutation)
    batch, folded, height, width, channels = transposed.shape
    if None in (batch, folded, height, width, channels):
        raise ValueError("Plane-wise 3D affine sampling requires static shapes.")

    merged = ops.reshape(transposed, (batch * folded, height, width, channels))
    repeated_matrices = ops.repeat(matrices, folded, axis=0)
    kernel_interpolation = "bilinear" if interpolation.lower() == "trilinear" else interpolation
    sampled = ops.image.affine_transform(
        ops.cast(merged, "float32"),
        repeated_matrices,
        interpolation=kernel_interpolation,
        fill_mode=fill_mode,
        fill_value=fill_value,
    )
    sampled = ops.reshape(sampled, (batch, folded, height, width, channels))

    transposed_positions = {
        axis: position for position, axis in enumerate((untouched, *plane_axes))
    }
    inverse_permutation = (
        0,
        transposed_positions["z"] + 1,
        transposed_positions["y"] + 1,
        transposed_positions["x"] + 1,
        4,
    )
    return ops.transpose(sampled, inverse_permutation)


def _homogeneous_matrix(linear: Any, translation: Any) -> Any:
    """Build a homogeneous matrix from a linear part and an offset."""
    linear = ops.convert_to_tensor(linear)
    translation = ops.cast(ops.convert_to_tensor(translation), linear.dtype)
    rank = linear.shape[-1]

    # Keep the two supported forms explicit.  In particular, TensorFlow can
    # interpret tuple arithmetic on TensorShape values differently from the
    # other Keras backends when the translation is batched.
    linear_rank = len(linear.shape)
    translation_rank = len(translation.shape)
    if linear_rank == 2 and translation_rank == 2:
        batch_size = translation.shape[0]
        linear = ops.broadcast_to(
            ops.expand_dims(linear, axis=0),
            (batch_size, rank, rank),
        )
    elif linear_rank == 3 and translation_rank == 1:
        batch_size = linear.shape[0]
        translation = ops.broadcast_to(
            ops.expand_dims(translation, axis=0),
            (batch_size, rank),
        )
    elif linear_rank != translation_rank + 1:
        raise ValueError(
            "`linear` and `translation` must both be unbatched or use the "
            "same leading batch shape."
        )

    top = ops.concatenate((linear, ops.expand_dims(translation, axis=-1)), axis=-1)
    bottom = ops.concatenate(
        (
            ops.zeros_like(linear[..., :1, :]),
            ops.ones_like(linear[..., :1, :1]),
        ),
        axis=-1,
    )
    return ops.concatenate((top, bottom), axis=-2)


def centered_affine_matrix(
    linear: Any,
    spatial_shape: Sequence[int],
    translation: Any | None = None,
) -> Any:
    """Build a linear affine transform around the spatial image center.

    Args:
        linear: A ``(R, R)`` matrix or a batched ``(..., R, R)`` matrix.
        spatial_shape: Spatial shape in tensor order, ``(H, W)`` or
            ``(D, H, W)``.
        translation: Optional direct translation in the same coordinate order
            as ``spatial_shape``. It is applied after the centered linear
            transform.

    Returns:
        A ``(R + 1, R + 1)`` or batched homogeneous matrix.

    Raises:
        ValueError: If the matrix rank and spatial rank do not match.
    """
    linear = ops.convert_to_tensor(linear)
    spatial_rank = len(spatial_shape)
    if linear.shape[-2:] != (spatial_rank, spatial_rank):
        raise ValueError(
            "`linear` must have trailing shape "
            f"({spatial_rank}, {spatial_rank}); received {linear.shape}."
        )

    dtype = linear.dtype
    center = ops.cast(
        (ops.convert_to_tensor(spatial_shape, dtype="float32") - 1.0) / 2.0,
        dtype,
    )
    centered_translation = center - ops.einsum("...ij,j->...i", linear, center)
    if translation is not None:
        centered_translation = centered_translation + ops.cast(
            ops.convert_to_tensor(translation), dtype
        )
    return _homogeneous_matrix(linear, centered_translation)


def sample_affine_volume(
    volume: Any,
    inverse_matrix: Any,
    interpolation: str,
    fill_mode: str,
    fill_value: float,
    translation: Any | None = None,
) -> Any:
    """Sample one channel-last 3D volume with an output-to-input matrix.

    The matrix operates on coordinates centered at the volume midpoint. Order
    one is used for linear interpolation and order zero for nearest-neighbor
    interpolation, matching the existing random spatial transform kernels.
    """
    depth, height, width, channels = volume.shape
    if channels is None:
        raise ValueError("Affine volume sampling requires a static channel dimension.")

    z, y, x = ops.meshgrid(
        ops.arange(depth),
        ops.arange(height),
        ops.arange(width),
        indexing="ij",
    )
    coordinates = ops.stack(
        [
            ops.cast(z, inverse_matrix.dtype),
            ops.cast(y, inverse_matrix.dtype),
            ops.cast(x, inverse_matrix.dtype),
        ],
        axis=0,
    )
    center = (
        ops.cast(
            ops.convert_to_tensor([depth - 1, height - 1, width - 1]),
            inverse_matrix.dtype,
        )
        / 2.0
    )
    centered = coordinates - ops.reshape(center, (3, 1, 1, 1))
    input_coordinates = ops.einsum("ij,jdhw->idhw", inverse_matrix, centered)
    if translation is not None:
        input_coordinates = input_coordinates + ops.reshape(translation, (3, 1, 1, 1))
    input_coordinates = input_coordinates + ops.reshape(center, (3, 1, 1, 1))
    order = 1 if interpolation.lower() in {"bilinear", "trilinear"} else 0
    return ops.stack(
        [
            ops.image.map_coordinates(
                volume[..., channel],
                input_coordinates,
                order=order,
                fill_mode=fill_mode,
                fill_value=fill_value,
            )
            for channel in range(channels)
        ],
        axis=-1,
    )


def _flatten_batched_gather(volume: Any, batch_indices: Any, indices: Sequence[Any]) -> Any:
    """Gather channel vectors from batched channel-last volumes."""
    shape = ops.shape(volume)
    flat_index = indices[0]
    spatial_sizes = shape[1:-1]
    for axis, index in enumerate(indices[1:], start=1):
        flat_index = flat_index * spatial_sizes[axis] + index

    spatial_size = 1
    for size in spatial_sizes:
        spatial_size = spatial_size * size
    flat_index = batch_indices * spatial_size + flat_index

    flattened = ops.reshape(volume, (-1, shape[-1]))
    gathered = ops.take(flattened, ops.reshape(flat_index, (-1,)), axis=0)
    return ops.reshape(gathered, list(ops.shape(flat_index)) + [shape[-1]])


def _normalize_batched_coordinates(
    volume: Any,
    coordinates: Any,
    fill_mode: str,
) -> tuple[Any, Any]:
    """Apply affine sampler boundary rules to ``(B, ..., rank)`` coordinates."""
    spatial_rank = coordinates.shape[-1]
    shape = ops.shape(volume)
    valid = ops.ones_like(coordinates[..., 0], dtype="bool")
    normalized = []
    for axis in range(spatial_rank):
        size = shape[axis + 1]
        coordinate = coordinates[..., axis]
        size_value = ops.cast(size, coordinate.dtype)
        if fill_mode == "constant":
            valid = ops.logical_and(
                valid,
                ops.logical_and(coordinate >= 0.0, coordinate <= size_value - 1.0),
            )
            normalized.append(ops.clip(coordinate, 0.0, size_value - 1.0))
        elif fill_mode == "mirror":
            static_size = volume.shape[axis + 1]
            if static_size == 1:
                normalized.append(ops.zeros_like(coordinate))
            else:
                period = ops.cast(2 * (int(static_size) - 1), coordinate.dtype)
                reflected = ops.mod(ops.abs(coordinate), period)
                edge = ops.cast(int(static_size) - 1, coordinate.dtype)
                normalized.append(ops.where(reflected <= edge, reflected, period - reflected))
        elif fill_mode == "reflect":
            static_size = volume.shape[axis + 1]
            if static_size == 1:
                normalized.append(ops.zeros_like(coordinate))
            else:
                period = ops.cast(2 * int(static_size), coordinate.dtype)
                reflected = ops.mod(coordinate, period)
                edge = ops.cast(int(static_size), coordinate.dtype)
                normalized.append(
                    ops.where(
                        reflected < edge,
                        reflected,
                        period - 1.0 - reflected,
                    )
                )
        elif fill_mode == "wrap":
            normalized.append(ops.mod(coordinate, size_value))
        else:
            normalized.append(ops.clip(coordinate, 0.0, size_value - 1.0))
    return ops.stack(normalized, axis=-1), valid


def _sample_batched_coordinates(
    volumes: Any,
    coordinates: Any,
    interpolation: str,
    fill_mode: str,
    fill_value: float,
) -> Any:
    """Sample channel-last volumes with broadcasted nearest/linear gathers."""
    spatial_rank = coordinates.shape[-1]
    coordinates, valid = _normalize_batched_coordinates(volumes, coordinates, fill_mode)
    shape = ops.shape(volumes)
    spatial_sizes = [ops.cast(shape[index + 1], coordinates.dtype) for index in range(spatial_rank)]
    batch_indices = ops.arange(shape[0], dtype="int32")
    batch_indices = ops.reshape(batch_indices, [shape[0]] + [1] * spatial_rank)
    batch_indices = ops.broadcast_to(batch_indices, ops.shape(coordinates[..., 0]))

    if interpolation.lower() in {"nearest"}:
        indices = [
            ops.cast(
                ops.clip(
                    ops.round(coordinates[..., axis]),
                    0.0,
                    spatial_sizes[axis] - 1.0,
                ),
                "int32",
            )
            for axis in range(spatial_rank)
        ]
        output = _flatten_batched_gather(volumes, batch_indices, indices)
    else:
        floors = [ops.floor(coordinates[..., axis]) for axis in range(spatial_rank)]
        fractions = [
            (coordinates[..., axis] - floors[axis])[..., None] for axis in range(spatial_rank)
        ]
        output = ops.zeros(list(ops.shape(floors[0])) + [shape[-1]], dtype=volumes.dtype)
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
            output = output + _flatten_batched_gather(volumes, batch_indices, indices) * weight

    if fill_mode == "constant":
        output = ops.where(valid[..., None], output, ops.cast(fill_value, output.dtype))
    return output


def sample_affine_volumes(
    volumes: Any,
    matrices: Any,
    interpolation: str,
    fill_mode: str,
    fill_value: float,
) -> Any:
    """Sample a batch of channel-last 3D volumes without per-sample mapping.

    ``matrices`` may be batched ``(B, 3, 3)`` centered linear matrices,
    ``(B, 3, 4)`` centered affine matrices, or full ``(B, 4, 4)`` homogeneous
    output-to-input matrices in uncentered coordinates.
    """
    depth, height, width, channels = volumes.shape[1:]
    if None in (depth, height, width, channels):
        raise ValueError("Batched affine sampling requires static volume shapes.")

    z, y, x = ops.meshgrid(ops.arange(depth), ops.arange(height), ops.arange(width), indexing="ij")
    grid = ops.stack(
        [ops.cast(z, matrices.dtype), ops.cast(y, matrices.dtype), ops.cast(x, matrices.dtype)],
        axis=-1,
    )
    grid = ops.broadcast_to(
        grid,
        (volumes.shape[0], depth, height, width, 3),
    )
    if matrices.shape[-2:] == (3, 3):
        center = (
            ops.cast(ops.convert_to_tensor([depth - 1, height - 1, width - 1]), matrices.dtype)
            / 2.0
        )
        centered = grid - center
        coordinates = ops.einsum("bij,bdhwj->bdhwi", matrices, centered) + center
    elif matrices.shape[-2:] == (3, 4):
        center = (
            ops.cast(ops.convert_to_tensor([depth - 1, height - 1, width - 1]), matrices.dtype)
            / 2.0
        )
        centered = grid - center
        coordinates = (
            ops.einsum("bij,bdhwj->bdhwi", matrices[..., :3], centered)
            + matrices[..., 3][:, None, None, None, :]
            + center
        )
    elif matrices.shape[-2:] == (4, 4):
        homogeneous = ops.concatenate((grid, ops.ones_like(grid[..., :1])), axis=-1)
        coordinates = ops.einsum("bij,bdhwj->bdhwi", matrices, homogeneous)[..., :3]
    else:
        raise ValueError("`matrices` must have trailing shape (3, 3), (3, 4), or (4, 4).")

    return _sample_batched_coordinates(
        ops.cast(volumes, "float32"),
        coordinates,
        interpolation,
        fill_mode,
        fill_value,
    )

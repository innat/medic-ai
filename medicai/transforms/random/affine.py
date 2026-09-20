"""Backend-neutral helpers for centered affine sampling matrices.

The public random affine transforms use Cartesian parameter names while the
medical tensors remain channel-last: ``x -> W``, ``y -> H``, and ``z -> D``.
Matrices represent forward geometry. Resampling code must use their inverse as
the output-to-input sampling matrix.
"""

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


def _homogeneous_matrix(linear: Any, translation: Any) -> Any:
    """Build a homogeneous matrix from a linear part and an offset."""
    linear = ops.convert_to_tensor(linear)
    translation = ops.cast(ops.convert_to_tensor(translation), linear.dtype)
    rank = linear.shape[-1]
    translation = ops.broadcast_to(translation, linear.shape[:-2] + (rank,))

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
    zero = ops.zeros_like(center)
    identity = ops.eye(spatial_rank, dtype=dtype)
    identity = ops.broadcast_to(
        identity,
        linear.shape[:-2] + (spatial_rank, spatial_rank),
    )
    centered = compose_affine_matrices(
        _homogeneous_matrix(identity, center),
        _homogeneous_matrix(linear, zero),
        _homogeneous_matrix(identity, -center),
    )

    if translation is None:
        return centered

    return compose_affine_matrices(
        _homogeneous_matrix(identity, translation),
        centered,
    )


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
    center = ops.cast(
        ops.convert_to_tensor([depth - 1, height - 1, width - 1]),
        inverse_matrix.dtype,
    ) / 2.0
    centered = coordinates - ops.reshape(center, (3, 1, 1, 1))
    input_coordinates = ops.einsum("ij,jdhw->idhw", inverse_matrix, centered)
    if translation is not None:
        input_coordinates = input_coordinates + ops.reshape(
            translation, (3, 1, 1, 1)
        )
    input_coordinates = input_coordinates + ops.reshape(center, (3, 1, 1, 1))
    input_coordinates = input_coordinates + ops.reshape(center, (3, 1, 1, 1))
    order = 1 if interpolation.lower() == "bilinear" else 0
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

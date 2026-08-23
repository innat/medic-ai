"""Backend-aware voxel sampling kernels for spatial resampling."""

from typing import Any

import keras
from keras import ops

from medicai.transforms.utils import _get_static_shape_tuple, get_tensor_rank


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
    shape = ops.shape(volume)
    spatial_width = shape[2]
    spatial_height = shape[1]
    linear_indices = (
        safe_indices[:, 0] * spatial_height * spatial_width
        + safe_indices[:, 1] * spatial_width
        + safe_indices[:, 2]
    )
    channels = _get_static_shape_tuple(volume)[-1]
    if channels is None:
        raise ValueError("Volume channel count must be statically known for sampling.")
    flat_volume = ops.reshape(volume, [-1, channels])
    gathered = ops.take(flat_volume, linear_indices, axis=0)
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
    shape = ops.cast(ops.convert_to_tensor(ops.shape(volume)[:3]), indices.dtype)
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
    original_dtype = keras.backend.standardize_dtype(volume.dtype)
    output_dtype = (
        original_dtype
        if original_dtype in {"float16", "float32", "float64", "bfloat16"}
        else "float32"
    )
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
    shape = ops.cast(ops.convert_to_tensor(ops.shape(volume)[:3]), d0.dtype)

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

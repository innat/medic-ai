import os
import warnings
from typing import Any, List, Sequence, Tuple

import numpy as np


def hide_warnings():
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def ensure_tuple_rep(val: Any, rep: int) -> Tuple[Any, ...]:
    """Ensure `val` is a tuple of length `rep`."""
    if isinstance(val, (int, float)):
        return (val,) * rep
    if len(val) == rep:
        return tuple(val)
    raise ValueError(f"Length of `val` ({len(val)}) must match `rep` ({rep}).")


def fall_back_tuple(val: Any, fallback: Sequence[int]) -> Tuple[int, ...]:
    """Ensure `val` is a tuple of the same length as `fallback`."""
    if val is None:
        return tuple(fallback)
    if isinstance(val, int):
        return (val,) * len(fallback)
    if len(val) != len(fallback):
        raise ValueError(f"Length of `val` ({len(val)}) must match `fallback` ({len(fallback)}).")
    return tuple(val)


def _get_scan_interval(
    image_size: Sequence[int],
    roi_size: Sequence[int],
    num_spatial_dims: int,
    overlap: Sequence[float],
) -> Tuple[int, ...]:
    """Compute scan intervals based on image size, roi size, and overlap."""
    scan_interval = []
    for i, o in zip(range(num_spatial_dims), overlap):
        if roi_size[i] == image_size[i]:
            scan_interval.append(roi_size[i])
        else:
            interval = int(roi_size[i] * (1 - o))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def dense_patch_slices(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    scan_interval: Sequence[int],
    return_slice: bool = True,
) -> List[Tuple[slice, ...]]:
    num_spatial_dims = len(image_size)

    # Calculate the number of patches along each dimension
    scan_num = []
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = (image_size[i] + scan_interval[i] - 1) // scan_interval[
                i
            ]  # Equivalent to math.ceil
            scan_dim = next(
                (d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i]),
                None,
            )
            scan_num.append(scan_dim + 1 if scan_dim is not None else 1)

    # Generate start indices for each dimension
    starts = []
    for dim in range(num_spatial_dims):
        dim_starts = []
        for idx in range(scan_num[dim]):
            start_idx = idx * scan_interval[dim]
            start_idx -= max(start_idx + patch_size[dim] - image_size[dim], 0)
            dim_starts.append(start_idx)
        starts.append(dim_starts)

    # Generate all combinations of start indices
    out = []
    from itertools import product

    for start_indices in product(*starts):
        if return_slice:
            out.append(
                tuple(slice(start, start + patch_size[d]) for d, start in enumerate(start_indices))
            )
        else:
            out.append(
                tuple((start, start + patch_size[d]) for d, start in enumerate(start_indices))
            )

    return out


def compute_importance_map(
    patch_size: Sequence[int],
    mode: str = "constant",
    sigma_scale: Sequence[float] = (0.125,),
    dtype=np.float32,
) -> np.ndarray:
    """Compute importance map for blending."""
    if mode == "constant":
        return np.ones(patch_size, dtype=dtype)

    elif mode == "gaussian":
        sigma = [s * p for s, p in zip(sigma_scale, patch_size)]
        grid = np.meshgrid(*[np.arange(p, dtype=dtype) for p in patch_size], indexing="ij")
        center = [(p - 1) / 2 for p in patch_size]
        dist = np.sqrt(sum((g - c) ** 2 for g, c in zip(grid, center)))
        return np.exp(-0.5 * (dist / sigma) ** 2)

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def get_valid_patch_size(image_size: Sequence[int], patch_size: Sequence[int]) -> Tuple[int, ...]:
    """
    Ensure the patch size is valid (i.e., patch_size <= image_size).
    """
    return tuple(min(p, i) for p, i in zip(patch_size, image_size))


def _crop_output(
    output: np.ndarray, pad_size: Sequence[Sequence[int]], original_size: Sequence[int]
) -> np.ndarray:
    """
    Crop the output to remove padding.

    Args:
        output: Output array with shape (batch_size, *padded_size, channels).
        pad_size: Padding applied to the input tensor.
        original_size: Original spatial size of the input tensor.

    Returns:
        Cropped output array with shape (batch_size, *original_size, channels).
    """
    crop_slices = [slice(None)]  # Keep batch dimension
    for i in range(len(original_size)):
        start = pad_size[i + 1][0]  # Skip batch dimension
        end = start + original_size[i]
        crop_slices.append(slice(start, end))
    crop_slices.append(slice(None))  # Keep channel dimension

    # Convert the list of slices to a tuple for proper indexing
    return output[tuple(crop_slices)]

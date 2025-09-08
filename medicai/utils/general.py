from typing import Any, List, Sequence, Tuple

import numpy as np


def hide_warnings():
    import logging
    import os
    import sys
    import warnings

    # Disable Python warnings
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn
    warnings.simplefilter(action="ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Disable Python logging
    logging.disable(logging.WARNING)
    logging.getLogger("tensorflow").disabled = True

    # TensorFlow environment variables
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"

    # Disable TensorFlow debugging information
    if "tensorflow" in sys.modules:
        tf = sys.modules["tensorflow"]
        tf.get_logger().setLevel("ERROR")
        tf.autograph.set_verbosity(0)

    # Disable ABSL (Ten1orFlow dependency) logging
    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
        # Redirect ABSL logs to null
        absl.logging.get_absl_handler().python_handler.stream = open(os.devnull, "w")
    except (ImportError, AttributeError):
        pass


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


def get_scan_interval(
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


def crop_output(
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


def resize_volumes(volumes, depth, height, width, method="trilinear"):
    from keras import ops

    def trilinear_resize(volumes, depth, height, width):
        original_dtype = volumes.dtype
        volumes = ops.cast(volumes, "float32")
        batch_size, in_d, in_h, in_w, channels = ops.shape(volumes)

        # --- Depth (axis=1) ---
        z = ops.linspace(0.0, ops.cast(in_d - 1, "float32"), depth)  # (out_d,)
        z0 = ops.cast(ops.floor(z), "int32")
        z1 = ops.minimum(z0 + 1, in_d - 1)
        dz = z - ops.cast(z0, "float32")  # (out_d,)

        # Gather along depth (axis=1) using 1D indices -> result has depth = out_d
        lower_d = ops.take(volumes, z0, axis=1)  # (B, out_d, H,  W, C)
        upper_d = ops.take(volumes, z1, axis=1)  # (B, out_d, H,  W, C)

        dz = ops.reshape(dz, (1, depth, 1, 1, 1))  # broadcast shape
        interp_d = (1.0 - dz) * lower_d + dz * upper_d  # (B, out_d, H, W, C)

        # --- Height (axis=2) ---
        y = ops.linspace(0.0, ops.cast(in_h - 1, "float32"), height)  # (out_h,)
        y0 = ops.cast(ops.floor(y), "int32")
        y1 = ops.minimum(y0 + 1, in_h - 1)
        dy = y - ops.cast(y0, "float32")

        lower_h = ops.take(interp_d, y0, axis=2)  # (B, out_d, out_h, W, C)
        upper_h = ops.take(interp_d, y1, axis=2)  # (B, out_d, out_h, W, C)

        dy = ops.reshape(dy, (1, 1, height, 1, 1))
        interp_h = (1.0 - dy) * lower_h + dy * upper_h  # (B, out_d, out_h, W, C)

        # --- Width (axis=3) ---
        x = ops.linspace(0.0, ops.cast(in_w - 1, "float32"), width)  # (out_w,)
        x0 = ops.cast(ops.floor(x), "int32")
        x1 = ops.minimum(x0 + 1, in_w - 1)
        dx = x - ops.cast(x0, "float32")

        lower_w = ops.take(interp_h, x0, axis=3)  # (B, out_d, out_h, out_w, C)
        upper_w = ops.take(interp_h, x1, axis=3)  # (B, out_d, out_h, out_w, C)

        dx = ops.reshape(dx, (1, 1, 1, width, 1))
        out = (1.0 - dx) * lower_w + dx * upper_w  # (B, out_d, out_h, out_w, C)
        out = ops.cast(out, original_dtype)
        return out

    if method == "trilinear":
        return trilinear_resize(volumes, depth, height, width)
    else:
        raise ValueError(f"Unsupported resize method: {method}")

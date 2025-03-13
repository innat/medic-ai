import tensorflow as tf
import numpy as np
from typing import Sequence, Tuple, List
from typing import Any, Callable, Sequence, Mapping, Optional, Tuple, Union

def sliding_window_inference(
    inputs: tf.Tensor,
    num_classes: int,
    roi_size: Sequence[int],
    sw_batch_size: int,
    predictor: Callable[..., tf.Tensor],
    overlap: Union[Sequence[float], float] = 0.25,
    mode: str = "constant",
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: str = "constant",
    cval: float = 0.0,
    roi_weight_map: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """
    Sliding window inference in TensorFlow, mimicking MONAI's implementation.

    Args:
        inputs: Input tensor with shape (batch_size, *spatial_dims, channels).
        roi_size: Spatial window size for inferences.
        sw_batch_size: Batch size for sliding window inference.
        predictor: Callable that takes a patch of input and returns predictions.
        overlap: Overlap ratio between windows (default: 0.25).
        mode: Blending mode for overlapping windows ("constant" or "gaussian").
        sigma_scale: Standard deviation coefficient for Gaussian blending.
        padding_mode: Padding mode for inputs when roi_size > input size.
        cval: Padding value for "constant" padding mode.
        roi_weight_map: Pre-computed weight map for each ROI (optional).

    Returns:
        The output tensor.
    """
    # Ensure overlap and sigma_scale are sequences
    num_spatial_dims = len(inputs.shape) - 2  # Exclude batch and channel dimensions
    overlap = ensure_tuple_rep(overlap, num_spatial_dims)
    sigma_scale = ensure_tuple_rep(sigma_scale, num_spatial_dims)

    # Validate overlap values
    for o in overlap:
        if o < 0 or o >= 1:
            raise ValueError(f"Overlap must be >= 0 and < 1, got {overlap}.")

    # Determine image spatial size and batch size
    batch_size, *image_size_, _ = inputs.shape  # Unpack for channels-last format
    roi_size = fall_back_tuple(roi_size, image_size_)

    # Pad input if necessary
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(num_spatial_dims):
        diff = max(roi_size[k] - image_size_[k], 0)
        half = diff // 2
        pad_size.append([half, diff - half])
    pad_size = [[0, 0]] + pad_size + [[0, 0]]  # Add padding for batch and channel dimensions

    if any(p for pair in pad_size for p in pair):
        inputs = tf.pad(inputs, pad_size, mode=padding_mode.upper(), constant_values=cval)

    # Compute scan intervals
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval)

    # Create importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and roi_weight_map is not None:
        importance_map = roi_weight_map
    else:
        importance_map = compute_importance_map(
            valid_patch_size, mode=mode, sigma_scale=sigma_scale, dtype=inputs.dtype
        )
        if len(importance_map.shape) == num_spatial_dims:
            importance_map = tf.expand_dims(tf.expand_dims(importance_map, -1), 0)  # Add batch and channel dims

    # Initialize output and count maps as NumPy arrays
    output_shape = [batch_size] + list(image_size) + [num_classes]
    output_image = np.zeros(output_shape, dtype='float32')
    count_map = np.zeros([1] + list(image_size) + [1], dtype='float32')

    # Apply sliding window inference in batches
    for i in range(0, len(slices), sw_batch_size):
        batch_slices = slices[i:i + sw_batch_size]
        patch_list = []
        for slice_idx in batch_slices:
            full_slice = (slice(None),) + slice_idx + (slice(None),)
            patch = inputs[full_slice]
            patch_list.append(patch)
        patches = tf.concat(patch_list, axis=0)  # Stack patches along batch dimension

        # Predict on the batch of patches
        pred = predictor(patches).numpy()  # Convert predictions to NumPy

        # Resize importance map if necessary
        if pred.shape[1:-1] != roi_size:  # Exclude batch and channel dimensions
            importance_map_resized = tf.image.resize(
                importance_map, pred.shape[1:-1], method="nearest"
                ).numpy()
        else:
            importance_map_resized = importance_map.numpy()

        # Accumulate predictions using NumPy
        for j, slice_idx in enumerate(batch_slices):
            output_slice = (slice(None),) + slice_idx + (slice(None),)
            output_image[output_slice] += pred[j] * importance_map_resized
            count_map[output_slice] += importance_map_resized

    # Normalize output by count map
    output_image /= count_map

    # Remove padding if necessary
    if any(p for pair in pad_size for p in pair):
        output_image = _crop_output(output_image, pad_size, image_size_)

    # Convert the final output back to a TensorFlow tensor
    return tf.convert_to_tensor(output_image)


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
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: Sequence[float]
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
    return_slice: bool = True
) -> List[Tuple[slice, ...]]:
    num_spatial_dims = len(image_size)

    # Calculate the number of patches along each dimension
    scan_num = []
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = (image_size[i] + scan_interval[i] - 1) // scan_interval[i]  # Equivalent to math.ceil
            scan_dim = next(
                (d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i]),
                None
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
            out.append(tuple(
                slice(start, start + patch_size[d]) for d, start in enumerate(start_indices)
            ))
        else:
            out.append(tuple(
                (start, start + patch_size[d]) for d, start in enumerate(start_indices)
            ))

    return out


def compute_importance_map(
    patch_size: Sequence[int], mode: str = "constant", sigma_scale: Sequence[float] = (0.125,), dtype: tf.DType = tf.float32
) -> tf.Tensor:
    """Compute importance map for blending."""
    if mode == "constant":
        return tf.ones(patch_size, dtype=dtype)
    elif mode == "gaussian":
        sigma = [s * p for s, p in zip(sigma_scale, patch_size)]
        grid = tf.meshgrid(*[tf.range(p, dtype=dtype) for p in patch_size])
        center = [(p - 1) / 2 for p in patch_size]
        dist = tf.sqrt(sum((g - c) ** 2 for g, c in zip(grid, center)))
        return tf.exp(-0.5 * (dist / sigma) ** 2)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def get_valid_patch_size(image_size: Sequence[int], patch_size: Sequence[int]) -> Tuple[int, ...]:
    """
    Ensure the patch size is valid (i.e., patch_size <= image_size).
    """
    return tuple(min(p, i) for p, i in zip(patch_size, image_size))


def _crop_output(output: np.ndarray, pad_size: Sequence[Sequence[int]], original_size: Sequence[int]) -> np.ndarray:
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
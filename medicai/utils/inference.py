from typing import Optional, Sequence, Union

import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

from medicai.utils.general import (
    _crop_output,
    _get_scan_interval,
    compute_importance_map,
    dense_patch_slices,
    ensure_tuple_rep,
    fall_back_tuple,
    get_valid_patch_size,
)


class SlidingWindowInference:
    def __init__(
        self,
        model,
        num_classes: int,
        roi_size: Sequence[int],
        sw_batch_size: int,
        overlap: Union[Sequence[float], float] = 0.25,
        mode: str = "constant",
        sigma_scale: Union[Sequence[float], float] = 0.125,
        padding_mode: str = "constant",
        cval: float = 0.0,
        roi_weight_map=None,
    ):
        """
        Initializes the SlidingWindowInference object.

        Args:
            model: Callable that takes a patch of input and returns predictions.
            num_classes (int): The number of output classes.
            roi_size (Sequence[int]): Spatial window size for inferences.
            sw_batch_size (int): Batch size for sliding window inference.
            overlap (Union[Sequence[float], float]): Overlap ratio between windows
                (default: 0.25). Can be a single float for isotropic overlap
                or a sequence of floats for anisotropic overlap.
            mode (str): Blending mode for overlapping windows. Options are:
                "constant" or "gaussian" (default: "constant").
            sigma_scale (Union[Sequence[float], float]): Standard deviation
                coefficient for Gaussian blending. Only used if ``mode`` is
                "gaussian". Can be a single float or a sequence of floats.
                (default: 0.125).
            padding_mode (str): Padding mode for inputs when ``roi_size`` is
                larger than the input size. Options are numpy padding modes
                (e.g., "constant", "reflect", "replicate") (default: "constant").
            cval (float): Padding value for "constant" padding mode (default: 0.0).
            roi_weight_map (Optional[np.ndarray]): Pre-computed weight map for
                each ROI. If None, it will be computed based on the ``mode``.
                Should have the same spatial dimensions as ``roi_size``.
                (default: None).
        """
        self.model = model
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode
        self.sigma_scale = sigma_scale
        self.padding_mode = padding_mode
        self.cval = cval
        self.roi_weight_map = roi_weight_map

    def __call__(self, inputs):
        """
        Call method to perform sliding window inference.

        Args:
            inputs (np.ndarray): Input tensor with shape
                (batch_size, *spatial_dims, channels).

        Returns:
            np.ndarray: The output tensor with the same batch size and
                spatial dimensions as the input, and the number of channels
                equal to ``num_classes``.
        """
        return sliding_window_inference(
            inputs=inputs,
            model=self.model,
            num_classes=self.num_classes,
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            overlap=self.overlap,
            mode=self.mode,
            sigma_scale=self.sigma_scale,
            padding_mode=self.padding_mode,
            cval=self.cval,
            roi_weight_map=self.roi_weight_map,
        )


def sliding_window_inference(
    inputs,
    model,
    num_classes: Optional[int],
    roi_size: Sequence[int],
    sw_batch_size: int,
    overlap: Union[Sequence[float], float] = 0.25,
    mode: str = "constant",
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: str = "constant",
    cval: float = 0.0,
    roi_weight_map=None,
):
    """
    Sliding window inference in TensorFlow, mimicking MONAI's implementation.

    Args:
        inputs (np.ndarray): Input tensor with shape
            (batch_size, *spatial_dims, channels).
        model: Callable that takes a patch of input and returns predictions.
        num_classes (Optional[int]): The number of output classes. If None,
            it will be inferred from the model's output shape.
        roi_size (Sequence[int]): Spatial window size for inferences.
        sw_batch_size (int): Batch size for sliding window inference.
        overlap (Union[Sequence[float], float]): Overlap ratio between windows
            (default: 0.25). Can be a single float for isotropic overlap
            or a sequence of floats for anisotropic overlap.
        mode (str): Blending mode for overlapping windows. Options are:
            "constant" or "gaussian" (default: "constant").
        sigma_scale (Union[Sequence[float], float]): Standard deviation
            coefficient for Gaussian blending. Only used if ``mode`` is
            "gaussian". Can be a single float or a sequence of floats.
            (default: 0.125).
        padding_mode (str): Padding mode for inputs when ``roi_size`` is
            larger than the input size. Options are numpy padding modes
            (e.g., "constant", "reflect", "replicate") (default: "constant").
        cval (float): Padding value for "constant" padding mode (default: 0.0).
        roi_weight_map (Optional[np.ndarray]): Pre-computed weight map for
            each ROI. If None, it will be computed based on the ``mode``.
            Should have the same spatial dimensions as ``roi_size``.
            (default: None).

    Returns:
        np.ndarray: The output tensor with the same batch size and
            spatial dimensions as the input, and the number of channels
            equal to ``num_classes``.
    """
    # Ensure overlap and sigma_scale are sequences
    inputs = np.array(inputs) if not isinstance(inputs, np.ndarray) else inputs
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
        inputs = np.pad(inputs, pad_size, mode=padding_mode.lower(), constant_values=cval)

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
            importance_map = np.expand_dims(
                np.expand_dims(importance_map, -1), 0
            )  # Add batch and channel dims

    # Initialize output and count maps as NumPy arrays
    num_classes = num_classes or model.output_shape[-1]
    output_shape = [batch_size] + list(image_size) + [num_classes]
    output_image = np.zeros(output_shape, dtype="float32")
    count_map = np.zeros([1] + list(image_size) + [1], dtype="float32")

    # Apply sliding window inference in batches
    for i in tqdm(range(0, len(slices), sw_batch_size), desc=f"Total patch {len(slices)}"):
        batch_slices = slices[i : i + sw_batch_size]
        patch_list = []
        for slice_idx in batch_slices:
            full_slice = (slice(None),) + slice_idx + (slice(None),)
            patch = inputs[full_slice]
            patch_list.append(patch)
        patches = np.concatenate(patch_list, axis=0)  # Stack patches along batch dimension

        # Predict on the batch of patches
        pred = model.predict(patches, verbose=0)

        # Resize importance map if necessary
        if pred.shape[1:-1] != roi_size:  # Exclude batch and channel dimensions
            _, d, h, w, _ = importance_map.shape
            target_shape = pred.shape[1:-1]
            scale_factors = (1, target_shape[0] / d, target_shape[1] / h, target_shape[2] / w, 1)
            importance_map_resized = zoom(importance_map, scale_factors, order=0)
        else:
            importance_map_resized = importance_map

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

    return output_image

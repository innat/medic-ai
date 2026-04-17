from typing import Optional, Sequence, Union

import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

from .swi_utils import (
    compute_importance_map,
    crop_output,
    dense_patch_slices,
    ensure_tuple_rep,
    fall_back_tuple,
    get_scan_interval,
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


def sliding_window_inference_old(
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
    scan_interval = get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
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

        # GitHub: https://github.com/keras-team/keras/issues/21167
        # padded if needed - its useful for XLA complilation to support tpu or jax backend.
        bs_actual = patches.shape[0]
        bs_target = sw_batch_size
        if bs_actual < bs_target:
            batch_pad_size = ((0, bs_target - bs_actual), (0, 0), (0, 0), (0, 0), (0, 0))
            patches = np.pad(patches, batch_pad_size, mode="constant", constant_values=0)

        # Predict on the batch of patches
        pred = model.predict(patches, verbose=0)
        pred = pred[:bs_actual]

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
        output_image = crop_output(output_image, pad_size, image_size_)

    return output_image


def extract_patches(
    inputs: np.ndarray,
    roi_size: Sequence[int],
    overlap: Union[Sequence[float], float] = 0.25,
    mode: str = "constant",
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: str = "constant",
    cval: float = 0.0,
    roi_weight_map: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict]:
    """
    Extracts patches from the inputs for sliding window inference.

    Args:
        inputs (np.ndarray): Input tensor with shape (batch_size, *spatial_dims, channels).
        roi_size (Sequence[int]): Spatial window size for inferences.
        overlap (Union[Sequence[float], float]): Overlap ratio between windows (default: 0.25).
        mode (str): Blending mode for overlapping windows ("constant" or "gaussian").
        sigma_scale (Union[Sequence[float], float]): Standard deviation coefficient for Gaussian blending.
        padding_mode (str): Padding mode for inputs.
        cval (float): Padding value.
        roi_weight_map (Optional[np.ndarray]): Pre-computed weight map for each ROI.

    Returns:
        tuple[np.ndarray, dict]: 
            - patches: An array of un-merged patches extracted from the input.
            - info: A dictionary containing structural metadata for merging.
    """
    inputs = np.array(inputs) if not isinstance(inputs, np.ndarray) else inputs
    num_spatial_dims = len(inputs.shape) - 2
    overlap = ensure_tuple_rep(overlap, num_spatial_dims)
    sigma_scale = ensure_tuple_rep(sigma_scale, num_spatial_dims)

    for o in overlap:
        if o < 0 or o >= 1:
            raise ValueError(f"Overlap must be >= 0 and < 1, got {overlap}.")

    batch_size, *image_size_, _ = inputs.shape
    roi_size = fall_back_tuple(roi_size, image_size_)

    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(num_spatial_dims):
        diff = max(roi_size[k] - image_size_[k], 0)
        half = diff // 2
        pad_size.append([half, diff - half])
    pad_size = [[0, 0]] + pad_size + [[0, 0]]

    if any(p for pair in pad_size for p in pair):
        inputs = np.pad(inputs, pad_size, mode=padding_mode.lower(), constant_values=cval)

    scan_interval = get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval)

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
            )

    patch_list = []
    for slice_idx in slices:
        full_slice = (slice(None),) + slice_idx + (slice(None),)
        patch_list.append(inputs[full_slice])
    
    patches = np.concatenate(patch_list, axis=0)

    info = {
        "image_size_": image_size_,
        "image_size": image_size,
        "pad_size": pad_size,
        "slices": slices,
        "importance_map": importance_map,
        "batch_size": batch_size,
        "num_spatial_dims": num_spatial_dims
    }
    return patches, info


def predict_patches(
    patches: np.ndarray,
    model,
    sw_batch_size: int,
    roi_size: Sequence[int],
    importance_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predicts on extracted patches using sliding window batch inference limit.

    Args:
        patches (np.ndarray): Extracted input patches.
        model: Callable model or object with `predict` method.
        sw_batch_size (int): Batch size for inference.
        roi_size (Sequence[int]): Spatial window size.
        importance_map (np.ndarray): Importance array for blending.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - predictions: Array of corresponding predicted patches.
            - importance_map_resized: Resized importance map matching output spatial scale.
    """
    total_samples = patches.shape[0]
    preds = []
    importance_map_resized = None

    for i in tqdm(range(0, total_samples, sw_batch_size), desc=f"Total patch {total_samples}"):
        batch_patches = patches[i : i + sw_batch_size]

        bs_actual = batch_patches.shape[0]
        bs_target = sw_batch_size
        if bs_actual < bs_target:
            batch_pad_size = ((0, bs_target - bs_actual), *[(0, 0)] * (len(batch_patches.shape) - 1))
            batch_patches = np.pad(batch_patches, batch_pad_size, mode="constant", constant_values=0)

        pred = model.predict(batch_patches, verbose=0)
        pred = pred[:bs_actual]
        preds.append(pred)

        if importance_map_resized is None:
            if pred.shape[1:-1] != tuple(roi_size):
                _, d, h, w, _ = importance_map.shape
                target_shape = pred.shape[1:-1]
                scale_factors = (1, target_shape[0] / d, target_shape[1] / h, target_shape[2] / w, 1)
                importance_map_resized = zoom(importance_map, scale_factors, order=0)
            else:
                importance_map_resized = importance_map

    predictions = np.concatenate(preds, axis=0) if preds else np.array([])
    return predictions, importance_map_resized


def merge_patches(
    predictions: np.ndarray,
    info: dict,
    importance_map_resized: np.ndarray,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Merges batch predictions back into the original volume layout.

    Args:
        predictions (np.ndarray): Predicted patches.
        info (dict): Metadata info from extract_patches.
        importance_map_resized (np.ndarray): Resized importance blend map.
        num_classes (Optional[int]): Number of output classes. 

    Returns:
        np.ndarray: Final reconstructed and blended volumetric array.
    """
    batch_size = info["batch_size"]
    image_size = info["image_size"]
    image_size_ = info["image_size_"]
    pad_size = info["pad_size"]
    slices = info["slices"]

    num_classes = num_classes or predictions.shape[-1]
    output_shape = [batch_size] + list(image_size) + [num_classes]
    
    output_image = np.zeros(output_shape, dtype=np.float32)
    count_map = np.zeros([1] + list(image_size) + [1], dtype=np.float32)

    for i, slice_idx in enumerate(slices):
        output_slice = (slice(None),) + slice_idx + (slice(None),)
        pred_slice = predictions[i * batch_size : (i + 1) * batch_size]
        output_image[output_slice] += pred_slice * importance_map_resized
        count_map[output_slice] += importance_map_resized

    output_image /= count_map

    if any(p for pair in pad_size for p in pair):
        output_image = crop_output(output_image, pad_size, image_size_)

    return output_image


def sliding_window_inference(
    inputs: np.ndarray,
    model,
    num_classes: Optional[int],
    roi_size: Sequence[int],
    sw_batch_size: int,
    overlap: Union[Sequence[float], float] = 0.25,
    mode: str = "constant",
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: str = "constant",
    cval: float = 0.0,
    roi_weight_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sliding window inference in TensorFlow, mimicking MONAI's implementation.
    This acts as a clean wrapper successively executing modular steps.

    Args:
        inputs (np.ndarray): Input tensor with shape (batch_size, *spatial_dims, channels).
        model: Callable that takes a patch of input and returns predictions.
        num_classes (Optional[int]): The number of output classes.
        roi_size (Sequence[int]): Spatial window size.
        sw_batch_size (int): Batch size for sliding window inference.
        overlap (Union[Sequence[float], float]): Overlap ratio between windows (default: 0.25).
        mode (str): Blending mode ("constant" or "gaussian").
        sigma_scale (Union[Sequence[float], float]): Std dev coefficient for Gaussian blending.
        padding_mode (str): Padding mode.
        cval (float): Padding value.
        roi_weight_map (Optional[np.ndarray]): Pre-computed weight map.

    Returns:
        np.ndarray: Reconstructed output tensor.
    """
    patches, info = extract_patches(
        inputs=inputs, 
        roi_size=roi_size, 
        overlap=overlap, 
        mode=mode, 
        sigma_scale=sigma_scale, 
        padding_mode=padding_mode, 
        cval=cval, 
        roi_weight_map=roi_weight_map
    )
    predictions, importance_map_resized = predict_patches(
        patches=patches, 
        model=model, 
        sw_batch_size=sw_batch_size, 
        roi_size=roi_size, 
        importance_map=info["importance_map"]
    )
    return merge_patches(
        predictions=predictions, 
        info=info, 
        importance_map_resized=importance_map_resized, 
        num_classes=num_classes
    )

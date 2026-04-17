from typing import Generator, Optional, Sequence, Union

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
        model: Object with a ``predict(x, verbose=...)`` method. Takes a patch of
            input and returns predictions.
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
    np.divide(output_image, count_map, out=output_image, where=count_map != 0)

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
    Prepares inputs for sliding window inference by computing padding,
    overlap slicing coordinates, and the importance map.

    This function does **not** materialise all patches into a single array.
    Instead it returns the padded input volume together with metadata so that
    patches can be lazily extracted in batches by ``predict_patches``.

    Args:
        inputs (np.ndarray): Input tensor with shape
            ``(batch_size, *spatial_dims, channels)``.  Supports arbitrary
            spatial dimensionality (2-D or 3-D).
        roi_size (Sequence[int]): Spatial window size for inferences.
        overlap (Union[Sequence[float], float]): Overlap ratio between
            windows (default: 0.25).
        mode (str): Blending mode for overlapping windows.  Options are
            ``"constant"`` or ``"gaussian"`` (default: ``"constant"``).
        sigma_scale (Union[Sequence[float], float]): Standard deviation
            coefficient for Gaussian blending (default: 0.125).
        padding_mode (str): Padding mode for inputs when ``roi_size``
            exceeds the input size (default: ``"constant"``).
        cval (float): Fill value for ``"constant"`` padding (default: 0.0).
        roi_weight_map (Optional[np.ndarray]): Pre-computed weight map
            for each ROI.  If ``None``, computed from ``mode``.

    Returns:
        tuple[np.ndarray, dict]:
            - **padded_inputs** -- The (potentially padded) input array.
            - **info** -- Dictionary with keys ``original_image_size``,
              ``padded_image_size``, ``pad_size``, ``slices``, ``importance_map``,
              ``batch_size``, ``num_spatial_dims``, ``roi_size``.
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
            importance_map = np.expand_dims(np.expand_dims(importance_map, -1), 0)

    info = {
        "original_image_size": image_size_,
        "padded_image_size": image_size,
        "pad_size": pad_size,
        "slices": slices,
        "importance_map": importance_map,
        "batch_size": batch_size,
        "num_spatial_dims": num_spatial_dims,
        "roi_size": roi_size,
    }
    return inputs, info


def predict_patches(
    padded_inputs: np.ndarray,
    info: dict,
    model,
    sw_batch_size: int,
) -> Generator[tuple[np.ndarray, list, np.ndarray], None, None]:
    """
    Generator that lazily extracts patches, runs inference, and yields
    results one batch at a time.

    Only one ``sw_batch_size``-worth of patches and predictions is held
    in memory at any given time, keeping peak memory proportional to
    ``O(sw_batch_size * patch_volume)`` rather than
    ``O(total_patches * patch_volume)``.

    Args:
        padded_inputs (np.ndarray): Padded input array returned by
            ``extract_patches``.
        info (dict): Metadata dictionary returned by ``extract_patches``.
        model: Object with a ``predict(x, verbose=...)`` method.
        sw_batch_size (int): Maximum number of patches per inference call.

    Yields:
        tuple[np.ndarray, list, np.ndarray]:
            - **pred_batch** -- Predictions for the current batch,
              shape ``(actual_batch, *output_spatial, num_classes)``.
            - **batch_slices** -- List of spatial slice tuples that
              locate each patch within the padded volume.
            - **importance_map_resized** -- Importance map matching the
              prediction spatial dimensions.
    """
    slices = info["slices"]
    roi_size = info["roi_size"]
    importance_map = info["importance_map"]
    importance_map_resized = None

    for i in tqdm(range(0, len(slices), sw_batch_size), desc=f"Total patch {len(slices)}"):
        batch_slices = slices[i : i + sw_batch_size]

        # Extract patches for this batch only (lazy)
        patch_list = []
        for slice_idx in batch_slices:
            full_slice = (slice(None),) + slice_idx + (slice(None),)
            patch_list.append(padded_inputs[full_slice])
        patches = np.concatenate(patch_list, axis=0)

        # XLA padding, dimension-agnostic
        # GitHub: https://github.com/keras-team/keras/issues/21167
        bs_actual = patches.shape[0]
        bs_target = sw_batch_size * info["batch_size"]
        if bs_actual < bs_target:
            batch_pad_size = (
                (0, bs_target - bs_actual),
                *[(0, 0)] * (len(patches.shape) - 1),
            )
            patches = np.pad(patches, batch_pad_size, mode="constant", constant_values=0)

        pred = model.predict(patches, verbose=0)
        pred = pred[:bs_actual]

        # Resize importance map if necessary, dimension-agnostic (computed once)
        if importance_map_resized is None:
            if pred.shape[1:-1] != tuple(roi_size):
                raise ValueError(
                    f"Model output spatial shape {pred.shape[1:-1]} differs from "
                    f"roi_size {tuple(roi_size)}. Slice-based accumulation requires "
                    f"the model to preserve spatial dimensions."
                )
            importance_map_resized = importance_map

        yield pred, batch_slices, importance_map_resized


def merge_patches(
    patch_generator: Generator[tuple[np.ndarray, list, np.ndarray], None, None],
    info: dict,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Consumes the generator from ``predict_patches`` and reconstructs the
    full output volume by blending overlapping predictions.

    Output and count arrays are pre-allocated once (on the first yielded
    batch) and updated in-place, so no intermediate array of all
    predictions is ever created.

    Args:
        patch_generator (Generator): Generator yielding
            ``(pred_batch, batch_slices, importance_map_resized)`` tuples,
            as produced by ``predict_patches``.
        info (dict): Metadata dictionary returned by ``extract_patches``.
        num_classes (Optional[int]): Number of output classes.  If ``None``,
            inferred from the first prediction batch.

    Returns:
        np.ndarray: Reconstructed and blended output volume with shape
            ``(batch_size, *original_spatial_dims, num_classes)``.
    """
    batch_size = info["batch_size"]
    padded_image_size = info["padded_image_size"]
    original_image_size = info["original_image_size"]
    pad_size = info["pad_size"]

    output_image = None
    count_map = None

    for pred_batch, batch_slices, importance_map_resized in patch_generator:
        # Lazy initialisation, infer num_classes from first prediction
        if output_image is None:
            nc = num_classes or pred_batch.shape[-1]
            output_shape = [batch_size] + list(padded_image_size) + [nc]
            output_image = np.zeros(output_shape, dtype=np.float32)
            count_map = np.zeros([1] + list(padded_image_size) + [1], dtype=np.float32)

        for j, slice_idx in enumerate(batch_slices):
            output_slice = (slice(None),) + slice_idx + (slice(None),)
            b = batch_size
            output_image[output_slice] += pred_batch[j * b : (j + 1) * b] * importance_map_resized
            count_map[output_slice] += importance_map_resized

    if output_image is None:
        raise ValueError("patch_generator yielded no batches; cannot reconstruct output.")

    np.divide(output_image, count_map, out=output_image, where=count_map != 0)

    if any(p for pair in pad_size for p in pair):
        output_image = crop_output(output_image, pad_size, original_image_size)

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
        inputs (np.ndarray): Input tensor with shape
            ``(batch_size, *spatial_dims, channels)``.
        model: Object with a ``predict(x, verbose=...)`` method. Takes a patch of
            input and returns predictions.
        num_classes (Optional[int]): The number of output classes.
        roi_size (Sequence[int]): Spatial window size.
        sw_batch_size (int): Batch size for sliding window inference.
        overlap (Union[Sequence[float], float]): Overlap ratio between
            windows (default: 0.25).
        mode (str): Blending mode (``"constant"`` or ``"gaussian"``).
        sigma_scale (Union[Sequence[float], float]): Std dev coefficient
            for Gaussian blending.
        padding_mode (str): Padding mode.
        cval (float): Padding value.
        roi_weight_map (Optional[np.ndarray]): Pre-computed weight map.

    Returns:
        np.ndarray: Reconstructed output tensor.
    """
    if len(inputs.shape) < 3:
        raise ValueError(
            f"Input tensor must have shape (batch, *spatial, channels), "
            f"got rank {len(inputs.shape)}."
        )

    try:
        padded_inputs, info = extract_patches(
            inputs=inputs,
            roi_size=roi_size,
            overlap=overlap,
            mode=mode,
            sigma_scale=sigma_scale,
            padding_mode=padding_mode,
            cval=cval,
            roi_weight_map=roi_weight_map,
        )
    except Exception as e:
        raise RuntimeError(
            f"Sliding window inference failed during the patch extraction phase: {e}"
        ) from e

    try:
        pred_gen = predict_patches(
            padded_inputs=padded_inputs,
            info=info,
            model=model,
            sw_batch_size=sw_batch_size,
        )
    except Exception as e:
        raise RuntimeError(
            f"Sliding window inference failed during the prediction phase setup: {e}"
        ) from e

    try:
        return merge_patches(
            patch_generator=pred_gen,
            info=info,
            num_classes=num_classes,
        )
    except Exception as e:
        raise RuntimeError(
            f"Sliding window inference failed during the prediction/merging phase: {e}"
        ) from e

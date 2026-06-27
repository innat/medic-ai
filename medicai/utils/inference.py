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


_NUMPY_PADDING_MODE_ALIASES = {
    "replicate": "edge",
    "circular": "wrap",
}


class SlidingWindowInference:
    """
    Sliding Window Inference for large-volume or high-resolution inputs. This class
    performs patch-based inference by dividing the input tensor into overlapping regions
    of interest (ROIs), running a model on each patch, and merging the outputs using a
    specified blending strategy.

    It is commonly used in medical imaging and volumetric segmentation tasks
    where the full input cannot be processed at once due to memory constraints.

    The inference pipeline works as follows:

    1. Input is divided into overlapping sliding windows (patches).
    2. Each patch is processed independently by the model.
    3. Predictions are aggregated back into the full spatial volume.
    4. Overlapping regions are blended using either constant or Gaussian weighting.

    Args:
        model: Callable that takes a patch of input and returns predictions.
        num_classes (int): The number of output classes.
        roi_size (Sequence[int]): Spatial window size for inferences.
        sw_batch_size (int): Batch size for sliding window inference.
        overlap (Union[Sequence[float], float]): Overlap ratio between windows
            (default: ``0.25``). Can be a single float for isotropic overlap
            or a sequence of floats for anisotropic overlap.
        mode (str): Blending mode for overlapping windows. Options are:
            ``"constant"`` or ``"gaussian"`` (default: ``"constant"``).
        sigma_scale (Union[Sequence[float], float]): Standard deviation
            coefficient for Gaussian blending. Only used if ``mode`` is
            ``"gaussian"``. Can be a single float or a sequence of floats.
            (default: ``0.125``).
        padding_mode (str): Padding mode for inputs when ``roi_size`` is
            larger than the input size. Options are numpy padding modes
            (e.g., ``"constant"``, ``"reflect"``, ``"replicate"``) (default: ``"constant"``).
        cval (float): Padding value for ``"constant"`` padding mode (default: ``0.0``).
        roi_weight_map (Optional[np.ndarray]): Pre-computed weight map for
            each ROI. If ``None``, it will be computed based on the ``mode``.
            Should have the same spatial dimensions as ``roi_size``.
            (default: ``None``).

    Examples:
        .. code-block:: python

            import numpy as np
            from medicai.models import UNet
            from medicai.utils import SlidingWindowInference

            model = UNet(
                encoder_name='densenet121', input_shape=(96, 96, 96, 1), num_classes=3
            )
            swi = SlidingWindowInference(
                model,
                num_classes=3,
                roi_size=(96, 96, 96),
                sw_batch_size=2,
                overlap=0.25,
                mode="gaussian"
            )
            x = np.random.randn(1, 128, 128, 128, 1).astype(np.float32)
            output = swi(x)
            print(output.shape) # (1, 128, 128, 128, 3)

    Returns:
        np.ndarray: The output tensor with the same batch size and
            spatial dimensions as the input, and the number of channels
            equal to ``num_classes``.
    """

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

    def __call__(self, x):
        """
        Call method to perform sliding window inference.

        Args:
            x (np.ndarray): Input tensor with shape
                (batch_size, *spatial_dims, channels).

        Returns:
            np.ndarray: The output tensor with the same batch size and
                spatial dimensions as the input, and the number of channels
                equal to ``num_classes``.
        """
        return sliding_window_inference(
            x=x,
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
    """Prepare padded inputs and patch metadata for sliding-window inference.

    Args:
        inputs: Input tensor with shape ``(batch_size, *spatial_dims, channels)``.
        roi_size: Spatial window size for inference.
        overlap: Overlap ratio between windows.
        mode: Importance-map blending mode. Supported values are ``"constant"``
            and ``"gaussian"``.
        sigma_scale: Gaussian blending coefficient used when ``mode="gaussian"``.
        padding_mode: NumPy padding mode applied when an input dimension is
            smaller than the requested ROI size.
        cval: Constant fill value used with ``padding_mode="constant"``.
        roi_weight_map: Optional precomputed ROI importance map.

    Returns:
        A tuple ``(padded_inputs, info)`` where ``info`` contains the metadata
        required to run patch prediction and merge the overlapping outputs.
    """
    inputs = np.array(inputs) if not isinstance(inputs, np.ndarray) else inputs
    if inputs.ndim < 3:
        raise ValueError(
            "Input tensor must have shape (batch_size, *spatial_dims, channels), "
            f"got rank {inputs.ndim}."
        )

    num_spatial_dims = len(inputs.shape) - 2
    overlap = ensure_tuple_rep(overlap, num_spatial_dims)
    sigma_scale = ensure_tuple_rep(sigma_scale, num_spatial_dims)

    for value in overlap:
        if value < 0 or value >= 1:
            raise ValueError(f"Overlap must be >= 0 and < 1, got {overlap}.")

    batch_size, *original_image_size, _ = inputs.shape
    roi_size = fall_back_tuple(roi_size, original_image_size)
    padded_image_size = tuple(
        max(original_image_size[index], roi_size[index]) for index in range(num_spatial_dims)
    )

    pad_size = []
    for index in range(num_spatial_dims):
        diff = max(roi_size[index] - original_image_size[index], 0)
        half = diff // 2
        pad_size.append([half, diff - half])
    pad_size = [[0, 0]] + pad_size + [[0, 0]]

    normalized_padding_mode = _normalize_padding_mode(padding_mode)
    if any(value for pair in pad_size for value in pair):
        pad_kwargs = {"mode": normalized_padding_mode}
        if normalized_padding_mode == "constant":
            pad_kwargs["constant_values"] = cval
        inputs = np.pad(inputs, pad_size, **pad_kwargs)

    scan_interval = get_scan_interval(padded_image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(padded_image_size, roi_size, scan_interval)

    valid_patch_size = get_valid_patch_size(padded_image_size, roi_size)
    if roi_weight_map is not None:
        importance_map = _normalize_roi_weight_map(
            roi_weight_map=roi_weight_map,
            patch_size=valid_patch_size,
            dtype=inputs.dtype,
            num_spatial_dims=num_spatial_dims,
        )
    else:
        importance_map = compute_importance_map(
            valid_patch_size, mode=mode, sigma_scale=sigma_scale, dtype=inputs.dtype
        )
    if len(importance_map.shape) == num_spatial_dims:
        importance_map = np.expand_dims(np.expand_dims(importance_map, -1), 0)

    info = {
        "batch_size": batch_size,
        "num_spatial_dims": num_spatial_dims,
        "original_image_size": tuple(original_image_size),
        "padded_image_size": padded_image_size,
        "pad_size": pad_size,
        "roi_size": tuple(roi_size),
        "slices": slices,
        "importance_map": importance_map,
    }
    return inputs, info


def _normalize_padding_mode(padding_mode: str) -> str:
    """Translate documented padding aliases to NumPy-compatible modes."""
    normalized_mode = padding_mode.lower()
    return _NUMPY_PADDING_MODE_ALIASES.get(normalized_mode, normalized_mode)


def _normalize_roi_weight_map(
    roi_weight_map,
    patch_size: Sequence[int],
    dtype,
    num_spatial_dims: int,
) -> np.ndarray:
    """Normalize a caller-supplied ROI weight map to ``(1, *roi_size, 1)``."""
    importance_map = np.asarray(roi_weight_map, dtype=dtype)

    if importance_map.ndim == 0:
        return np.full((1, *patch_size, 1), importance_map, dtype=dtype)

    if importance_map.ndim == num_spatial_dims:
        if tuple(importance_map.shape) != tuple(patch_size):
            raise ValueError(
                "roi_weight_map must match the effective ROI spatial size when "
                "provided without batch/channel dimensions."
            )
        return importance_map[None, ..., None]

    if importance_map.ndim != num_spatial_dims + 2:
        raise ValueError(
            "roi_weight_map must be a scalar, a spatial-only tensor shaped like roi_size, "
            "or a tensor shaped (1, *roi_size, 1)."
        )

    expected_shape = (1, *patch_size, 1)
    if tuple(importance_map.shape) != expected_shape:
        raise ValueError(
            f"roi_weight_map must have shape {expected_shape}, got {tuple(importance_map.shape)}."
        )
    return importance_map


def _resize_importance_map(importance_map: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Resize an importance map to match the prediction spatial shape."""
    if tuple(importance_map.shape[1:-1]) == tuple(target_shape):
        return importance_map

    scale_factors = [1.0]
    scale_factors.extend(
        target_shape[index] / importance_map.shape[index + 1] for index in range(len(target_shape))
    )
    scale_factors.append(1.0)
    return zoom(importance_map, tuple(scale_factors), order=0)


def predict_patches(
    padded_inputs: np.ndarray,
    info: dict,
    model,
    sw_batch_size: int,
) -> Generator[tuple[np.ndarray, list[tuple[slice, ...]], np.ndarray], None, None]:
    """Run inference over patch batches and yield predictions lazily.

    Args:
        padded_inputs: Padded input tensor returned by :func:`extract_patches`.
        info: Metadata dictionary returned by :func:`extract_patches`.
        model: Object exposing ``predict(x, verbose=0)``.
        sw_batch_size: Number of spatial windows evaluated per call.

    Yields:
        Tuples of ``(pred_batch, batch_slices, importance_map_resized)`` for
        each patch batch.
    """
    if sw_batch_size <= 0:
        raise ValueError(f"sw_batch_size must be a positive integer, got {sw_batch_size}.")

    slices = info["slices"]
    batch_size = info["batch_size"]
    importance_map = info["importance_map"]
    importance_map_resized = None
    has_multiple_patch_batches = len(slices) > sw_batch_size
    roi_size = tuple(info["roi_size"])

    progress_desc = f"Window positions {len(slices)} | input batch {batch_size}"
    for start in tqdm(range(0, len(slices), sw_batch_size), desc=progress_desc):
        batch_slices = slices[start : start + sw_batch_size]
        patch_list = []
        for slice_idx in batch_slices:
            full_slice = (slice(None),) + slice_idx + (slice(None),)
            patch_list.append(padded_inputs[full_slice])
        patches = np.concatenate(patch_list, axis=0)

        # Pad only when multiple predict calls are needed so XLA-compiling
        # backends can keep a static batch size across calls.
        actual_patch_batch = patches.shape[0]
        target_patch_batch = sw_batch_size * batch_size
        if has_multiple_patch_batches and actual_patch_batch < target_patch_batch:
            batch_pad_size = ((0, target_patch_batch - actual_patch_batch),) + ((0, 0),) * (
                patches.ndim - 1
            )
            patches = np.pad(patches, batch_pad_size, mode="constant", constant_values=0)

        pred = model.predict(patches, verbose=0)
        pred = pred[:actual_patch_batch]

        if tuple(pred.shape[1:-1]) != roi_size:
            raise ValueError(
                "sliding_window_inference requires the model output spatial shape "
                f"to match roi_size. Got output shape {tuple(pred.shape[1:-1])} "
                f"and roi_size {roi_size}."
            )

        if importance_map_resized is None:
            importance_map_resized = _resize_importance_map(importance_map, pred.shape[1:-1])

        yield pred, batch_slices, importance_map_resized


def merge_patches(
    patch_generator: Generator[tuple[np.ndarray, list[tuple[slice, ...]], np.ndarray], None, None],
    info: dict,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Merge overlapping patch predictions into a full-resolution output.

    Args:
        patch_generator: Patch prediction generator produced by
            :func:`predict_patches`.
        info: Metadata dictionary returned by :func:`extract_patches`.
        num_classes: Number of output channels. If ``None``, inferred from the
            first yielded prediction batch.

    Returns:
        Reconstructed output tensor with shape
        ``(batch_size, *original_spatial_dims, num_classes)``.
    """
    batch_size = info["batch_size"]
    padded_image_size = info["padded_image_size"]
    original_image_size = info["original_image_size"]
    pad_size = info["pad_size"]

    output_image = None
    count_map = None

    for pred_batch, batch_slices, importance_map_resized in patch_generator:
        if output_image is None:
            resolved_num_classes = num_classes or pred_batch.shape[-1]
            output_shape = [batch_size] + list(padded_image_size) + [resolved_num_classes]
            output_image = np.zeros(output_shape, dtype=np.float32)
            count_map = np.zeros([1] + list(padded_image_size) + [1], dtype=np.float32)

        for index, slice_idx in enumerate(batch_slices):
            output_slice = (slice(None),) + slice_idx + (slice(None),)
            start = index * batch_size
            stop = start + batch_size
            output_image[output_slice] += pred_batch[start:stop] * importance_map_resized
            count_map[output_slice] += importance_map_resized

    if output_image is None or count_map is None:
        raise ValueError("Patch generator yielded no predictions.")

    np.divide(output_image, count_map, out=output_image, where=count_map != 0)

    if any(value for pair in pad_size for value in pair):
        output_image = crop_output(output_image, pad_size, original_image_size)

    return output_image


def sliding_window_inference(
    x,
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
    """Run sliding-window inference over large 2D or 3D inputs.

    Args:
        x: Input tensor with shape ``(batch_size, *spatial_dims, channels)``.
        model: Object exposing ``predict(x, verbose=0)``.
        num_classes: Number of output channels. If ``None``, inferred from the
            first prediction batch.
        roi_size: Spatial window size for inference.
        sw_batch_size: Number of spatial windows evaluated per predict call.
        overlap: Overlap ratio between windows.
        mode: Importance-map blending mode. Supported values are ``"constant"``
            and ``"gaussian"``.
        sigma_scale: Gaussian blending coefficient used when ``mode="gaussian"``.
        padding_mode: NumPy padding mode applied when an input dimension is
            smaller than the requested ROI size.
        cval: Constant fill value used with ``padding_mode="constant"``.
        roi_weight_map: Optional precomputed ROI importance map.

    Returns:
        Reconstructed output tensor with shape
        ``(batch_size, *original_spatial_dims, num_classes)``.
    """
    padded_inputs, info = extract_patches(
        inputs=x,
        roi_size=roi_size,
        overlap=overlap,
        mode=mode,
        sigma_scale=sigma_scale,
        padding_mode=padding_mode,
        cval=cval,
        roi_weight_map=roi_weight_map,
    )
    patch_generator = predict_patches(
        padded_inputs=padded_inputs,
        info=info,
        model=model,
        sw_batch_size=sw_batch_size,
    )
    return merge_patches(
        patch_generator=patch_generator,
        info=info,
        num_classes=num_classes,
    )

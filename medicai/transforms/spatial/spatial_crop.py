from typing import Sequence

import tensorflow as tf

from ..base import KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import ensure_spatial_tuple, get_spatial_rank, get_spatial_shape


class SpatialCrop(KeyedTransform):
    """Deterministically crop a spatial region from selected tensors.

    This transform crops channel-last 2D tensors ``(H, W, C)`` and 3D tensors
    ``(D, H, W, C)`` using either a crop ``roi_start`` or a crop
    ``roi_center`` together with ``roi_size``.

    ``roi_size`` is interpreted per spatial dimension. Non-positive values are
    treated as "use the full available size" for that dimension after bounds
    checking. Crop bounds are clipped so the output always stays within the
    selected tensor's spatial extent.

    Args:
        keys: Keys of the tensors to crop.
        roi_size: Desired spatial crop size.
        roi_start: Spatial start coordinates of the crop. Mutually exclusive
            with ``roi_center``.
        roi_center: Spatial center coordinates of the crop. Mutually exclusive
            with ``roi_start``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Crop a centered 3D patch from an image-label pair:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import SpatialCrop

            transform = SpatialCrop(
                keys=["image", "label"],
                roi_size=(32, 64, 64),
                roi_center=(24, 48, 48),
            )

            image = tf.random.normal((48, 96, 96, 1))
            label = tf.random.uniform((48, 96, 96, 1), maxval=2, dtype=tf.int32)

            result = transform({"image": image, "label": label})

    Returns:
        ``TensorBundle``: The input bundle with cropped tensors and a
        non-invertible transform trace entry appended when at least one
        selected key is present.

    Raises:
        ValueError: If both ``roi_start`` and ``roi_center`` are provided, or
            if a spatial parameter does not match the selected tensor rank.
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
    """

    def __init__(
        self,
        keys: Sequence[str],
        roi_size: int | Sequence[int],
        roi_start: Sequence[int] | None = None,
        roi_center: Sequence[int] | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        if roi_start is not None and roi_center is not None:
            raise ValueError("Only one of `roi_start` or `roi_center` may be provided.")

        self.roi_size = roi_size
        self.roi_start = tuple(roi_start) if roi_start is not None else None
        self.roi_center = tuple(roi_center) if roi_center is not None else None

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        crop_info = {"start": None, "size": None}

        def apply_crop(tensor: tf.Tensor, _: str) -> tf.Tensor:
            starts, roi_size = self.compute_crop_bounds(tensor)
            crop_info["start"] = starts
            crop_info["size"] = roi_size
            return self.crop_tensor(tensor, starts, roi_size)

        present_keys = self.apply_to_present_keys(bundle, apply_crop)

        if crop_info["start"] is not None:
            bundle.push_transform(
                self.build_trace_entry(
                    params={
                        "keys": list(present_keys),
                        "roi_start": crop_info["start"],
                        "roi_size": crop_info["size"],
                    },
                    applied=True,
                    random=False,
                    invertible=False,
                )
            )
        return bundle

    def compute_crop_bounds(self, tensor: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute bounded crop start coordinates and crop size.

        Args:
            tensor: Channel-last 2D or 3D sample tensor to crop.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: A pair ``(starts, roi_size)`` where
            both tensors have one value per spatial dimension.
        """
        spatial_rank = get_spatial_rank(tensor)
        spatial_shape = get_spatial_shape(tensor)
        roi_size = tf.convert_to_tensor(
            ensure_spatial_tuple(self.roi_size, spatial_rank, "roi_size"),
            dtype=tf.int32,
        )
        roi_size = tf.where(roi_size > 0, roi_size, spatial_shape)
        roi_size = tf.minimum(roi_size, spatial_shape)

        if self.roi_start is not None:
            starts = tf.convert_to_tensor(
                ensure_spatial_tuple(self.roi_start, spatial_rank, "roi_start"),
                dtype=tf.int32,
            )
        elif self.roi_center is not None:
            center = tf.convert_to_tensor(
                ensure_spatial_tuple(self.roi_center, spatial_rank, "roi_center"),
                dtype=tf.int32,
            )
            starts = tf.maximum(center - roi_size // 2, 0)
        else:
            starts = tf.zeros((spatial_rank,), dtype=tf.int32)

        ends = tf.minimum(starts + roi_size, spatial_shape)
        starts = tf.maximum(ends - roi_size, 0)
        return starts, roi_size

    def crop_tensor(self, tensor: tf.Tensor, starts: tf.Tensor, roi_size: tf.Tensor) -> tf.Tensor:
        """Crop one tensor using TensorFlow slicing.

        Args:
            tensor: Channel-last 2D or 3D sample tensor.
            starts: Start indices for each spatial dimension.
            roi_size: Crop size for each spatial dimension.

        Returns:
            ``tf.Tensor``: The cropped tensor with the original channel
            dimension preserved.
        """
        begin = tf.concat([starts, [0]], axis=0)
        size = tf.concat([roi_size, [tf.shape(tensor)[-1]]], axis=0)
        return tf.slice(tensor, begin=begin, size=size)

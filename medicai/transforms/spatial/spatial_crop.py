from typing import Sequence

import tensorflow as tf

from ..base import KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import ensure_spatial_tuple, get_spatial_rank, get_spatial_shape


class SpatialCrop(KeyedTransform):
    """Deterministically crop a spatial region from selected tensors.

    This transform crops channel-last 2D tensors ``(H, W, C)`` and 3D tensors
    ``(D, H, W, C)`` using either a crop ``crop_start`` or a crop
    ``crop_center`` together with ``crop_size``.

    ``crop_size`` is interpreted per spatial dimension. Non-positive values are
    treated as "use the full available size" for that dimension after bounds
    checking. Crop bounds are clipped so the output always stays within the
    selected tensor's spatial extent.

    Args:
        keys: Keys of the tensors to crop.
        crop_size: Desired spatial crop size.
        crop_start: Spatial start coordinates of the crop. Mutually exclusive
            with ``crop_center``.
        crop_center: Spatial center coordinates of the crop. Mutually exclusive
            with ``crop_start``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Crop a centered 3D patch from an image-label pair:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import SpatialCrop

            transform = SpatialCrop(
                keys=["image", "label"],
                crop_size=(32, 64, 64),
                crop_center=(24, 48, 48),
            )

            image = tf.random.normal((48, 96, 96, 1))
            label = tf.random.uniform(
                (48, 96, 96, 1), maxval=2, dtype=tf.int32
            )
            result = transform({"image": image, "label": label})

            print(result["image"].shape)
            print(result["label"].shape)

        Crop a centered 2D patch from an image-label pair:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import SpatialCrop

            transform = SpatialCrop(
                keys=["image", "label"],
                crop_size=(64, 64),
                crop_center=(48, 48),
            )

            image = tf.random.normal((96, 96, 1))
            label = tf.random.uniform(
                (96, 96, 1), maxval=2, dtype=tf.int32
            )
            result = transform({"image": image, "label": label})

            print(result["image"].shape)
            print(result["label"].shape)

    Returns:
        ``TensorBundle``: The input bundle with cropped tensors and a
        non-invertible transform trace entry appended when at least one
        selected key is present.

    Raises:
        ValueError: If both ``crop_start`` and ``crop_center`` are provided, or
            if a spatial parameter does not match the selected tensor rank.
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
    """

    def __init__(
        self,
        keys: Sequence[str],
        crop_size: int | Sequence[int],
        crop_start: Sequence[int] | None = None,
        crop_center: Sequence[int] | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        if crop_start is not None and crop_center is not None:
            raise ValueError("Only one of `crop_start` or `crop_center` may be provided.")

        self.crop_size = crop_size
        self.crop_start = tuple(crop_start) if crop_start is not None else None
        self.crop_center = tuple(crop_center) if crop_center is not None else None

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        crop_info = {"start": None, "size": None}

        def apply_crop(tensor: tf.Tensor, _: str) -> tf.Tensor:
            starts, crop_size = self.compute_crop_bounds(tensor)
            crop_info["start"] = starts
            crop_info["size"] = crop_size
            return self.crop_tensor(tensor, starts, crop_size)

        present_keys = self.apply_to_present_keys(bundle, apply_crop)

        if crop_info["start"] is not None:
            bundle.push_transform(
                self.build_trace_entry(
                    params={
                        "keys": list(present_keys),
                        "crop_start": crop_info["start"],
                        "crop_size": crop_info["size"],
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
            tuple[tf.Tensor, tf.Tensor]: A pair ``(starts, crop_size)`` where
            both tensors have one value per spatial dimension.
        """
        spatial_rank = get_spatial_rank(tensor)
        spatial_shape = get_spatial_shape(tensor)
        crop_size = tf.convert_to_tensor(
            ensure_spatial_tuple(self.crop_size, spatial_rank, "crop_size"),
            dtype=tf.int32,
        )
        crop_size = tf.where(crop_size > 0, crop_size, spatial_shape)
        crop_size = tf.minimum(crop_size, spatial_shape)

        if self.crop_start is not None:
            starts = tf.convert_to_tensor(
                ensure_spatial_tuple(self.crop_start, spatial_rank, "crop_start"),
                dtype=tf.int32,
            )
        elif self.crop_center is not None:
            center = tf.convert_to_tensor(
                ensure_spatial_tuple(self.crop_center, spatial_rank, "crop_center"),
                dtype=tf.int32,
            )
            starts = tf.maximum(center - crop_size // 2, 0)
        else:
            starts = tf.zeros((spatial_rank,), dtype=tf.int32)

        ends = tf.minimum(starts + crop_size, spatial_shape)
        starts = tf.maximum(ends - crop_size, 0)
        return starts, crop_size

    def crop_tensor(self, tensor: tf.Tensor, starts: tf.Tensor, crop_size: tf.Tensor) -> tf.Tensor:
        """Crop one tensor using TensorFlow slicing.

        Args:
            tensor: Channel-last 2D or 3D sample tensor.
            starts: Start indices for each spatial dimension.
            crop_size: Crop size for each spatial dimension.

        Returns:
            ``tf.Tensor``: The cropped tensor with the original channel
            dimension preserved.
        """
        begin = tf.concat([starts, [0]], axis=0)
        size = tf.concat([crop_size, [tf.shape(tensor)[-1]]], axis=0)
        return tf.slice(tensor, begin=begin, size=size)

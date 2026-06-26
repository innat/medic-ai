from typing import Sequence

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform, _pop_last_transform_trace
from ..tensor_bundle import TensorBundle
from ..utils import ensure_spatial_tuple, get_spatial_rank, get_spatial_shape


class SpatialCrop(KeyedTransform, InvertibleTransform):
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

    ``inverse()`` is a placement inverse: it pads the cropped tensor back into
    its original spatial canvas using the recorded crop coordinates. This is
    especially useful for mapping model predictions back to the pre-crop
    layout. It does not reconstruct values that were discarded outside the crop
    region; those locations are filled with zeros.

    Returns:
        ``TensorBundle``: The input bundle with cropped tensors and an
        invertible transform trace entry appended when at least one
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
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        if crop_start is not None and crop_center is not None:
            raise ValueError("Only one of `crop_start` or `crop_center` may be provided.")

        self.crop_size = crop_size
        self.crop_start = tuple(crop_start) if crop_start is not None else None
        self.crop_center = tuple(crop_center) if crop_center is not None else None

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        crop_starts = {}
        crop_sizes = {}
        original_shapes = {}

        def apply_crop(tensor: tf.Tensor, key: str) -> tf.Tensor:
            starts, crop_size = self.compute_crop_bounds(tensor)
            crop_starts[key] = starts
            crop_sizes[key] = crop_size
            original_shapes[key] = get_spatial_shape(tensor)
            return self.crop_tensor(tensor, starts, crop_size)

        present_keys = self.apply_to_present_keys(bundle, apply_crop)

        if crop_starts:
            self.record_transform(
                bundle,
                {
                    "keys": list(present_keys),
                    "crop_start": crop_starts,
                    "crop_size": crop_sizes,
                    "original_shapes": original_shapes,
                },
            )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = self._get_last_spatial_crop_trace(bundle)
        if trace is None:
            return bundle

        crop_starts = trace["params"].get("crop_start", {})
        original_shapes = trace["params"].get("original_shapes", {})

        def apply_inverse_crop(tensor: tf.Tensor, key: str) -> tf.Tensor:
            crop_start = crop_starts.get(key)
            original_shape = original_shapes.get(key)
            if crop_start is None or original_shape is None:
                return tensor
            return self.pad_to_original_shape(tensor, crop_start, original_shape)

        self.apply_to_present_keys(bundle, apply_inverse_crop, keys=trace["params"].get("keys", []))
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

    def pad_to_original_shape(
        self,
        tensor: tf.Tensor,
        crop_start: tf.Tensor,
        original_shape: tf.Tensor,
    ) -> tf.Tensor:
        """Pad a cropped tensor back into its original spatial canvas."""
        crop_start = tf.cast(crop_start, tf.int32)
        original_shape = tf.cast(original_shape, tf.int32)
        current_shape = get_spatial_shape(tensor)
        pad_before = crop_start
        pad_after = original_shape - crop_start - current_shape
        paddings = tf.concat(
            [
                tf.stack([pad_before, pad_after], axis=1),
                tf.constant([[0, 0]], dtype=tf.int32),
            ],
            axis=0,
        )
        return tf.pad(tensor, paddings)

    def _get_last_spatial_crop_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)

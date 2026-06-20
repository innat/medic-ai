from typing import Sequence

import tensorflow as tf

from ..base import RandomTransform
from ..spatial.spatial_crop import SpatialCrop
from ..tensor_bundle import TensorBundle
from ..utils import get_spatial_rank, get_spatial_shape


class RandomSpatialCrop(RandomTransform):
    """Randomly crop a spatial region of interest from selected tensors.

    ``RandomSpatialCrop`` samples a crop center and, optionally, a crop size
    before extracting a spatial patch with the deterministic
    :class:`~medicai.transforms.SpatialCrop` kernel.

    This transform supports:

    - 2D tensors shaped ``(H, W, C)``
    - 3D tensors shaped ``(D, H, W, C)``

    When ``invalid_label`` is provided, the crop center can be sampled using
    the ``"label"`` tensor to favor valid regions.

    Args:
        keys: Keys of the tensors to crop.
        roi_size: Minimum or fixed crop size.
        max_roi_size: Maximum crop size when ``random_size=True``.
        random_center: If ``True``, sample crop centers randomly.
        random_size: If ``True``, sample crop sizes between ``roi_size`` and
            ``max_roi_size``.
        invalid_label: Label value treated as invalid when enforcing valid
            crop regions.
        min_valid_ratio: Minimum fraction of valid labels required in a crop.
        max_attempts: Maximum attempts when searching for a crop that satisfies
            ``min_valid_ratio``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Randomly crop a 2D image using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomSpatialCrop

            transform = RandomSpatialCrop(keys=["image"], roi_size=(32, 32))
            image = tf.random.normal((64, 64, 1))
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Randomly crop a 3D image stored in a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomSpatialCrop, TensorBundle

            transform = RandomSpatialCrop(keys=["image"], roi_size=(16, 32, 32))
            image = tf.random.normal((32, 64, 64, 1))
            bundle = TensorBundle({"image": image})
            result = transform(bundle)
            output = result["image"]
            print(output.shape)
    """

    def __init__(
        self,
        keys: Sequence[str],
        roi_size,
        max_roi_size=None,
        random_center: bool = True,
        random_size: bool = False,
        invalid_label=None,
        min_valid_ratio: float = 0.0,
        max_attempts: int = 1,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=1.0)
        self.keys = tuple(keys)
        self.roi_size = roi_size
        self.max_roi_size = max_roi_size
        self.random_center = random_center
        self.random_size = random_size
        self.invalid_label = invalid_label
        self.min_valid_ratio = min_valid_ratio
        self.max_attempts = max_attempts
        self.allow_missing_keys = allow_missing_keys

        if not (0.0 <= min_valid_ratio <= 1.0):
            raise ValueError(f"min_valid_ratio must be in range [0.0, 1.0], got {min_valid_ratio}")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be a positive integer, got {max_attempts}")
        if min_valid_ratio > 0.0 and invalid_label is None:
            raise ValueError(
                "If min_valid_ratio > 0, you must provide an invalid_label (e.g., 0) "
                "to calculate the ratio of valid pixels."
            )

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        sample_key = self.keys[0]
        if sample_key not in bundle.data:
            if self.allow_missing_keys:
                return bundle
            raise KeyError(f"Key '{sample_key}' not found in input data.")

        sample_tensor = bundle.data[sample_key]
        spatial_rank = get_spatial_rank(sample_tensor)
        spatial_shape = get_spatial_shape(sample_tensor)
        roi_size = self._get_roi_size(spatial_shape, spatial_rank)

        if self.invalid_label is None:
            center = self._get_random_center(spatial_shape, roi_size, spatial_rank)
        else:
            if "label" not in bundle.data:
                raise KeyError("`label` key is required when `invalid_label` is specified.")
            center = self._get_label_aware_center(
                spatial_shape, roi_size, bundle["label"], spatial_rank
            )

        crop = SpatialCrop(
            keys=self.keys,
            roi_size=self.roi_size,
            allow_missing_keys=self.allow_missing_keys,
        )

        starts = tf.maximum(center - roi_size // 2, 0)
        ends = tf.minimum(starts + roi_size, spatial_shape)
        starts = tf.maximum(ends - roi_size, 0)
        present_keys = crop.apply_to_present_keys(
            bundle,
            lambda tensor, _: crop.crop_tensor(tensor, starts, roi_size),
        )
        bundle.push_transform(
            self.build_trace_entry(
                params={
                    "keys": list(present_keys),
                    "roi_start": starts,
                    "roi_size": roi_size,
                    "random_center": self.random_center,
                    "random_size": self.random_size,
                },
                applied=True,
                random=True,
                kernel="SpatialCrop",
            )
        )
        return bundle

    def _get_roi_size(self, spatial_shape: tf.Tensor, spatial_rank: int) -> tf.Tensor:
        if isinstance(self.roi_size, int):
            roi_size = tf.fill([spatial_rank], tf.cast(self.roi_size, tf.int32))
        else:
            roi_size = tf.convert_to_tensor(self.roi_size, dtype=tf.int32)

        if self.random_size:
            max_roi_size = (
                tf.fill([spatial_rank], tf.cast(self.max_roi_size, tf.int32))
                if isinstance(self.max_roi_size, int)
                else (
                    tf.convert_to_tensor(self.max_roi_size, dtype=tf.int32)
                    if self.max_roi_size is not None
                    else spatial_shape
                )
            )
            max_roi_size = tf.where(max_roi_size <= 0, spatial_shape, max_roi_size)

            def sample_dim(min_s, max_s, img_s):
                min_s = tf.where(min_s <= 0, img_s, min_s)
                max_s = tf.where(max_s <= 0, img_s, max_s)
                max_s = tf.minimum(max_s, img_s)
                min_s = tf.minimum(min_s, max_s)
                return tf.random.uniform([], minval=min_s, maxval=max_s + 1, dtype=tf.int32)

            roi_size = tf.stack(
                [
                    sample_dim(roi_size[i], max_roi_size[i], spatial_shape[i])
                    for i in range(roi_size.shape[0])
                ]
            )
        else:
            roi_size = tf.where(roi_size > 0, roi_size, spatial_shape)
            roi_size = tf.minimum(roi_size, spatial_shape)
        return roi_size

    def _get_random_center(
        self, spatial_shape: tf.Tensor, roi_size: tf.Tensor, spatial_rank: int
    ) -> tf.Tensor:
        if not self.random_center:
            return spatial_shape // 2

        max_start = tf.maximum(spatial_shape - roi_size, 0)
        random_start = tf.stack(
            [
                tf.random.uniform([], maxval=max_start[i] + 1, dtype=tf.int32)
                for i in range(spatial_rank)
            ]
        )
        return random_start + roi_size // 2

    def _get_label_aware_center(
        self, spatial_shape: tf.Tensor, roi_size: tf.Tensor, label: tf.Tensor, spatial_rank: int
    ) -> tf.Tensor:
        label = tf.squeeze(label)
        valid_mask = label != self.invalid_label
        valid_coords = tf.where(valid_mask)

        def fallback():
            return self._get_random_center(spatial_shape, roi_size, spatial_rank)

        def sample_valid_center():
            idx = tf.random.uniform([], 0, tf.shape(valid_coords)[0], dtype=tf.int32)
            return tf.cast(valid_coords[idx][:spatial_rank], tf.int32)

        center = tf.cond(tf.shape(valid_coords)[0] > 0, sample_valid_center, fallback)

        if self.min_valid_ratio > 0:
            center = self._enforce_min_valid_ratio(
                center, spatial_shape, roi_size, label, spatial_rank
            )

        return center

    def _enforce_min_valid_ratio(
        self,
        center: tf.Tensor,
        spatial_shape: tf.Tensor,
        roi_size: tf.Tensor,
        label: tf.Tensor,
        spatial_rank: int,
    ) -> tf.Tensor:
        def body(i, current_center):
            starts = tf.maximum(current_center - roi_size // 2, 0)
            ends = tf.minimum(starts + roi_size, spatial_shape)
            starts = tf.maximum(ends - roi_size, 0)

            if spatial_rank == 2:
                h0, w0 = tf.unstack(starts)
                h1, w1 = tf.unstack(ends)
                crop = label[h0:h1, w0:w1]
            else:
                d0, h0, w0 = tf.unstack(starts)
                d1, h1, w1 = tf.unstack(ends)
                crop = label[d0:d1, h0:h1, w0:w1]

            valid_ratio = tf.reduce_mean(tf.cast(crop != self.invalid_label, tf.float32))
            new_center = tf.cond(
                valid_ratio >= self.min_valid_ratio,
                lambda: current_center,
                lambda: self._get_random_center(spatial_shape, roi_size, spatial_rank),
            )
            return i + 1, new_center

        def cond(i, _):
            return i < self.max_attempts

        _, center = tf.while_loop(cond, body, [0, center], parallel_iterations=1)
        return center

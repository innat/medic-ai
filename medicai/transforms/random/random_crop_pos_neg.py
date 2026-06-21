from __future__ import annotations

from typing import Sequence

import tensorflow as tf

from ..base import RandomTransform
from ..spatial.spatial_crop import SpatialCrop
from ..tensor_bundle import TensorBundle
from ..utils import get_spatial_rank, get_spatial_shape


class RandomCropByPosNegLabel(RandomTransform):
    """Randomly crop one image-label patch using positive and negative label sampling.

    A crop center is sampled from either positive-label voxels or negative
    voxels according to the ``pos:neg`` ratio, then the same patch is cropped
    from both image and label tensors.

    This transform supports:

    - 2D tensors shaped ``(H, W, C)``
    - 3D tensors shaped ``(D, H, W, C)``

    Args:
        keys: Two keys containing the image tensor and label tensor.
        spatial_size: Output crop size as ``(H, W)`` for 2D inputs or
            ``(D, H, W)`` for 3D inputs.
        pos: Relative weight for positive-center sampling.
        neg: Relative weight for negative-center sampling.
        num_samples: Number of samples to return. Currently only ``1`` is
            supported.
        image_reference_key: Optional key for an intensity reference tensor
            used to constrain negative sampling.
        image_threshold: Threshold applied to ``image_reference_key`` during
            negative sampling.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Randomly crop a 2D image-label pair using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomCropByPosNegLabel

            transform = RandomCropByPosNegLabel(
                keys=["image", "label"],
                spatial_size=(32, 32),
                pos=1,
                neg=1,
            )

            image = tf.random.normal((64, 64, 1))
            label = tf.cast(image > 0, tf.int32)
            result = transform({"image": image, "label": label})
            output = result["image"]
            print(output.shape)

        Randomly crop a 3D image-label pair stored in a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomCropByPosNegLabel, TensorBundle

            transform = RandomCropByPosNegLabel(
                keys=["image", "label"],
                spatial_size=(16, 32, 32),
                pos=1,
                neg=1,
            )

            image = tf.random.normal((32, 64, 64, 1))
            label = tf.cast(image > 0, tf.int32)
            bundle = TensorBundle({"image": image, "label": label})
            result = transform(bundle)
            output = result["image"]
            print(output.shape)
    """

    def __init__(
        self,
        keys: Sequence[str],
        spatial_size: Sequence[int],
        pos: int,
        neg: int,
        num_samples: int = 1,
        image_reference_key: str = None,
        image_threshold: float = 0.0,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=1.0)
        if pos < 0 or neg < 0:
            raise ValueError("pos and neg must be non-negative.")
        if pos == 0 and neg == 0:
            raise ValueError("pos and neg cannot both be zero.")
        if len(keys) != 2:
            class_name = type(self).__name__
            raise ValueError(
                f"{class_name} transformation requires a pair of image and label as keys. "
            )
        if num_samples != 1:
            class_name = self.__class__.__name__
            raise ValueError(f"{class_name} transformation currently supports only num_samples=1.")

        self.keys = tuple(keys)
        self.spatial_size = spatial_size
        self.pos = pos
        self.neg = neg
        self.num_samples = num_samples
        self.pos_ratio = pos / (pos + neg)
        self.image_reference_key = image_reference_key
        self.image_threshold = image_threshold
        self.allow_missing_keys = allow_missing_keys

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        image_key, label_key = self.keys
        if image_key not in bundle.data or label_key not in bundle.data:
            if self.allow_missing_keys:
                return bundle
            missing = image_key if image_key not in bundle.data else label_key
            raise KeyError(f"Key '{missing}' not found in input data.")

        image = bundle.data[image_key]
        label = bundle.data[label_key]
        spatial_rank = get_spatial_rank(image)
        if spatial_rank not in (2, 3):
            raise ValueError(
                f"{type(self).__name__} currently supports only 2D or 3D inputs; got spatial rank "
                f"{spatial_rank} for shape {image.shape}."
            )

        image_reference = None
        if self.image_reference_key is not None:
            if self.image_reference_key not in bundle.data:
                raise KeyError(f"Key '{self.image_reference_key}' not found in input data.")
            image_reference = bundle.data[self.image_reference_key]
        center = self.sample_center(image, label, image_reference, spatial_rank)
        roi_size = tf.convert_to_tensor(self.spatial_size, dtype=tf.int32)
        if roi_size.shape.rank != 1 or roi_size.shape[0] != spatial_rank:
            raise ValueError(
                f"`spatial_size` must contain exactly {spatial_rank} values for input shape "
                f"{image.shape}; received {self.spatial_size}."
            )
        spatial_shape = get_spatial_shape(image)
        starts = tf.maximum(center - roi_size // 2, 0)
        ends = tf.minimum(starts + roi_size, spatial_shape)
        starts = tf.maximum(ends - roi_size, 0)

        crop = SpatialCrop(keys=self.keys, roi_size=self.spatial_size, allow_missing_keys=False)
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
                    "pos": self.pos,
                    "neg": self.neg,
                    "image_reference_key": self.image_reference_key,
                },
                applied=True,
                random=True,
                invertible=False,
                kernel="SpatialCrop",
            )
        )
        return bundle

    def sample_center(
        self,
        image: tf.Tensor,
        label: tf.Tensor,
        image_reference: tf.Tensor | None,
        spatial_rank: int,
    ) -> tf.Tensor:
        """Sample one crop center using positive/negative label sampling."""
        positive = tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < self.pos_ratio
        return tf.cond(
            positive,
            lambda: self._sample_positive_center(label, spatial_rank),
            lambda: self._sample_negative_center(image, label, image_reference, spatial_rank),
        )

    def _sample_positive_center(self, label: tf.Tensor, spatial_rank: int) -> tf.Tensor:
        coords = tf.where(label > 0)
        return self._sample_from_coords(
            coords,
            fallback_shape=get_spatial_shape(label),
            spatial_rank=spatial_rank,
        )

    def _sample_negative_center(
        self,
        image: tf.Tensor,
        label: tf.Tensor,
        image_reference: tf.Tensor | None,
        spatial_rank: int,
    ) -> tf.Tensor:
        if image_reference is not None and self.image_threshold is not None:
            max_intensity_ref = tf.reduce_max(image_reference, axis=-1, keepdims=True)
            label_is_zero = tf.reduce_any(label == 0, axis=-1, keepdims=True)
            valid_mask = label_is_zero & (max_intensity_ref > self.image_threshold)
            coords = tf.where(valid_mask)
        else:
            coords = tf.where(tf.reduce_any(label == 0, axis=-1))
        return self._sample_from_coords(
            coords,
            fallback_shape=get_spatial_shape(image),
            spatial_rank=spatial_rank,
        )

    def _sample_from_coords(
        self,
        coords: tf.Tensor,
        fallback_shape: tf.Tensor,
        spatial_rank: int,
    ) -> tf.Tensor:
        """Sample one spatial coordinate, falling back to any valid voxel if empty."""

        def fallback_coords():
            num_cols = tf.shape(coords)[1]
            random_coord = tf.stack(
                [
                    tf.random.uniform(
                        shape=(),
                        minval=0,
                        maxval=fallback_shape[axis],
                        dtype=tf.int32,
                    )
                    for axis in range(spatial_rank)
                ]
            )
            padding = tf.zeros([num_cols - spatial_rank], dtype=tf.int32)
            full_coord = tf.concat([random_coord, padding], axis=0)
            return tf.expand_dims(tf.cast(full_coord, coords.dtype), axis=0)

        coords = tf.cond(tf.shape(coords)[0] > 0, lambda: coords, fallback_coords)
        idx = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(coords)[0], dtype=tf.int32)
        return tf.cast(coords[idx][:spatial_rank], tf.int32)

from __future__ import annotations

from typing import Sequence

import keras
import tensorflow as tf

from ..base import RandomTransform, _normalize_keys, _pop_last_transform_trace
from ..spatial.spatial_crop import SpatialCrop
from ..tensor_bundle import TensorBundle
from ..utils import (
    ensure_batch_axis_for_layout,
    get_input_layout_info,
    get_spatial_shape_for_layout,
    resolve_input_layout,
    restore_from_batch_axis,
    validate_tensor_matches_layout,
)


class RandomCropByPosNegLabel(RandomTransform):
    """Randomly crop one image-label patch using positive and negative label sampling.

    A crop center is sampled from either positive-label voxels or negative
    voxels according to the ``pos:neg`` ratio, then the same patch is cropped
    from both image and label tensors.

    Depending on ``input_layout``, this transform supports:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

    Args:
        keys: Two keys containing the image tensor and label tensor.
        target_shape: Output crop size as ``(H, W)`` for 2D inputs or
            ``(D, H, W)`` for 3D inputs.
        pos: Relative weight for positive-center sampling.
        neg: Relative weight for negative-center sampling.
        num_samples: Number of samples to return. Currently only ``1`` is
            supported.
        input_layout: Channel-last tensor layout. Supported values are
            ``"HWC"``, ``"DHWC"``, ``"BHWC"``, and ``"BDHWC"``. In batch
            layouts, one sampled crop is shared across the full batch.
        image_reference_key: Optional key for an intensity reference tensor
            used to constrain negative sampling.
        image_threshold: Threshold applied to ``image_reference_key`` during
            negative sampling.
        seed: Optional random seed. Supports ``None``, an integer seed, or a
            ``keras.random.SeedGenerator``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Randomly crop a 2D image-label pair using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomCropByPosNegLabel

            transform = RandomCropByPosNegLabel(
                keys=["image", "label"],
                target_shape=(32, 32),
                pos=1,
                neg=1,
                input_layout="HWC",
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
                target_shape=(16, 32, 32),
                pos=1,
                neg=1,
                input_layout="DHWC",
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
        target_shape: Sequence[int],
        pos: int,
        neg: int,
        num_samples: int = 1,
        *,
        input_layout: str,
        image_reference_key: str = None,
        image_threshold: float = 0.0,
        seed: int | keras.random.SeedGenerator | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=1.0, seed=seed)
        self._validate_sampling_weights(pos, neg)
        normalized_keys = self._validate_keys(keys)
        self._validate_num_samples(num_samples)

        self.keys = normalized_keys
        self.target_shape = target_shape
        self.pos = pos
        self.neg = neg
        self.num_samples = num_samples
        self.pos_ratio = pos / (pos + neg)
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            transform_name=type(self).__name__,
        )
        self.layout_info = get_input_layout_info(self.input_layout)
        self.image_reference_key = image_reference_key
        self.image_threshold = image_threshold
        self.allow_missing_keys = allow_missing_keys
        self.crop = SpatialCrop(
            keys=self.keys,
            crop_size=self.target_shape,
            input_layout=self.input_layout,
            allow_missing_keys=self.allow_missing_keys,
        )

    @property
    def invertible(self) -> bool:
        return True

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        params = self.get_random_params(bundle)
        return self.apply_with_params(bundle, params)

    def get_random_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Sample one crop configuration shared across selected keys."""
        image_key, label_key = self.keys
        if image_key not in bundle.data or label_key not in bundle.data:
            if self.allow_missing_keys:
                return {"skip": True}
            missing = image_key if image_key not in bundle.data else label_key
            raise KeyError(f"Key '{missing}' not found in input data.")

        image = bundle.data[image_key]
        label = bundle.data[label_key]
        layout = validate_tensor_matches_layout(
            image,
            self.input_layout,
            transform_name=type(self).__name__,
        )
        validate_tensor_matches_layout(
            label,
            self.input_layout,
            transform_name=type(self).__name__,
        )
        spatial_rank = layout.spatial_rank
        image_batched, _ = ensure_batch_axis_for_layout(
            image,
            input_layout=self.input_layout,
        )
        label_batched, _ = ensure_batch_axis_for_layout(
            label,
            input_layout=self.input_layout,
        )

        image_reference = None
        if self.image_reference_key is not None:
            if self.image_reference_key not in bundle.data:
                raise KeyError(f"Key '{self.image_reference_key}' not found in input data.")
            image_reference = bundle.data[self.image_reference_key]
        image_reference_batched = None
        if image_reference is not None:
            image_reference_batched, _ = ensure_batch_axis_for_layout(
                image_reference,
                input_layout=self.input_layout,
            )
        center = self.sample_center(
            image_batched,
            label_batched,
            image_reference_batched,
            spatial_rank,
        )
        crop_size = tf.convert_to_tensor(self.target_shape, dtype=tf.int32)
        if crop_size.shape.rank != 1 or crop_size.shape[0] != spatial_rank:
            raise ValueError(
                f"`target_shape` must contain exactly {spatial_rank} values for input shape "
                f"{image.shape}; received {self.target_shape}."
            )
        spatial_shape = get_spatial_shape_for_layout(
            image_batched,
            input_layout=self.crop.input_layout,
        )
        starts = tf.maximum(center - crop_size // 2, 0)
        ends = tf.minimum(starts + crop_size, spatial_shape)
        starts = tf.maximum(ends - crop_size, 0)
        return {
            "skip": False,
            "crop_start": starts,
            "crop_size": crop_size,
            "pos": self.pos,
            "neg": self.neg,
            "image_reference_key": self.image_reference_key,
            "input_layout": self.input_layout,
        }

    def _validate_sampling_weights(self, pos: int, neg: int) -> None:
        """Validate positive/negative sampling weights."""
        if pos < 0 or neg < 0:
            raise ValueError("pos and neg must be non-negative.")
        if pos == 0 and neg == 0:
            raise ValueError("pos and neg cannot both be zero.")

    def _validate_keys(self, keys: Sequence[str]) -> list[str]:
        """Validate that exactly two keys are provided."""
        normalized_keys = _normalize_keys(keys)
        if len(normalized_keys) != 2:
            class_name = type(self).__name__
            raise ValueError(
                f"{class_name} transformation requires a pair of image and label as keys. "
            )
        return normalized_keys

    def _validate_num_samples(self, num_samples: int) -> None:
        """Validate the currently supported sample count."""
        if num_samples != 1:
            class_name = type(self).__name__
            raise ValueError(f"{class_name} transformation currently supports only num_samples=1.")

    def apply_with_params(
        self,
        bundle: TensorBundle,
        params: dict[str, object],
    ) -> TensorBundle:
        """Apply the sampled crop configuration to all selected keys."""
        if params["skip"]:
            return bundle

        original_shapes = {}

        def apply_crop(tensor: tf.Tensor, key: str) -> tf.Tensor:
            original_shapes[key] = get_spatial_shape_for_layout(
                tensor,
                input_layout=self.input_layout,
            )
            batched_tensor, added_batch_axis = ensure_batch_axis_for_layout(
                tensor,
                input_layout=self.input_layout,
            )
            cropped = self.crop.crop_tensor(
                batched_tensor,
                params["crop_start"],
                params["crop_size"],
            )
            return restore_from_batch_axis(cropped, added_batch_axis)

        present_keys = self.crop.apply_to_present_keys(
            bundle,
            apply_crop,
        )
        self.record_random_transform(
            bundle,
            params=self.build_trace_params(params, present_keys, original_shapes),
            applied=True,
            kernel="SpatialCrop",
        )
        return bundle

    def build_trace_params(
        self,
        params: dict[str, object],
        present_keys: Sequence[str],
        original_shapes: dict[str, tf.Tensor],
    ) -> dict[str, object]:
        """Build random trace metadata for the current positive/negative crop."""
        return {
            "keys": list(present_keys),
            "crop_start": params["crop_start"],
            "crop_size": params["crop_size"],
            "original_shapes": original_shapes,
            "pos": params["pos"],
            "neg": params["neg"],
            "image_reference_key": params["image_reference_key"],
            "input_layout": params["input_layout"],
        }

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = self._get_last_random_crop_trace(bundle)
        if trace is None:
            return bundle

        crop_start = trace["params"].get("crop_start")
        original_shapes = trace["params"].get("original_shapes", {})

        def apply_inverse_crop(tensor: tf.Tensor, key: str) -> tf.Tensor:
            original_shape = original_shapes.get(key)
            if original_shape is None:
                return tensor
            batched_tensor, added_batch_axis = ensure_batch_axis_for_layout(
                tensor,
                input_layout=self.input_layout,
            )
            restored = self.crop.pad_to_original_shape(batched_tensor, crop_start, original_shape)
            return restore_from_batch_axis(restored, added_batch_axis)

        self.crop.apply_to_present_keys(
            bundle,
            apply_inverse_crop,
            keys=trace["params"].get("keys", []),
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
        positive = self.random_uniform(
            shape=(),
            minval=0.0,
            maxval=1.0,
            dtype=tf.float32,
        ) < self.pos_ratio
        return tf.cond(
            positive,
            lambda: self._sample_positive_center(label, spatial_rank),
            lambda: self._sample_negative_center(image, label, image_reference, spatial_rank),
        )

    def _sample_positive_center(self, label: tf.Tensor, spatial_rank: int) -> tf.Tensor:
        coords = tf.where(tf.reduce_any(label > 0, axis=(0, -1)))
        return self._sample_from_coords(
            coords,
            fallback_shape=get_spatial_shape_for_layout(
                label,
                input_layout=self.crop.input_layout,
            ),
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
            max_intensity_ref = tf.reduce_max(image_reference, axis=(0, -1))
            label_is_zero = tf.reduce_any(label == 0, axis=(0, -1))
            valid_mask = label_is_zero & (max_intensity_ref > self.image_threshold)
            coords = tf.where(valid_mask)
        else:
            coords = tf.where(tf.reduce_any(label == 0, axis=(0, -1)))
        return self._sample_from_coords(
            coords,
            fallback_shape=get_spatial_shape_for_layout(
                image,
                input_layout=self.crop.input_layout,
            ),
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
            num_cols = coords.shape[1] if coords.shape[1] is not None else tf.shape(coords)[1]
            random_unit = self.random_uniform(
                shape=(spatial_rank,),
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32,
            )
            random_coord = tf.cast(
                tf.floor(random_unit * tf.cast(fallback_shape[:spatial_rank], tf.float32)),
                tf.int32,
            )
            padding = tf.zeros([num_cols - spatial_rank], dtype=tf.int32)
            full_coord = tf.concat([random_coord, padding], axis=0)
            return tf.expand_dims(tf.cast(full_coord, coords.dtype), axis=0)

        coords = tf.cond(tf.shape(coords)[0] > 0, lambda: coords, fallback_coords)
        idx = self.random_integers(
            shape=(),
            minval=0,
            maxval=tf.shape(coords)[0],
            dtype=tf.int32,
        )
        return tf.cast(coords[idx][:spatial_rank], tf.int32)

    def _get_last_random_crop_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)

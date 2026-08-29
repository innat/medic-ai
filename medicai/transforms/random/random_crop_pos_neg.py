from __future__ import annotations

from typing import Any, Sequence

import keras
from keras import ops

from ..base import RandomTransform, _normalize_keys, _pop_last_transform_trace
from ..spatial.spatial_crop import SpatialCrop
from ..tensor_bundle import TensorBundle
from ..utils import (
    ensure_batch_axis_for_layout,
    get_input_layout_info,
    get_spatial_shape_for_layout,
    get_tensor_rank,
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

        TensorFlow backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

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

        JAX backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            import jax.numpy as jnp
            from medicai.transforms import RandomCropByPosNegLabel

            transform = RandomCropByPosNegLabel(
                keys=["image", "label"],
                target_shape=(16, 32, 32),
                pos=1,
                neg=1,
                input_layout="DHWC",
            )

            image = jax.random.normal(
                jax.random.PRNGKey(7), shape=(32, 64, 64, 1)
            )
            label = jnp.asarray(image > 0, dtype=jnp.int32)
            result = transform({"image": image, "label": label})
            output = result["image"]
            print(output.shape)

        Torch backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import RandomCropByPosNegLabel

            transform = RandomCropByPosNegLabel(
                keys=["image", "label"],
                target_shape=(32, 32),
                pos=1,
                neg=1,
                input_layout="BHWC",
            )

            torch.manual_seed(7)
            image = torch.randn((2, 64, 64, 1))
            label = (image > 0).to(torch.int32)
            result = transform({"image": image, "label": label})
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
        image_reference_key: str | None = None,
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
        self.batch_input_layout = "BDHWC" if self.layout_info.spatial_rank == 3 else "BHWC"
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
        crop_size = ops.convert_to_tensor(self.target_shape, dtype="int32")
        if get_tensor_rank(crop_size) != 1 or crop_size.shape[0] != spatial_rank:
            raise ValueError(
                f"`target_shape` must contain exactly {spatial_rank} values for input shape "
                f"{image.shape}; received {self.target_shape}."
            )
        spatial_shape = get_spatial_shape_for_layout(
            image_batched,
            input_layout=self.batch_input_layout,
        )
        starts = ops.maximum(center - crop_size // 2, 0)
        ends = ops.minimum(starts + crop_size, spatial_shape)
        starts = ops.maximum(ends - crop_size, 0)
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

        def apply_crop(tensor: Any, key: str) -> Any:
            original_shapes[key] = get_spatial_shape_for_layout(
                tensor,
                input_layout=self.input_layout,
            )
            batched_tensor, added_batch_axis = ensure_batch_axis_for_layout(
                tensor,
                input_layout=self.input_layout,
            )
            cropped = self.crop_tensor(
                batched_tensor,
                params["crop_start"],
                params["crop_size"],
                input_layout=self.batch_input_layout,
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
        original_shapes: dict[str, Any],
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

        def apply_inverse_crop(tensor: Any, key: str) -> Any:
            original_shape = original_shapes.get(key)
            if original_shape is None:
                return tensor
            batched_tensor, added_batch_axis = ensure_batch_axis_for_layout(
                tensor,
                input_layout=self.input_layout,
            )
            restored = self.pad_to_original_shape(
                batched_tensor,
                crop_start,
                original_shape,
                input_layout=self.batch_input_layout,
            )
            return restore_from_batch_axis(restored, added_batch_axis)

        self.crop.apply_to_present_keys(
            bundle,
            apply_inverse_crop,
            keys=trace["params"].get("keys", []),
        )
        return bundle

    def sample_center(
        self,
        image,
        label,
        image_reference,
        spatial_rank: int,
    ):
        """Sample one crop center using positive/negative label sampling."""
        positive = (
            self.random_uniform(
                shape=(),
                minval=0.0,
                maxval=1.0,
                dtype="float32",
            )
            < self.pos_ratio
        )
        positive_center = self._sample_positive_center(label, spatial_rank)
        negative_center = self._sample_negative_center(image, label, image_reference, spatial_rank)
        return ops.where(positive, positive_center, negative_center)

    def _sample_positive_center(self, label, spatial_rank: int):
        return self._sample_from_mask(
            ops.any(label > 0, axis=(0, -1)),
            fallback_shape=get_spatial_shape_for_layout(
                label,
                input_layout=self.batch_input_layout,
            ),
            spatial_rank=spatial_rank,
        )

    def _sample_negative_center(
        self,
        image,
        label,
        image_reference,
        spatial_rank: int,
    ):
        if image_reference is not None and self.image_threshold is not None:
            max_intensity_ref = ops.max(image_reference, axis=(0, -1))
            label_is_zero = ops.any(label == 0, axis=(0, -1))
            valid_mask = label_is_zero & (max_intensity_ref > self.image_threshold)
        else:
            valid_mask = ops.any(label == 0, axis=(0, -1))
        return self._sample_from_mask(
            valid_mask,
            fallback_shape=get_spatial_shape_for_layout(
                image,
                input_layout=self.batch_input_layout,
            ),
            spatial_rank=spatial_rank,
        )

    def _sample_from_mask(
        self,
        valid_mask,
        fallback_shape,
        spatial_rank: int,
    ):
        """Sample one spatial coordinate, falling back to any valid voxel if empty."""
        axes = [ops.arange(fallback_shape[index]) for index in range(spatial_rank)]
        grid = ops.meshgrid(*axes, indexing="ij")
        coords = ops.reshape(ops.stack(grid, axis=-1), (-1, spatial_rank))
        valid_flat = ops.reshape(valid_mask, (-1,))
        valid_int = ops.cast(valid_flat, "int32")
        num_valid = ops.sum(valid_int)
        ranks = ops.cumsum(valid_int) - 1
        valid_rank = self.random_integers(
            shape=(),
            minval=0,
            maxval=ops.maximum(num_valid, 1),
            dtype="int32",
        )
        selected_index = ops.argmax(
            ops.cast((valid_int > 0) & (ranks == valid_rank), "int32"),
            axis=0,
        )
        selected = ops.cast(coords[selected_index], "int32")
        random_unit = self.random_uniform(
            shape=(spatial_rank,),
            minval=0.0,
            maxval=1.0,
            dtype="float32",
        )
        fallback = ops.cast(
            ops.floor(random_unit * ops.cast(fallback_shape[:spatial_rank], "float32")),
            "int32",
        )
        return ops.where(num_valid > 0, selected, fallback)

    def _get_last_random_crop_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)

    def crop_tensor(
        self,
        tensor,
        crop_start,
        crop_size,
        *,
        input_layout: str,
    ):
        """Crop one tensor using the provided layout contract."""
        layout = get_input_layout_info(input_layout)
        if layout.batched:
            begin = ops.concatenate(
                [
                    ops.convert_to_tensor([0], dtype="int32"),
                    crop_start,
                    ops.convert_to_tensor([0], dtype="int32"),
                ],
                axis=0,
            )
            size = ops.concatenate(
                [
                    ops.reshape(ops.shape(tensor)[0], (1,)),
                    crop_size,
                    ops.reshape(ops.shape(tensor)[-1], (1,)),
                ],
                axis=0,
            )
        else:
            begin = ops.concatenate(
                [
                    crop_start,
                    ops.convert_to_tensor([0], dtype="int32"),
                ],
                axis=0,
            )
            size = ops.concatenate(
                [
                    crop_size,
                    ops.reshape(ops.shape(tensor)[-1], (1,)),
                ],
                axis=0,
            )
        return ops.slice(tensor, start_indices=begin, shape=size)

    def pad_to_original_shape(
        self,
        tensor,
        crop_start,
        original_shape,
        *,
        input_layout: str,
    ):
        """Pad one tensor back into its original spatial canvas."""
        crop_start = ops.cast(crop_start, "int32")
        original_shape = ops.cast(original_shape, "int32")
        current_shape = get_spatial_shape_for_layout(tensor, input_layout=input_layout)
        pad_before = crop_start
        pad_after = ops.maximum(original_shape - crop_start - current_shape, 0)
        paddings = [[0, 0]]
        for before, after in zip(ops.unstack(pad_before), ops.unstack(pad_after), strict=True):
            paddings.append([before, after])
        paddings.append([0, 0])
        return ops.pad(tensor, paddings)

from __future__ import annotations

from numbers import Integral
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
    resolve_input_layout,
    restore_from_batch_axis,
    validate_tensor_matches_layout,
)

_SUPPORTED_NUM_SAMPLES = {1}
_DEFAULT_NUM_SAMPLES = 1
_DEFAULT_IMAGE_THRESHOLD = 0.0
_SAMPLE_INPUT_LAYOUTS = ("HWC", "DHWC")
_BATCH_INPUT_LAYOUTS = ("BHWC", "BDHWC")
_SUPPORTED_INPUT_LAYOUTS = _SAMPLE_INPUT_LAYOUTS + _BATCH_INPUT_LAYOUTS


class RandomCropByPosNegLabel(RandomTransform):
    """Randomly crop one image-label patch using positive and negative label sampling.

    A crop center is sampled from either positive-label voxels or negative
    voxels according to the ``pos:neg`` ratio, then the same patch is cropped
    from both image and label tensors.

    Binary labels and sparse multiclass labels are supported with one channel.
    Multi-label targets are supported with multiple channels: a voxel is
    foreground when any label channel is nonzero and background only when all
    label channels are zero. Channel-wise binary masks should therefore omit
    any dedicated background channel. One-hot targets that include a background
    channel are not supported as-is. Class-specific sampling is not performed.

    Args:
        keys: Two keys containing the image tensor and label tensor.
        target_shape: Output crop size as ``(H, W)`` for 2D inputs or
            ``(D, H, W)`` for 3D inputs.
        pos: Relative weight for positive-center sampling.
        neg: Relative weight for negative-center sampling.
        num_samples: Number of samples to return. Currently only ``1`` is
            supported.
        input_layout: Channel-last sample layout. Supported values are
            ``"HWC"`` and ``"DHWC"``. Batch layouts are intentionally not
            supported because label-aware dynamic cropping requires a
            backend-specific per-sample indexing path.
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
                input_layout="HWC",
            )

            torch.manual_seed(7)
            image = torch.randn((64, 64, 1))
            label = (image > 0).to(torch.int32)
            result = transform({"image": image, "label": label})
            output = result["image"]
            print(output.shape)

        Multi-label segmentation:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import keras
            from medicai.transforms import RandomCropByPosNegLabel

            transform = RandomCropByPosNegLabel(
                keys=["image", "label"],
                target_shape=(96, 128, 128),
                pos=1,
                neg=1,
                input_layout="DHWC",
            )

            image = keras.ops.zeros((155, 240, 240, 4), dtype="float32")
            label = keras.ops.zeros((155, 240, 240, 3), dtype="int32")
            result = transform({"image": image, "label": label})
            print(result["image"].shape, result["label"].shape)
    """

    def __init__(
        self,
        keys: Sequence[str],
        target_shape: Sequence[int],
        pos: int,
        neg: int,
        num_samples: int = _DEFAULT_NUM_SAMPLES,
        *,
        input_layout: str,
        image_reference_key: str | None = None,
        image_threshold: float = _DEFAULT_IMAGE_THRESHOLD,
        seed: int | keras.random.SeedGenerator | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=1.0, seed=seed)
        self._validate_sampling_weights(pos, neg)
        normalized_keys = self._validate_keys(keys)
        self._validate_num_samples(num_samples)

        self.keys = normalized_keys
        self.pos = pos
        self.neg = neg
        self.num_samples = num_samples
        self.pos_ratio = pos / (pos + neg)
        self.input_layout = self._validate_input_layout(input_layout)
        self.layout_info = get_input_layout_info(self.input_layout)
        self.target_shape = self._validate_target_shape(target_shape)
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
        """Sample one crop configuration for the input sample."""
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

        spatial_shape = get_spatial_shape_for_layout(
            image_batched,
            input_layout=self.batch_input_layout,
        )

        # Keep each crop inside the image while preserving the requested size.
        starts = ops.maximum(center - crop_size // 2, 0)
        ends = ops.minimum(starts + crop_size, spatial_shape)
        starts = ops.maximum(ends - crop_size, 0)
        starts = ops.squeeze(starts, axis=0)

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
        if num_samples not in _SUPPORTED_NUM_SAMPLES:
            class_name = type(self).__name__
            raise ValueError(f"{class_name} transformation currently supports only num_samples=1.")

    def _validate_input_layout(self, input_layout: str) -> str:
        """Validate the public layout and reject unsupported batch layouts."""
        normalized = resolve_input_layout(
            input_layout=input_layout,
            allowed_layouts=_SUPPORTED_INPUT_LAYOUTS,
            transform_name=type(self).__name__,
        )
        if normalized in _BATCH_INPUT_LAYOUTS:
            raise ValueError(
                f"{type(self).__name__} does not support batch input_layout "
                f"{normalized!r} yet. Use a sample layout: "
                f"{', '.join(_SAMPLE_INPUT_LAYOUTS)}. If batched label-aware "
                "cropping is important for your workflow, please raise a "
                "GitHub issue with a proposal at "
                "https://github.com/innat/medic-ai/issues."
            )
        return normalized

    def _validate_target_shape(self, target_shape: Sequence[int]) -> tuple[int, ...]:
        """Validate and normalize the crop size for the configured spatial rank."""
        if not isinstance(target_shape, (tuple, list)):
            raise TypeError("`target_shape` must be a tuple or list of integer sizes.")

        spatial_rank = self.layout_info.spatial_rank
        if len(target_shape) != spatial_rank:
            raise ValueError(
                f"`target_shape` must contain exactly {spatial_rank} values; "
                f"received {target_shape}."
            )

        if not all(isinstance(value, Integral) for value in target_shape):
            raise TypeError("`target_shape` values must be integers.")

        normalized = tuple(int(value) for value in target_shape)
        if any(value <= 0 for value in normalized):
            raise ValueError("`target_shape` values must be positive.")
        return normalized

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
            layout = get_input_layout_info(self.input_layout)
            spatial_shape = tuple(tensor.shape[axis] for axis in layout.spatial_axes)
            if any(dimension is None for dimension in spatial_shape):
                raise ValueError(
                    "RandomCropByPosNegLabel requires statically known spatial dimensions."
                )
            original_shapes[key] = tuple(int(dimension) for dimension in spatial_shape)
            batched_tensor, added_batch_axis = ensure_batch_axis_for_layout(
                tensor,
                input_layout=self.input_layout,
            )

            cropped = self.crop_tensor(
                batched_tensor,
                params["crop_start"],
                self.target_shape,
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
        batch_size = ops.shape(label)[0]
        positive = (
            self.random_uniform(
                shape=(batch_size,),
                minval=0.0,
                maxval=1.0,
                dtype="float32",
            )
            < self.pos_ratio
        )
        positive_center = self._sample_positive_center(label, spatial_rank)
        negative_center = self._sample_negative_center(
            image,
            label,
            image_reference,
            spatial_rank,
        )
        return ops.where(positive[:, None], positive_center, negative_center)

    def _sample_positive_center(self, label, spatial_rank: int):
        return self._sample_from_mask(
            self._foreground_mask(label),
            fallback_shape=get_spatial_shape_for_layout(
                label,
                input_layout=self.batch_input_layout,
            ),
            spatial_rank=spatial_rank,
        )

    def _foreground_mask(self, label):
        """Return the union foreground mask across label channels."""
        return ops.any(label > 0, axis=-1)

    def _background_mask(self, label):
        """Return voxels where every label channel is zero."""
        return ops.all(label == 0, axis=-1)

    def _sample_negative_center(
        self,
        image,
        label,
        image_reference,
        spatial_rank: int,
    ):
        if image_reference is not None and self.image_threshold is not None:
            max_intensity_ref = ops.max(image_reference, axis=-1)
            valid_mask = self._background_mask(label) & (max_intensity_ref > self.image_threshold)
        else:
            valid_mask = self._background_mask(label)
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
        batch_size = ops.shape(valid_mask)[0]
        valid_flat = ops.reshape(valid_mask, (batch_size, -1))
        valid_values = ops.cast(valid_flat, "int32")
        num_valid = ops.sum(valid_values, axis=1)
        random_rank = ops.floor(
            self.random_uniform(
                shape=(batch_size,),
                minval=0.0,
                maxval=1.0,
                dtype="float32",
            )
            * ops.cast(num_valid, "float32")
        )
        random_rank = ops.cast(random_rank, "int32")
        cumulative = ops.cumsum(valid_values, axis=1)
        selected_flat = ops.argmax(
            ops.cast(cumulative > random_rank[:, None], "int32"),
            axis=1,
        )
        remaining = ops.cast(selected_flat, "int32")
        coordinates = []
        for dimension in reversed(range(spatial_rank)):
            size = ops.cast(fallback_shape[dimension], "int32")
            coordinates.append(ops.mod(remaining, size))
            remaining = ops.floor_divide(remaining, size)

        selected = ops.stack(list(reversed(coordinates)), axis=-1)
        random_unit = self.random_uniform(
            shape=(batch_size, spatial_rank),
            minval=0.0,
            maxval=1.0,
            dtype="float32",
        )
        fallback = ops.cast(
            ops.floor(random_unit * ops.cast(fallback_shape[:spatial_rank], "float32")),
            "int32",
        )
        return ops.where(num_valid[:, None] > 0, selected, fallback)

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
        crop_size = tuple(int(value) for value in crop_size)
        channel_size = tensor.shape[-1]
        if channel_size is None:
            raise ValueError("RandomCropByPosNegLabel requires a static channel dimension.")

        if layout.batched:
            begin = ops.concatenate(
                [
                    ops.convert_to_tensor([0], dtype="int32"),
                    crop_start,
                    ops.convert_to_tensor([0], dtype="int32"),
                ],
                axis=0,
            )
            size = (1, *crop_size, channel_size)
        else:
            begin = ops.concatenate(
                [
                    crop_start,
                    ops.convert_to_tensor([0], dtype="int32"),
                ],
                axis=0,
            )
            size = (*crop_size, channel_size)
        return ops.slice(tensor, start_indices=begin, shape=size)

    def pad_to_original_shape(
        self,
        tensor,
        crop_start,
        original_shape,
        *,
        input_layout: str,
    ):
        """Place one cropped tensor back into its original spatial canvas."""
        layout = get_input_layout_info(input_layout)
        crop_start = ops.cast(crop_start, "int32")
        original_shape = tuple(int(dimension) for dimension in original_shape)
        channel_size = tensor.shape[-1]
        if channel_size is None:
            raise ValueError("RandomCropByPosNegLabel requires a static channel dimension.")

        if layout.batched:
            batch_size = tensor.shape[0]
            if batch_size is None:
                raise ValueError("RandomCropByPosNegLabel requires a static batch dimension.")
            target_shape = (batch_size, *original_shape, channel_size)
            start_indices = ops.concatenate(
                [
                    ops.convert_to_tensor([0], dtype="int32"),
                    crop_start,
                    ops.convert_to_tensor([0], dtype="int32"),
                ],
                axis=0,
            )
        else:
            target_shape = (*original_shape, channel_size)
            start_indices = ops.concatenate(
                [crop_start, ops.convert_to_tensor([0], dtype="int32")],
                axis=0,
            )

        canvas = ops.zeros(target_shape, dtype=tensor.dtype)
        return ops.slice_update(canvas, start_indices, tensor)

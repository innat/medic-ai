from typing import Any, Sequence

import keras
from keras import ops

from ..base import RandomTransform, _normalize_keys, _pop_last_transform_trace
from ..spatial.spatial_crop import SpatialCrop
from ..tensor_bundle import TensorBundle
from ..utils import (
    get_spatial_shape_for_layout,
    get_tensor_rank,
    resolve_input_layout,
    validate_tensor_matches_layout,
)


class RandomSpatialCrop(RandomTransform):
    """Randomly crop a spatial region from selected tensors.

    ``RandomSpatialCrop`` samples a crop center and, optionally, a crop size
    before extracting a spatial patch with the deterministic
    :class:`~medicai.transforms.SpatialCrop` kernel.

    Depending on ``input_layout``, this transform supports:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

    When ``invalid_label`` is provided, the crop center can be sampled using
    the ``"label"`` tensor to favor valid regions.

    Args:
        keys: Keys of the tensors to crop.
        crop_size: Minimum or fixed crop size.
        max_crop_size: Maximum crop size when ``random_shape=True``.
        random_center: If ``True``, sample crop centers randomly.
        random_shape: If ``True``, sample crop sizes between ``crop_size`` and
            ``max_crop_size``.
        input_layout: Channel-last tensor layout. Supported values are
            ``"HWC"``, ``"DHWC"``, ``"BHWC"``, and ``"BDHWC"``. In batch
            layouts, one sampled crop is shared across the full batch.
        seed: Optional random seed. Supports ``None``, an integer seed, or a
            ``keras.random.SeedGenerator``.
        invalid_label: Label value treated as invalid when enforcing valid
            crop regions.
        min_valid_ratio: Minimum fraction of valid labels required in a crop.
        max_attempts: Maximum attempts when searching for a crop that satisfies
            ``min_valid_ratio``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:

        TensorFlow backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import RandomSpatialCrop

            transform = RandomSpatialCrop(
                keys=["image"],
                crop_size=(32, 32),
                input_layout="HWC",
            )
            image = tf.random.normal((64, 64, 1))
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        JAX backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            from medicai.transforms import RandomSpatialCrop

            transform = RandomSpatialCrop(
                keys=["image"],
                crop_size=(16, 32, 32),
                input_layout="DHWC",
            )
            image = jax.random.normal(
                jax.random.PRNGKey(7), shape=(32, 64, 64, 1)
            )
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Torch backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import RandomSpatialCrop

            transform = RandomSpatialCrop(
                keys=["image"], crop_size=(32, 32), input_layout="BHWC"
            )
            torch.manual_seed(7)
            batch = torch.randn((2, 64, 64, 1))
            result = transform({"image": batch})
            output = result["image"]
            print(output.shape)
    """

    def __init__(
        self,
        keys: Sequence[str],
        crop_size,
        max_crop_size=None,
        random_center: bool = True,
        random_shape: bool = False,
        *,
        input_layout: str,
        seed: int | keras.random.SeedGenerator | None = None,
        invalid_label=None,
        min_valid_ratio: float = 0.0,
        max_attempts: int = 1,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=1.0, seed=seed)
        self.keys = _normalize_keys(keys)
        self.crop_size = crop_size
        self.max_crop_size = max_crop_size
        self.random_center = random_center
        self.random_shape = random_shape
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            transform_name=type(self).__name__,
        )
        self.invalid_label = invalid_label
        self.min_valid_ratio = min_valid_ratio
        self.max_attempts = max_attempts
        self.allow_missing_keys = allow_missing_keys
        self.crop = SpatialCrop(
            keys=self.keys,
            crop_size=self.crop_size,
            input_layout=self.input_layout,
            allow_missing_keys=self.allow_missing_keys,
        )

        if not (0.0 <= min_valid_ratio <= 1.0):
            raise ValueError(f"min_valid_ratio must be in range [0.0, 1.0], got {min_valid_ratio}")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be a positive integer, got {max_attempts}")
        if min_valid_ratio > 0.0 and invalid_label is None:
            raise ValueError(
                "If min_valid_ratio > 0, you must provide an invalid_label (e.g., 0) "
                "to calculate the ratio of valid pixels."
            )

    @property
    def invertible(self) -> bool:
        return True

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        params = self.get_random_params(bundle)
        return self.apply_with_params(bundle, params)

    def get_random_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Sample one crop configuration shared across selected keys."""
        sample_key = self.keys[0]
        if sample_key not in bundle.data:
            if self.allow_missing_keys:
                return {"skip": True}
            raise KeyError(f"Key '{sample_key}' not found in input data.")

        sample_tensor = bundle.data[sample_key]
        layout = validate_tensor_matches_layout(
            sample_tensor,
            self.input_layout,
            transform_name=type(self).__name__,
        )
        spatial_rank = layout.spatial_rank
        spatial_shape = get_spatial_shape_for_layout(
            sample_tensor,
            input_layout=self.input_layout,
        )
        crop_size = self._get_crop_size(spatial_shape, spatial_rank)

        if self.invalid_label is None:
            center = self._get_random_center(spatial_shape, crop_size, spatial_rank)
        else:
            label_key = self.keys[1] if len(self.keys) > 1 else "label"
            if label_key not in bundle.data:
                raise KeyError(f"`{label_key}` key is required when `invalid_label` is specified.")
            center = self._get_label_aware_center(
                spatial_shape, crop_size, bundle[label_key], spatial_rank
            )

        starts = ops.maximum(center - crop_size // 2, 0)
        ends = ops.minimum(starts + crop_size, spatial_shape)
        starts = ops.maximum(ends - crop_size, 0)
        return {
            "skip": False,
            "crop_start": starts,
            "crop_size": crop_size,
            "random_center": self.random_center,
            "random_shape": self.random_shape,
            "input_layout": self.input_layout,
        }

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
            return self.crop.crop_tensor(tensor, params["crop_start"], params["crop_size"])

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

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = self._get_last_random_spatial_crop_trace(bundle)
        if trace is None:
            return bundle

        crop_start = trace["params"].get("crop_start")
        original_shapes = trace["params"].get("original_shapes", {})

        def apply_inverse_crop(tensor: Any, key: str) -> Any:
            original_shape = original_shapes.get(key)
            if original_shape is None:
                return tensor
            return self.crop.pad_to_original_shape(tensor, crop_start, original_shape)

        self.crop.apply_to_present_keys(
            bundle,
            apply_inverse_crop,
            keys=trace["params"].get("keys", []),
        )
        return bundle

    def build_trace_params(
        self,
        params: dict[str, object],
        present_keys: Sequence[str],
        original_shapes: dict[str, Any],
    ) -> dict[str, object]:
        """Build random trace metadata for the current spatial crop."""
        return {
            "keys": list(present_keys),
            "crop_start": params["crop_start"],
            "crop_size": params["crop_size"],
            "original_shapes": original_shapes,
            "random_center": params["random_center"],
            "random_shape": params["random_shape"],
            "input_layout": params["input_layout"],
        }

    def _get_crop_size(self, spatial_shape: Any, spatial_rank: int) -> Any:
        if isinstance(self.crop_size, int):
            crop_size = ops.full((spatial_rank,), self.crop_size, dtype="int32")
        else:
            crop_size = ops.convert_to_tensor(self.crop_size, dtype="int32")
            if get_tensor_rank(crop_size) != 1 or crop_size.shape[0] != spatial_rank:
                raise ValueError(
                    f"Expected spatial rank in (2, 3) with crop_size length matching the "
                    f"input spatial rank {spatial_rank}, got crop_size={self.crop_size!r}."
                )

        if self.random_shape:
            max_crop_size = (
                ops.full((spatial_rank,), self.max_crop_size, dtype="int32")
                if isinstance(self.max_crop_size, int)
                else (
                    ops.convert_to_tensor(self.max_crop_size, dtype="int32")
                    if self.max_crop_size is not None
                    else spatial_shape
                )
            )
            max_crop_size = ops.where(max_crop_size <= 0, spatial_shape, max_crop_size)
            min_s = ops.where(crop_size <= 0, spatial_shape, crop_size)
            max_s = ops.where(max_crop_size <= 0, spatial_shape, max_crop_size)
            max_s = ops.minimum(max_s, spatial_shape)
            min_s = ops.minimum(min_s, max_s)
            span = max_s - min_s + 1
            random_unit = self.random_uniform(
                shape=[spatial_rank],
                minval=0.0,
                maxval=1.0,
                dtype="float32",
            )
            crop_size = min_s + ops.cast(ops.floor(random_unit * ops.cast(span, "float32")), "int32")
        else:
            crop_size = ops.where(crop_size > 0, crop_size, spatial_shape)
            crop_size = ops.minimum(crop_size, spatial_shape)
        return crop_size

    def _get_random_center(
        self, spatial_shape: Any, crop_size: Any, spatial_rank: int
    ) -> Any:
        if not self.random_center:
            return spatial_shape // 2

        max_start = ops.maximum(spatial_shape - crop_size, 0)
        random_start = self.random_integers(
            shape=[spatial_rank],
            minval=0,
            maxval=ops.max(max_start) + 1,
            dtype="int32",
        )
        random_start = ops.minimum(random_start, max_start)
        return random_start + crop_size // 2

    def _get_label_aware_center(
        self, spatial_shape: Any, crop_size: Any, label: Any, spatial_rank: int
    ) -> Any:
        if get_tensor_rank(label) > spatial_rank:
            valid_mask = ops.any(label != self.invalid_label, axis=-1)
        else:
            valid_mask = label != self.invalid_label
        valid_coords = ops.where(valid_mask)
        if isinstance(valid_coords, (tuple, list)):
            valid_coords = ops.stack(valid_coords, axis=-1)
        elif valid_coords.shape[0] == spatial_rank:
            valid_coords = ops.transpose(valid_coords, axes=(1, 0))

        def fallback():
            return self._get_random_center(spatial_shape, crop_size, spatial_rank)

        num_valid = ops.shape(valid_coords)[0]
        safe_coords = ops.concatenate(
            [
                ops.cast(valid_coords, "int32"),
                ops.zeros((1, spatial_rank), dtype="int32"),
            ],
            axis=0,
        )
        idx = self.random_integers(
            shape=(),
            minval=0,
            maxval=ops.maximum(num_valid, 1),
            dtype="int32",
        )
        selected = ops.cast(safe_coords[idx][:spatial_rank], "int32")
        center = ops.where(
            num_valid > 0,
            selected,
            fallback(),
        )

        if self.min_valid_ratio > 0:
            center = self._enforce_min_valid_ratio(
                center, spatial_shape, crop_size, label, spatial_rank
            )

        return center

    def _enforce_min_valid_ratio(
        self,
        center: Any,
        spatial_shape: Any,
        crop_size: Any,
        label: Any,
        spatial_rank: int,
    ) -> Any:
        def body(i, current_center):
            starts = ops.maximum(current_center - crop_size // 2, 0)
            ends = ops.minimum(starts + crop_size, spatial_shape)
            starts = ops.maximum(ends - crop_size, 0)
            if get_tensor_rank(label) > spatial_rank:
                begin = ops.concatenate(
                    [starts, ops.convert_to_tensor([0], dtype="int32")],
                    axis=0,
                )
                size = ops.concatenate(
                    [crop_size, ops.reshape(ops.shape(label)[-1], (1,))],
                    axis=0,
                )
                crop = ops.slice(label, start_indices=begin, shape=size)
            else:
                crop = ops.slice(label, start_indices=starts, shape=crop_size)

            valid_ratio = ops.mean(ops.cast(crop != self.invalid_label, "float32"))
            new_center = ops.cond(
                valid_ratio >= self.min_valid_ratio,
                lambda: current_center,
                lambda: self._get_random_center(spatial_shape, crop_size, spatial_rank),
            )
            return i + 1, new_center

        def cond(i, _):
            return i < self.max_attempts

        _, center = ops.while_loop(
            cond,
            body,
            (
                ops.convert_to_tensor(0, dtype="int32"),
                center,
            ),
            maximum_iterations=self.max_attempts,
        )
        return center

    def _get_last_random_spatial_crop_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)

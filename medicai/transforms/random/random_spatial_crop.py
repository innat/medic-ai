from typing import Sequence

import keras
import tensorflow as tf

from ..base import RandomTransform, _normalize_keys, _pop_last_transform_trace
from ..spatial.spatial_crop import SpatialCrop
from ..tensor_bundle import TensorBundle
from ..utils import (
    get_legacy_layout_components,
    get_spatial_shape,
    resolve_input_layout,
    validate_layout,
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
        Randomly crop a 2D image using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomSpatialCrop

            transform = RandomSpatialCrop(keys=["image"], crop_size=(32, 32))
            image = tf.random.normal((64, 64, 1))
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Randomly crop a 3D image stored in a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomSpatialCrop, TensorBundle

            transform = RandomSpatialCrop(keys=["image"], crop_size=(16, 32, 32))
            image = tf.random.normal((32, 64, 64, 1))
            bundle = TensorBundle({"image": image})
            result = transform(bundle)
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
        input_layout: str | None = None,
        spatial_dims: int | None = None,
        input_mode: str | None = None,
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
        self._uses_explicit_input_layout = input_layout is not None
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            input_mode=input_mode,
            spatial_dims=spatial_dims,
            transform_name=type(self).__name__,
        )
        self.input_mode, self.spatial_dims = get_legacy_layout_components(self.input_layout)
        self.invalid_label = invalid_label
        self.min_valid_ratio = min_valid_ratio
        self.max_attempts = max_attempts
        self.allow_missing_keys = allow_missing_keys
        self.crop = SpatialCrop(
            keys=self.keys,
            crop_size=self.crop_size,
            input_layout=self.input_layout,
            input_mode=self.input_mode,
            spatial_dims=self.spatial_dims,
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
        if self._uses_explicit_input_layout:
            layout = validate_tensor_matches_layout(
                sample_tensor,
                self.input_layout,
                transform_name=type(self).__name__,
            )
        else:
            layout = validate_layout(
                sample_tensor,
                input_mode=self.input_mode,
                allowed_spatial_ranks=(2, 3),
                spatial_dims=self.spatial_dims,
                transform_name=type(self).__name__,
            )
        spatial_rank = layout.spatial_rank
        spatial_shape = get_spatial_shape(sample_tensor, input_mode=self.input_mode)
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

        starts = tf.maximum(center - crop_size // 2, 0)
        ends = tf.minimum(starts + crop_size, spatial_shape)
        starts = tf.maximum(ends - crop_size, 0)
        return {
            "skip": False,
            "crop_start": starts,
            "crop_size": crop_size,
            "random_center": self.random_center,
            "random_shape": self.random_shape,
            "input_layout": self.input_layout,
            "input_mode": self.input_mode,
            "spatial_dims": self.spatial_dims,
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

        def apply_crop(tensor: tf.Tensor, key: str) -> tf.Tensor:
            original_shapes[key] = get_spatial_shape(tensor, input_mode=self.input_mode)
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

        def apply_inverse_crop(tensor: tf.Tensor, key: str) -> tf.Tensor:
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
        original_shapes: dict[str, tf.Tensor],
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
            "input_mode": params["input_mode"],
            "spatial_dims": params["spatial_dims"],
        }

    def _get_crop_size(self, spatial_shape: tf.Tensor, spatial_rank: int) -> tf.Tensor:
        if isinstance(self.crop_size, int):
            crop_size = tf.fill([spatial_rank], tf.cast(self.crop_size, tf.int32))
        else:
            crop_size = tf.convert_to_tensor(self.crop_size, dtype=tf.int32)

        if self.random_shape:
            max_crop_size = (
                tf.fill([spatial_rank], tf.cast(self.max_crop_size, tf.int32))
                if isinstance(self.max_crop_size, int)
                else (
                    tf.convert_to_tensor(self.max_crop_size, dtype=tf.int32)
                    if self.max_crop_size is not None
                    else spatial_shape
                )
            )
            max_crop_size = tf.where(max_crop_size <= 0, spatial_shape, max_crop_size)
            min_s = tf.where(crop_size <= 0, spatial_shape, crop_size)
            max_s = tf.where(max_crop_size <= 0, spatial_shape, max_crop_size)
            max_s = tf.minimum(max_s, spatial_shape)
            min_s = tf.minimum(min_s, max_s)
            span = max_s - min_s + 1
            random_unit = self.random_uniform(
                shape=[spatial_rank],
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32,
            )
            crop_size = min_s + tf.cast(tf.floor(random_unit * tf.cast(span, tf.float32)), tf.int32)
        else:
            crop_size = tf.where(crop_size > 0, crop_size, spatial_shape)
            crop_size = tf.minimum(crop_size, spatial_shape)
        return crop_size

    def _get_random_center(
        self, spatial_shape: tf.Tensor, crop_size: tf.Tensor, spatial_rank: int
    ) -> tf.Tensor:
        if not self.random_center:
            return spatial_shape // 2

        max_start = tf.maximum(spatial_shape - crop_size, 0)
        random_start = self.random_integers(
            shape=[spatial_rank],
            minval=0,
            maxval=tf.reduce_max(max_start) + 1,
            dtype=tf.int32,
        )
        random_start = tf.minimum(random_start, max_start)
        return random_start + crop_size // 2

    def _get_label_aware_center(
        self, spatial_shape: tf.Tensor, crop_size: tf.Tensor, label: tf.Tensor, spatial_rank: int
    ) -> tf.Tensor:
        if label.shape.rank is not None and label.shape.rank > spatial_rank:
            valid_mask = tf.reduce_any(label != self.invalid_label, axis=-1)
        else:
            valid_mask = label != self.invalid_label
        valid_coords = tf.where(valid_mask)

        def fallback():
            return self._get_random_center(spatial_shape, crop_size, spatial_rank)

        def sample_valid_center():
            idx = self.random_integers(
                shape=(),
                minval=0,
                maxval=tf.shape(valid_coords)[0],
                dtype=tf.int32,
            )
            return tf.cast(valid_coords[idx][:spatial_rank], tf.int32)

        center = tf.cond(tf.shape(valid_coords)[0] > 0, sample_valid_center, fallback)

        if self.min_valid_ratio > 0:
            center = self._enforce_min_valid_ratio(
                center, spatial_shape, crop_size, label, spatial_rank
            )

        return center

    def _enforce_min_valid_ratio(
        self,
        center: tf.Tensor,
        spatial_shape: tf.Tensor,
        crop_size: tf.Tensor,
        label: tf.Tensor,
        spatial_rank: int,
    ) -> tf.Tensor:
        def body(i, current_center):
            starts = tf.maximum(current_center - crop_size // 2, 0)
            ends = tf.minimum(starts + crop_size, spatial_shape)
            starts = tf.maximum(ends - crop_size, 0)
            if label.shape.rank is not None and label.shape.rank > spatial_rank:
                begin = tf.concat([starts, tf.constant([0], dtype=tf.int32)], axis=0)
                size = tf.concat([crop_size, [tf.shape(label)[-1]]], axis=0)
                crop = tf.slice(label, begin=begin, size=size)
            else:
                crop = tf.slice(label, begin=starts, size=crop_size)

            valid_ratio = tf.reduce_mean(tf.cast(crop != self.invalid_label, tf.float32))
            new_center = tf.cond(
                valid_ratio >= self.min_valid_ratio,
                lambda: current_center,
                lambda: self._get_random_center(spatial_shape, crop_size, spatial_rank),
            )
            return i + 1, new_center

        def cond(i, _):
            return i < self.max_attempts

        _, center = tf.while_loop(cond, body, [0, center], parallel_iterations=1)
        return center

    def _get_last_random_spatial_crop_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)

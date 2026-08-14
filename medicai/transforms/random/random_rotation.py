from typing import Sequence

import keras
import tensorflow as tf

from ..base import (
    RandomTransform,
    _apply_if_applied,
    _normalize_keys,
    _pop_last_transform_trace,
)
from ..tensor_bundle import TensorBundle
from ..utils import (
    ensure_batch_axis,
    get_legacy_layout_components,
    resolve_input_layout,
    restore_from_batch_axis,
    validate_tensor_matches_layout,
)


def get_rotation_matrix(angle: tf.Tensor, h: tf.Tensor, w: tf.Tensor) -> tf.Tensor:
    """Compute a projective transform matrix for 2D rotation around the image center."""
    h = tf.cast(h, tf.float32)
    w = tf.cast(w, tf.float32)

    x0 = w / 2.0
    y0 = h / 2.0

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    return tf.stack(
        [
            cos_a,
            sin_a,
            x0 - x0 * cos_a - y0 * sin_a,
            -sin_a,
            cos_a,
            y0 - y0 * cos_a + x0 * sin_a,
            0.0,
            0.0,
        ],
        axis=0,
    )


def rotate_volume(
    image: tf.Tensor,
    angle: tf.Tensor,
    interpolation: str = "BILINEAR",
    fill_value: float = 0.0,
) -> tf.Tensor:
    """Rotate a 4D ``(N, H, W, C)`` tensor slice-wise over the height-width plane."""
    original_dtype = image.dtype
    image = tf.cast(image, tf.float32)
    img_shape = tf.shape(image)
    h, w = img_shape[1], img_shape[2]
    matrix = get_rotation_matrix(angle, h, w)
    matrices = tf.tile(tf.expand_dims(matrix, 0), [img_shape[0], 1])

    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=matrices,
        output_shape=[h, w],
        interpolation=interpolation,
        fill_mode="CONSTANT",
        fill_value=tf.cast(fill_value, tf.float32),
    )
    return tf.cast(rotated, original_dtype)


class RandomRotate(RandomTransform):
    """Randomly rotate 3D volumes using slice-wise 2D projection transforms.

    ``RandomRotate`` samples an angle and rotates each depth slice in the
    height-width plane. The first key is treated like an image tensor and uses
    bilinear interpolation, while the optional second key is treated like a
    label tensor and uses nearest-neighbor interpolation.

    When ``fill_mode="constant"``, the transform exposes an inverse path that
    rotates by the negated sampled angle using the recorded trace. This is
    useful for geometric bookkeeping, but for non-zero arbitrary angles it is
    still a resampling-based, best-effort inverse rather than an exact
    round-trip reconstruction.

    Depending on ``input_layout``, this transform supports:

    - sample 3D tensors shaped ``(D, H, W, C)`` with ``input_layout="DHWC"``
    - batch 3D tensors shaped ``(B, D, H, W, C)`` with ``input_layout="BDHWC"``

    Args:
        keys: One or two keys. When two keys are provided, they are typically
            image then label.
        factor: Maximum absolute sampled rotation angle in radians.
        prob: Probability of applying the rotation.
        fill_value: Constant fill value for the primary image key when
            ``fill_mode="constant"``.
        fill_mode: Either ``"constant"`` or ``"crop"``.
        input_layout: Channel-last tensor layout. Supported values are
            ``"DHWC"`` and ``"BDHWC"``.
        seed: Optional random seed. Supports ``None``, an integer seed, or a
            ``keras.random.SeedGenerator``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Randomly rotate a 3D image-label pair using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomRotate

            transform = RandomRotate(
                keys=["image", "label"],
                factor=0.2,
                prob=0.5,
                input_layout="DHWC",
            )
            image = tf.random.normal((32, 64, 64, 1))
            label = tf.cast(image > 0, tf.int32)
            result = transform({"image": image, "label": label})
            output = result["image"]
            print(output.shape)

        Randomly rotate a 3D image-label pair stored in a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomRotate, TensorBundle

            transform = RandomRotate(
                keys=["image", "label"],
                factor=0.2,
                prob=0.5,
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
        factor: float = 0.1,
        prob: float = 0.8,
        fill_value: float = 0.0,
        fill_mode: str = "constant",
        *,
        input_layout: str,
        seed: int | keras.random.SeedGenerator | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob, seed=seed)
        normalized_keys = _normalize_keys(keys)
        if len(normalized_keys) not in (1, 2):
            raise ValueError("`keys` must have length 1 or 2.")
        if factor < 0:
            raise ValueError(f"`factor` must be non-negative. Received {factor}.")
        if fill_mode not in {"crop", "constant"}:
            raise ValueError("fill_mode must be either 'crop' or 'constant'.")

        self.keys = normalized_keys
        self.factor = factor
        self.fill_value = fill_value
        self.fill_mode = fill_mode
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            transform_name=type(self).__name__,
        )
        self.input_mode, self.spatial_dims = get_legacy_layout_components(self.input_layout)
        if self.input_layout not in {"DHWC", "BDHWC"}:
            raise ValueError(f"{type(self).__name__} supports only input_layout='DHWC' or 'BDHWC'.")
        self.allow_missing_keys = allow_missing_keys

    @property
    def invertible(self) -> bool:
        return self.fill_mode == "constant"

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        params = self.get_random_params(bundle)
        if params["skip"]:
            return bundle
        return self.apply_with_params(bundle, params)

    def get_random_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Sample one rotation configuration shared across selected keys."""
        present_keys = []
        for key in self.keys:
            if key in bundle.data:
                present_keys.append(key)
            elif not self.allow_missing_keys:
                raise KeyError(f"Key '{key}' not found in input data.")

        if not present_keys:
            return {"skip": True}

        sample_tensor = bundle.data[present_keys[0]]
        layout = validate_tensor_matches_layout(
            sample_tensor,
            self.input_layout,
            transform_name=type(self).__name__,
        )
        del layout

        should_rotate = self.sample_should_apply()
        angle = self.random_uniform(
            shape=(),
            minval=-self.factor,
            maxval=self.factor,
            dtype=tf.float32,
        )
        return {
            "skip": False,
            "keys": list(present_keys),
            "should_apply": should_rotate,
            "angle": angle,
            "factor": self.factor,
            "fill_mode": self.fill_mode,
            "input_layout": self.input_layout,
            "input_mode": self.input_mode,
            "spatial_dims": self.spatial_dims,
        }

    def apply_with_params(
        self,
        bundle: TensorBundle,
        params: dict[str, object],
    ) -> TensorBundle:
        """Apply the sampled rotation configuration to all selected keys."""
        for key in params["keys"]:
            tensor = bundle.data[key]
            bundle.data[key] = _apply_if_applied(
                params["should_apply"],
                lambda tensor=tensor, key=key: self.rotate_tensor(tensor, key, params["angle"]),
                lambda tensor=tensor: tensor,
            )

        self.record_random_transform(
            bundle,
            params=self.build_trace_params(params),
            applied=params["should_apply"],
            kernel="rotate_volume",
        )
        return bundle

    def build_trace_params(self, params: dict[str, object]) -> dict[str, object]:
        """Build random trace metadata for the current rotation."""
        return {
            "keys": params["keys"],
            "factor": params["factor"],
            "angle": params["angle"],
            "fill_mode": params["fill_mode"],
            "input_layout": params["input_layout"],
            "input_mode": params["input_mode"],
            "spatial_dims": params["spatial_dims"],
        }

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        if not self.invertible:
            return bundle

        trace = self._get_last_random_rotate_trace(bundle)
        if trace is None:
            return bundle

        applied = trace.get("applied", False)
        angle = trace["params"].get("angle")

        def apply_inverse_rotate(tensor: tf.Tensor, key: str) -> tf.Tensor:
            return _apply_if_applied(
                applied,
                lambda tensor=tensor, key=key: self.rotate_tensor(tensor, key, -angle),
                lambda tensor=tensor: tensor,
            )

        for key in trace["params"].get("keys", []):
            if key not in bundle.data:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"Key '{key}' not found in input data.")
            tensor = bundle.data[key]
            bundle.data[key] = apply_inverse_rotate(tensor, key)
        return bundle

    def rotate_tensor(self, tensor: tf.Tensor, key: str, angle: tf.Tensor) -> tf.Tensor:
        """Rotate one tensor and apply optional center crop cleanup."""
        batched_tensor, added_batch_axis = ensure_batch_axis(
            tensor,
            input_mode=self.input_mode,
            spatial_dims=self.spatial_dims,
            allowed_spatial_ranks=(3,),
        )
        rotated = self.rotate_batch_tensor(batched_tensor, key, angle)
        return restore_from_batch_axis(rotated, added_batch_axis)

    def rotate_batch_tensor(self, tensor: tf.Tensor, key: str, angle: tf.Tensor) -> tf.Tensor:
        """Rotate one batch-layout tensor and apply optional center crop cleanup."""
        interpolation = "BILINEAR" if key == self.keys[0] else "NEAREST"
        fill_value = self.fill_value if key == self.keys[0] else 0.0

        layout = validate_layout(
            tensor,
            input_mode="batch",
            allowed_spatial_ranks=(3,),
            spatial_dims=self.spatial_dims,
            transform_name=type(self).__name__,
        )
        del layout

        shape = tf.shape(tensor)
        batch_size = shape[0]
        depth = shape[1]
        height = shape[2]
        width = shape[3]
        channels = shape[4]

        flat_tensor = tf.reshape(tensor, [batch_size * depth, height, width, channels])
        flat_rotated = rotate_volume(
            flat_tensor,
            angle,
            interpolation=interpolation,
            fill_value=fill_value,
        )
        rotated = tf.reshape(flat_rotated, [batch_size, depth, height, width, channels])

        if self.fill_mode == "crop":
            rotated = self._crop_after_rotation(rotated, angle, interpolation)
        return rotated

    def _crop_after_rotation(
        self,
        tensor: tf.Tensor,
        angle: tf.Tensor,
        interpolation: str,
    ) -> tf.Tensor:
        """Apply a Largest Rectangle Rotation style center crop after rotation."""
        shape = tf.shape(tensor)
        batch_size = shape[0]
        depth = shape[1]
        height = shape[2]
        width = shape[3]
        channels = shape[4]
        lrr_w, lrr_h = self._get_lrr_size(width, height, angle)
        crop_fraction = (
            tf.minimum(
                lrr_h / tf.cast(height, tf.float32),
                lrr_w / tf.cast(width, tf.float32),
            )
            * 0.98
        )
        crop_fraction = tf.clip_by_value(crop_fraction, 1e-6, 1.0 - 1e-6)
        method = "bilinear" if interpolation == "BILINEAR" else "nearest"

        flat_tensor = tf.reshape(tensor, [batch_size * depth, height, width, channels])
        cropped = tf.image.central_crop(flat_tensor, crop_fraction)
        resized = tf.image.resize(
            cropped,
            [height, width],
            method=method,
        )
        return tf.reshape(resized, [batch_size, depth, height, width, channels])

    def _get_lrr_size(
        self, width: tf.Tensor, height: tf.Tensor, angle: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute Largest Rectangle Rotation size."""
        angle = tf.abs(angle)
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)
        sin_a, cos_a = tf.sin(angle), tf.cos(angle)

        def width_limited():
            lrr_w = width / (sin_a + (width / height) * cos_a)
            lrr_h = (height / width) * lrr_w
            return lrr_w, lrr_h

        def height_limited():
            lrr_h = height / (sin_a + (height / width) * cos_a)
            lrr_w = (width / height) * lrr_h
            return lrr_w, lrr_h

        return tf.cond(width <= height, width_limited, height_limited)

    def _get_last_random_rotate_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)

from typing import Sequence

import tensorflow as tf

from ..base import RandomTransform
from ..tensor_bundle import TensorBundle
from ..utils import get_spatial_rank


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
    """Rotate a 4D ``(D, H, W, C)`` tensor slice-wise over the height-width plane."""
    img_shape = tf.shape(image)
    h, w = img_shape[1], img_shape[2]
    matrix = get_rotation_matrix(angle, h, w)
    matrices = tf.tile(tf.expand_dims(matrix, 0), [img_shape[0], 1])

    return tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=matrices,
        output_shape=[h, w],
        interpolation=interpolation,
        fill_mode="CONSTANT",
        fill_value=tf.cast(fill_value, image.dtype),
    )


class RandomRotate(RandomTransform):
    """Randomly rotate 3D volumes using slice-wise 2D projection transforms.

    ``RandomRotate`` samples an angle and rotates each depth slice in the
    height-width plane. The first key is treated like an image tensor and uses
    bilinear interpolation, while the optional second key is treated like a
    label tensor and uses nearest-neighbor interpolation.

    This transform currently supports only 3D channel-last tensors shaped
    ``(D, H, W, C)``.

    Args:
        keys: One or two keys. When two keys are provided, they are typically
            image then label.
        factor: Maximum absolute sampled rotation angle in radians.
        prob: Probability of applying the rotation.
        fill_value: Constant fill value for the primary image key when
            ``fill_mode="constant"``.
        fill_mode: Either ``"constant"`` or ``"crop"``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Randomly rotate a 3D image-label pair using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomRotate

            transform = RandomRotate(keys=["image", "label"], factor=0.2, prob=0.5)
            image = tf.random.normal((32, 64, 64, 1))
            label = tf.cast(image > 0, tf.int32)
            result = transform({"image": image, "label": label})
            output = result["image"]
            print(output.shape)

        Randomly rotate a 3D image-label pair stored in a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomRotate, TensorBundle

            transform = RandomRotate(keys=["image", "label"], factor=0.2, prob=0.5)
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
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob)
        if not isinstance(keys, (list, tuple)):
            raise TypeError("`keys` must be a list or tuple.")
        if len(keys) not in (1, 2):
            raise ValueError("`keys` must have length 1 or 2.")
        if factor < 0:
            raise ValueError(f"`factor` must be non-negative. Received {factor}.")
        if fill_mode not in {"crop", "constant"}:
            raise ValueError("fill_mode must be either 'crop' or 'constant'.")

        self.keys = tuple(keys)
        self.factor = factor
        self.fill_value = fill_value
        self.fill_mode = fill_mode
        self.allow_missing_keys = allow_missing_keys

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present_keys = []
        for key in self.keys:
            if key in bundle.data:
                present_keys.append(key)
            elif not self.allow_missing_keys:
                raise KeyError(f"Key '{key}' not found in input data.")

        if not present_keys:
            return bundle

        sample_tensor = bundle.data[present_keys[0]]
        spatial_rank = get_spatial_rank(sample_tensor)
        if spatial_rank != 3:
            raise ValueError(
                f"RandRotate currently supports only 3D tensors; got spatial rank {spatial_rank} "
                f"for shape {sample_tensor.shape}."
            )

        should_rotate = self.sample_should_apply()
        angle = tf.random.uniform([], -self.factor, self.factor)

        for key in present_keys:
            tensor = bundle.data[key]
            bundle.data[key] = tf.cond(
                should_rotate,
                lambda tensor=tensor, key=key: self.rotate_tensor(tensor, key, angle),
                lambda tensor=tensor: tensor,
            )

        self.record_random_transform(
            bundle,
            params={
                "keys": list(present_keys),
                "factor": self.factor,
                "angle": angle,
                "fill_mode": self.fill_mode,
            },
            applied=should_rotate,
            kernel="rotate_volume",
        )
        return bundle

    def rotate_tensor(self, tensor: tf.Tensor, key: str, angle: tf.Tensor) -> tf.Tensor:
        """Rotate one tensor and apply optional center crop cleanup."""
        interpolation = "BILINEAR" if key == self.keys[0] else "NEAREST"
        fill_value = self.fill_value if key == self.keys[0] else 0.0
        rotated = rotate_volume(tensor, angle, interpolation=interpolation, fill_value=fill_value)

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
        height, width = shape[1], shape[2]
        lrr_w, lrr_h = self._get_lrr_size(width, height, angle)
        crop_fraction = (
            tf.minimum(
                lrr_h / tf.cast(height, tf.float32),
                lrr_w / tf.cast(width, tf.float32),
            )
            * 0.98
        )
        method = "bilinear" if interpolation == "BILINEAR" else "nearest"
        return tf.image.resize(
            tf.image.central_crop(tensor, crop_fraction),
            [height, width],
            method=method,
        )

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

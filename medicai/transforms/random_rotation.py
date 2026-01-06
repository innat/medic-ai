from typing import Dict, Union

import tensorflow as tf

from .tensor_bundle import TensorBundle


def get_rotation_matrix(angle, h, w):
    """
    Compute a projective transform matrix for 2D rotation around the image center.

    This returns an 8-parameter affine transform compatible with
    `tf.image.transform` / `ImageProjectiveTransformV3`, rotating the image
    by `angle` radians about its spatial center (H, W).

    Args:
        angle: Rotation angle in radians (positive = counter-clockwise).
        h: Image height.
        w: Image width.

    Returns:
        A tensor of shape (8,) representing the projective transform:
        [a0, a1, a2, b0, b1, b2, 0, 0].
    """
    h = tf.cast(h, tf.float32)
    w = tf.cast(w, tf.float32)

    x0 = w / 2.0
    y0 = h / 2.0

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    # Coefficients for the affine transform
    # [a0, a1, a2, b0, b1, b2, c0, c1]
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


def rotate_volume(image, angle, interpolation="BILINEAR", fill_value=0.0):
    """
    Rotates a 4D tensor (D, H, W, 1) using projective transforms.
    Treats D as the batch dimension.
    """
    img_shape = tf.shape(image)
    h, w = img_shape[1], img_shape[2]

    # Get the 8-parameter transformation matrix
    matrix = get_rotation_matrix(angle, h, w)

    # We broadcast the same matrix to all slices in the depth
    matrices = tf.tile(tf.expand_dims(matrix, 0), [img_shape[0], 1])

    return tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=matrices,
        output_shape=[h, w],
        interpolation=interpolation,
        fill_mode="CONSTANT",
        fill_value=tf.cast(fill_value, image.dtype),
    )


class RandRotate:
    """
    Random rotation transform for 3D volumes (slice-wise 2D rotation).

    Applies the same random rotation angle to all specified keys.
    Intended for:
        - Classification: keys=["image"]
        - Segmentation: keys=["image", "label"]
    Expected volume shape: (D, H, W, C)

    Rotation is applied slice-wise with optional Largest Rectangle Rotation (LRR)
    cropping to remove black borders.

    Example:

    import tensorflow as tf
    from medicai.transforms import RandRotate

    ## Segmentation
    image = tf.random.uniform((128, 128, 128, 4))
    label = tf.random.uniform((128, 128, 128, 3))
    rand_rotate = RandRotate(
        keys=["image", "label"],
        factor=0.3,
        prob=0.6,
        fill_mode="crop"
    )
    output = rand_cutout({"image": image, "label": label})
    augmented_image, augmented_label = output["image"], output["label"]

    ## Classification
    image = tf.random.uniform((128, 128, 128, 4))
    rand_rotate = RandRotate(
        keys=["image"],
        factor=0.3,
        prob=0.6,
        fill_mode="crop"
    )
    output = rand_cutout({"image": image})
    augmented_image = output["image"]
    """

    def __init__(
        self,
        keys,
        factor=0.1,
        prob=0.8,
        fill_value=0.0,
        fill_mode="constant",
    ):
        """
        Args:
            keys:
                Either ["image"] or ["image", "label"].
            factor:
                Maximum absolute rotation angle (in radians).
            prob:
                Probability of applying rotation.
            fill_value:
                Fill value used for image rotation.
            fill_mode:
                Currently only supports "crop" or "constant".
        """

        # keys handling: must be list/tuple
        if not isinstance(keys, (list, tuple)):
            raise TypeError("`keys` must be a list or tuple.")

        if len(keys) not in (1, 2):
            raise ValueError("`keys` must have length 1 or 2.")

        if fill_mode not in ["crop", "constant"]:
            raise ValueError("Currently only fill_mode='crop' is supported. " "Or, `constant`.")

        self.keys = keys
        self.factor = factor
        self.prob = prob
        self.fill_value = fill_value
        self.fill_mode = fill_mode

    def _get_lrr_size(self, w, h, angle):
        """Compute Largest Rectangle Rotation (LRR) size."""
        angle = tf.abs(angle)
        w, h = tf.cast(w, tf.float32), tf.cast(h, tf.float32)
        sin_a, cos_a = tf.sin(angle), tf.cos(angle)

        if w <= h:
            lrr_w = w / (sin_a + (w / h) * cos_a)
            lrr_h = (h / w) * lrr_w
        else:
            lrr_h = h / (sin_a + (h / w) * cos_a)
            lrr_w = (w / h) * lrr_h

        return lrr_w, lrr_h

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:

        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        rand_val = tf.random.uniform(())

        def apply_rotate():
            rotated_data = inputs.data.copy()
            angle = tf.random.uniform([], -self.factor, self.factor)
            safety_margin = 0.98

            for key in self.keys:
                tensor = rotated_data[key]

                # Interpolation selection
                if key == self.keys[0]:
                    interp = "BILINEAR"
                    fill = self.fill_value
                else:
                    interp = "NEAREST"
                    fill = 0.0

                tensor = rotate_volume(tensor, angle, interp, fill)

                if self.fill_mode == "crop":
                    shape = tf.shape(tensor)
                    H, W = shape[1], shape[2]

                    lrr_w, lrr_h = self._get_lrr_size(W, H, angle)
                    crop_fraction = (
                        tf.minimum(
                            lrr_h / tf.cast(H, tf.float32),
                            lrr_w / tf.cast(W, tf.float32),
                        )
                        * safety_margin
                    )

                    method = "bilinear" if interp == "BILINEAR" else "nearest"
                    tensor = tf.image.resize(
                        tf.image.central_crop(tensor, crop_fraction),
                        [H, W],
                        method=method,
                    )

                rotated_data[key] = tensor

            return rotated_data

        def no_rotate():
            return inputs.data.copy()

        rotated_data = tf.cond(rand_val <= self.prob, apply_rotate, no_rotate)
        return TensorBundle(rotated_data, inputs.meta)

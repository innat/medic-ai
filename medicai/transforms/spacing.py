from medicai.utils.general import hide_warnings

hide_warnings()

import warnings
from typing import Sequence, Tuple

import tensorflow as tf

from medicai.transforms.resize import Resize

from .tensor_bundle import TensorBundle


class Spacing:
    def __init__(
        self,
        keys: Sequence[str] = ("image", "label"),
        pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        mode: Tuple[str, str] = ("bilinear", "nearest"),
    ):
        self.pixdim = pixdim
        self.keys = keys
        self.image_mode = mode[0]
        self.label_mode = mode[1]
        self.mode = dict(zip(keys, mode))

    def get_spacing_from_affine(self, affine):
        width_spacing = tf.norm(affine[:3, 0])
        depth_spacing = tf.norm(affine[:3, 1])
        height_spacing = tf.norm(affine[:3, 2])
        return (width_spacing, depth_spacing, height_spacing)

    def __call__(self, inputs: TensorBundle) -> TensorBundle:
        for key in self.keys:
            if key not in inputs.data:
                continue

            image = inputs.data[key]
            affine = inputs.meta.get("affine", None)

            if affine is not None:
                original_spacing = self.get_spacing_from_affine(affine)
            else:
                original_spacing = (1.0, 1.0, 1.0)
                warnings.warn(
                    "Affine matrix is not provided. Using default spacing (1.0, 1.0, 1.0)."
                )

            resample_image = self.spacingd_resample(
                image, original_spacing, self.pixdim, mode=self.mode.get(key)
            )
            inputs.data[key] = resample_image
        return inputs

    def spacingd_resample(
        self,
        image: tf.Tensor,
        original_spacing: Tuple[float, float, float],
        desired_spacing: Tuple[float, float, float],
        mode: str = "bilinear",
    ) -> tf.Tensor:
        scale_d = original_spacing[0] / desired_spacing[0]
        scale_h = original_spacing[1] / desired_spacing[1]
        scale_w = original_spacing[2] / desired_spacing[2]

        original_shape = tf.shape(image)
        original_depth = tf.cast(original_shape[0], tf.float32)
        original_height = tf.cast(original_shape[1], tf.float32)
        original_width = tf.cast(original_shape[2], tf.float32)

        new_depth = tf.cast(original_depth * scale_d, tf.int32)
        new_height = tf.cast(original_height * scale_h, tf.int32)
        new_width = tf.cast(original_width * scale_w, tf.int32)

        spatial_shape = (new_depth, new_height, new_width)
        inputs = TensorBundle({"image": image})
        resized_dhw = Resize(keys=["image"], mode=[mode], spatial_shape=spatial_shape)(inputs).data[
            "image"
        ]
        return resized_dhw

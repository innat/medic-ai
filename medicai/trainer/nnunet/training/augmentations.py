from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import keras
import tensorflow as tf

from medicai.transforms import RandCropByPosNegLabel, RandFlip, RandRotate


@dataclass
class AugmentationConfig:
    """Probability gates and parameters for transforms."""

    p_rotation: float = 0.2
    p_scale: float = 0.2
    p_elastic: float = 0.2
    p_gamma: float = 0.3
    p_noise: float = 0.1
    p_mirror: float = 0.5
    rotation_angle_range: float = 0.26  # ~15 degrees in rads
    scale_range: Tuple[float, float] = (0.85, 1.15)
    elastic_alpha: float = 300.0
    elastic_sigma: float = 14.0
    gamma_range: Tuple[float, float] = (0.7, 1.5)
    noise_variance: Tuple[float, float] = (0.0, 0.1)
    mirror_axes: Tuple[int, ...] = (0, 1, 2)


class AugmentationPipeline:
    """
    Applies a configurable chain of augmentations via `medicai.transforms`.
    Designed to operate natively on GPU/TPU with TensorFlow.
    """

    def __init__(
        self,
        config: AugmentationConfig,
        patch_size: Optional[Sequence[int]] = None,
        use_fg_oversampling: bool = True,
        seed: Optional[int] = None,
    ):
        self.config = config
        self.patch_size = patch_size

        c = self.config
        keys = ["image", "label"]

        if use_fg_oversampling and patch_size is not None:
            # 1:2 ratio means 1/3 foreground patches
            self.crop = RandCropByPosNegLabel(
                keys=keys, spatial_size=tuple(patch_size), pos=1, neg=2
            )
        else:
            from medicai.transforms import RandSpatialCrop

            self.crop = RandSpatialCrop(keys=keys, roi_size=patch_size)

        self.flip = RandFlip(keys=keys, prob=c.p_mirror, spatial_axis=[0])
        self.flip2 = RandFlip(keys=keys, prob=c.p_mirror, spatial_axis=[1])
        self.flip3 = RandFlip(keys=keys, prob=c.p_mirror, spatial_axis=[2])

        self.rotate = RandRotate(
            keys=keys, factor=c.rotation_angle_range, prob=c.p_rotation, fill_mode="constant"
        )

    def __call__(
        self,
        image: tf.Tensor,
        label: Optional[tf.Tensor] = None,
        patch_size: Optional[Sequence[int]] = None,
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:

        # Determine 2D vs 3D based on number of spatial axes.
        # image shape: [D, H, W, C] (3D) or [H, W, C] (2D)
        is_3d = len(image.shape) == 4

        tensor_dict = {"image": tf.convert_to_tensor(image, dtype=tf.float32)}
        if label is not None:
            tensor_dict["label"] = tf.convert_to_tensor(label, dtype=tf.float32)

        # 1. Spatial Crop
        if hasattr(self.crop, "roi_size") and patch_size is not None:
            roi_size = list(patch_size)
            if is_3d:
                self.crop.roi_size = roi_size
            else:
                self.crop.roi_size = roi_size[-2:]
        elif hasattr(self.crop, "spatial_size") and patch_size is not None:
            self.crop.spatial_size = tuple(patch_size)

        tensor_dict = self.crop(tensor_dict).data

        # 2. Medicai Flips
        tensor_dict = self.flip(tensor_dict).data
        tensor_dict = self.flip2(tensor_dict).data
        if is_3d:
            tensor_dict = self.flip3(tensor_dict).data

        # 3. Medicai Rotation
        # Note: RandRotate is written for 3D slicing, fallback to Keras layers for 2D
        if is_3d:
            tensor_dict = self.rotate(tensor_dict).data
        else:
            # Placeholder for 2D custom rotation utilizing Keras CV / core ops layer
            rotated = keras.layers.RandomRotation(self.config.rotation_angle_range)(
                tf.expand_dims(tensor_dict["image"], 0)
            )
            tensor_dict["image"] = tf.squeeze(rotated, 0)
            if label is not None:
                rotated_lbl = keras.layers.RandomRotation(
                    self.config.rotation_angle_range, interpolation="nearest"
                )(tf.expand_dims(tensor_dict["label"], 0))
                tensor_dict["label"] = tf.squeeze(rotated_lbl, 0)

        img_out = tensor_dict["image"]

        if label is not None:
            # Nearest neighbor casts backward
            lbl_out = tf.cast(tf.math.round(tensor_dict["label"]), tf.int64)
            return img_out, lbl_out

        return img_out, None

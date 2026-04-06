from dataclasses import dataclass

import keras
from keras import ops

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
    scale_range = (0.85, 1.15)
    elastic_alpha: float = 300.0
    elastic_sigma: float = 14.0
    gamma_range = (0.7, 1.5)
    noise_variance = (0.0, 0.1)
    mirror_axes = (0, 1, 2)


class AugmentationPipeline:
    """
    Applies a configurable chain of augmentations via ``medicai.transforms``.
    Backend-agnostic: uses ``keras.ops`` instead of raw TF calls.
    """

    def __init__(
        self,
        config=None,
        patch_size=None,
        use_fg_oversampling=True,
        seed=None,
    ):
        if config is None:
            config = AugmentationConfig()
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
        image,
        label=None,
        patch_size=None,
    ):

        # Determine 2D vs 3D based on number of spatial axes.
        # image shape: [D, H, W, C] (3D) or [H, W, C] (2D)
        is_3d = len(image.shape) == 4

        # Convert to backend tensors
        tensor_dict = {"image": ops.convert_to_tensor(image, dtype="float32")}
        if label is not None:
            tensor_dict["label"] = ops.convert_to_tensor(label, dtype="float32")

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

        # 2. Random Flips
        tensor_dict = self.flip(tensor_dict).data
        tensor_dict = self.flip2(tensor_dict).data
        if is_3d:
            tensor_dict = self.flip3(tensor_dict).data

        # 3. Random Rotation
        if is_3d:
            tensor_dict = self.rotate(tensor_dict).data
        else:
            # 2D rotation using the same RandRotate with consistent
            # randomness for both image and label
            tensor_dict = self.rotate(tensor_dict).data

        img_out = tensor_dict["image"]

        if label is not None:
            # Nearest neighbor cast backward
            lbl_out = ops.cast(ops.round(tensor_dict["label"]), "int64")
            return img_out, lbl_out

        return img_out, None

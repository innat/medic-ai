from dataclasses import dataclass

import keras
from keras import ops

from medicai.transforms import RandomFlip, RandomRotate


@dataclass
class AugmentationConfig:
    """Probability gates and parameters for transforms."""

    p_rotation: float = 0.2
    # TODO: Wire the following configured augmentations into AugmentationPipeline.
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
    # TODO: Wire spatial scaling/elastic deformation and add intensity scaling,
    # gamma, Gaussian noise/blur, contrast, and simulated low-resolution transforms.
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
        seed=None,
    ):
        if config is None:
            config = AugmentationConfig()
        self.config = config
        self.patch_size = patch_size

        c = self.config
        keys = ["image", "label"]

        spatial_rank = len(patch_size) if patch_size is not None else 3
        if spatial_rank not in (2, 3):
            raise ValueError("patch_size must describe either a 2D or 3D spatial patch.")
        self.input_layout = "HWC" if spatial_rank == 2 else "DHWC"

        self.flip = RandomFlip(
            keys=keys, prob=c.p_mirror, spatial_axis=0, input_layout=self.input_layout
        )
        self.flip2 = RandomFlip(
            keys=keys, prob=c.p_mirror, spatial_axis=1, input_layout=self.input_layout
        )
        self.flip3 = (
            RandomFlip(keys=keys, prob=c.p_mirror, spatial_axis=2, input_layout="DHWC")
            if spatial_rank == 3
            else None
        )

        self.rotate = RandomRotate(
            keys=keys,
            factor=c.rotation_angle_range,
            prob=c.p_rotation,
            fill_mode="constant",
            input_layout=self.input_layout,
        )

    def __call__(
        self,
        image,
        label=None,
        patch_size=None,
        label_is_regions=False,
    ):

        # Convert to backend tensors
        tensor_dict = {"image": ops.convert_to_tensor(image, dtype="float32")}
        if label is not None:
            tensor_dict["label"] = ops.convert_to_tensor(label, dtype="float32")

        # 2. Random Flips
        tensor_dict = self.flip(tensor_dict).data
        tensor_dict = self.flip2(tensor_dict).data
        if self.flip3 is not None:
            tensor_dict = self.flip3(tensor_dict).data

        # 3. Random Rotation
        tensor_dict = self.rotate(tensor_dict).data

        img_out = tensor_dict["image"]

        if label is not None:
            # Nearest neighbor cast backward
            label_dtype = "float32" if label_is_regions else "int64"
            lbl_out = ops.cast(ops.round(tensor_dict["label"]), label_dtype)
            return img_out, lbl_out

        return img_out, None

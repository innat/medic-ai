
from .deeplabv3_3d import DeepLabV3Plus3D

from medicai.utils.model_utils import BACKBONE_ARGS


def DeepLabV3Plus2D(variant="densenet121", **kwargs):
    pass


def DeepLabV3(*, input_shape, variant="densenet121", num_classes=1000, **kwargs):

    ndim = len(input_shape) - 1  # exclude channel dim

    if variant not in BACKBONE_ARGS:
        raise ValueError(f"Unknown variant {variant}. Must be one of {list(BACKBONE_ARGS.keys())}")

    if ndim == 2:
        ...
    elif ndim == 3:
        return DeepLabV3Plus3D(input_shape=input_shape, variant=variant, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unsupported input shape: {input_shape}")



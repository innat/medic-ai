from keras.applications import DenseNet121, DenseNet169, DenseNet201

from medicai.utils.model_utils import BACKBONE_ARGS, BACKBONE_ZOO, KERAS_APPLICATION

from .densenet_3d import DenseNet3D


def DenseNet2D(variant="densenet121", **kwargs):
    variant = variant.lower()
    model_fn = KERAS_APPLICATION.get(variant)
    if model_fn is None:
        raise ValueError(
            f"Unknown DenseNet variant: {variant}. Choose from {list(KERAS_APPLICATION.keys())}"
        )
    return model_fn(**kwargs)


def DenseNet(*, input_shape, variant="densenet121", **kwargs):
    ndim = len(input_shape) - 1  # exclude channel dim

    if variant not in BACKBONE_ZOO:
        raise ValueError(f"Unknown variant {variant}. Must be one of {list(BACKBONE_ZOO.keys())}")

    blocks = BACKBONE_ARGS[variant]

    if ndim == 2:
        return DenseNet2D(input_shape=input_shape, **kwargs)
    elif ndim == 3:
        return DenseNet3D(input_shape=input_shape, blocks=blocks, **kwargs)
    else:
        raise ValueError(f"Unsupported input shape: {input_shape}")


BACKBONE_ZOO["densenet121"] = DenseNet
BACKBONE_ZOO["densenet169"] = DenseNet
BACKBONE_ZOO["densenet201"] = DenseNet

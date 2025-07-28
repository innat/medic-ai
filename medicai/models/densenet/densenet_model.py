from keras.applications import DenseNet121, DenseNet169, DenseNet201

from .densenet_3d import DenseNet3D


def DenseNet2D(variant="densenet121", **kwargs):
    variant = variant.lower()

    variant_map = {
        "densenet121": DenseNet121,
        "densenet169": DenseNet169,
        "densenet201": DenseNet201,
    }

    model_fn = variant_map.get(variant)
    if model_fn is None:
        raise ValueError(
            f"Unknown DenseNet variant: {variant}. Choose from {list(variant_map.keys())}"
        )
    return model_fn(**kwargs)


def DenseNet(*, input_shape, variant="densenet121", **kwargs):
    ndim = len(input_shape) - 1  # exclude channel dim

    if variant not in BACKBONE_ARGS:
        raise ValueError(f"Unknown variant {variant}. Must be one of {list(BACKBONE_ARGS.keys())}")

    blocks = BACKBONE_ARGS[variant]

    if ndim == 2:
        return DenseNet2D(input_shape=input_shape, **kwargs)
    elif ndim == 3:
        return DenseNet3D(input_shape=input_shape, blocks=blocks, **kwargs)
    else:
        raise ValueError(f"Unsupported input shape: {input_shape}")


BACKBONE_ZOO = {
    "densenet121": DenseNet,
    "densenet169": DenseNet,
    "densenet201": DenseNet,
}
BACKBONE_ARGS = {
    "densenet121": [309, 137, 49, 3],
    "densenet169": [365, 137, 49, 3],
    "densenet201": [477, 137, 49, 3],
}

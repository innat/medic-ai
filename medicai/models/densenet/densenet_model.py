from keras.applications import DenseNet121, DenseNet169, DenseNet201

from medicai.utils.model_utils import BACKBONE_ARGS, BACKBONE_ZOO, KERAS_APPLICATION

from .densenet_3d import DenseNet3D


def DenseNet2D(variant="densenet121", **kwargs):
    """
    Returns a 2D DenseNet model from Keras Applications.

    Args:
        variant (str): Name of the DenseNet variant (e.g., "densenet121").
        **kwargs: Arguments passed to `keras.applications.DenseNet*`.

    Returns:
        keras.Model: A 2D DenseNet model.

    Raises:
        ValueError: If the variant name is not supported.
    """
    variant = variant.lower()
    model_fn = KERAS_APPLICATION.get(variant)
    if model_fn is None:
        raise ValueError(
            f"Unknown DenseNet variant: {variant}. Choose from {list(KERAS_APPLICATION.keys())}"
        )
    return model_fn(**kwargs)


def DenseNet(*, input_shape, variant="densenet121", num_classes=1000, **kwargs):
    """
    Creates a 2D or 3D DenseNet model based on input shape dimensionality.

    Args:
        input_shape (tuple): Input tensor shape excluding batch size.
            Example: (height, width, channels) for 2D,
            (depth, height, width, channels) for 3D.
        variant (str): DenseNet variant to use. Must be one of:
            "densenet121", "densenet169", "densenet201".
        num_classes (int): Number of output classes for classification head.
            For 2D DenseNet, this is mapped to 'classes' argument.
        **kwargs: Additional keyword arguments passed to the underlying
            DenseNet constructor.

    Returns:
        keras.Model: Constructed 2D or 3D DenseNet model without top layer.

    Raises:
        ValueError: If variant is unknown or input shape dimensionality
            is unsupported.
    """
    ndim = len(input_shape) - 1  # exclude channel dim

    if variant not in BACKBONE_ARGS:
        raise ValueError(f"Unknown variant {variant}. Must be one of {list(BACKBONE_ARGS.keys())}")

    blocks = BACKBONE_ARGS[variant]

    if ndim == 2:
        kwargs["classes"] = num_classes
        kwargs.pop("num_classes", None)
        return DenseNet2D(input_shape=input_shape, **kwargs)
    elif ndim == 3:
        return DenseNet3D(input_shape=input_shape, blocks=blocks, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unsupported input shape: {input_shape}")


BACKBONE_ZOO["densenet121"] = DenseNet
BACKBONE_ZOO["densenet169"] = DenseNet
BACKBONE_ZOO["densenet201"] = DenseNet

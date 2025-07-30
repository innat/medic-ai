from keras import Model, layers

from medicai.blocks.unet_imagenet_decoder import UNetDecoder
from medicai.utils.model_utils import BACKBONE_ZOO, KERAS_APPLICATION, SKIP_CONNECTION_ARGS


def get_unet_backbone(variant: str, input_shape, dim: int, **kwargs):
    """
    Retrieves a UNet-compatible backbone model, either 2D or 3D.

    Args:
        variant (str): Name of the backbone variant (e.g., "densenet121").
        input_shape (tuple): Shape of the input tensor excluding batch size.
        dim (int): Dimensionality of the model (2 for 2D, 3 for 3D).
        **kwargs: Additional arguments passed to the backbone constructor.

    Returns:
        keras.Model: Backbone model with feature extraction layers.

    Raises:
        ValueError: If the specified variant is not found in the corresponding registry.
    """
    if dim == 3:
        # 3D Model
        if variant not in BACKBONE_ZOO:
            raise ValueError(f"3D variant '{variant}' not found.")
        return BACKBONE_ZOO[variant](
            input_shape=input_shape,
            variant=variant,
            include_top=False,
            **kwargs,
        )
    else:
        # 2D Keras applications
        if variant not in KERAS_APPLICATION:
            raise ValueError(f"2D variant '{variant}' not found.")
        return KERAS_APPLICATION[variant](
            input_shape=input_shape,
            include_top=False,
            weights=None,
            **kwargs,
        )


def UNet(
    variant,
    num_classes,
    input_shape=(None, None, None, 1),
    weights=None,
    classifier_activation=None,
    decoder_block_type="upsampling",
    decoder_filters=(256, 128, 64, 32, 16),
):
    """
    Builds a flexible UNet architecture with either 2D or 3D backbones.

    Args:
        variant (str): Name of the backbone variant (e.g., "densenet121").
        num_classes (int): Number of output classes.
        input_shape (tuple): Shape of the input tensor, including channels.
        weights (str, optional): Pretrained weights path or identifier (currently unused).
        classifier_activation (str): Activation function for final layer (e.g., "sigmoid", "softmax").
        decoder_block_type (str): Type of decoder block ("upsampling" or "transpose").
        decoder_filters (tuple): Number of filters in each decoder block layer.

    Returns:
        keras.Model: A compiled UNet model (2D or 3D).

    Raises:
        ValueError: If input shape is not compatible with either 2D or 3D model.
    """
    dim = 3 if len(input_shape) == 4 else 2
    ConvFinal = layers.Conv3D if dim == 3 else layers.Conv2D

    # Load backbone
    base_model = get_unet_backbone(
        variant=variant,
        input_shape=input_shape,
        dim=dim,
    )

    inputs = base_model.input

    # Get skip connections
    selected_layers = SKIP_CONNECTION_ARGS[variant]
    skip_layers = [
        (
            base_model.get_layer(name=i).output
            if isinstance(i, str)
            else base_model.get_layer(index=i).output
        )
        for i in selected_layers
    ]

    # Apply decoder
    x = base_model.output
    decoder = UNetDecoder(skip_layers, decoder_filters, dim, block_type=decoder_block_type)
    x = decoder(x)

    # Final segmentation head
    x = ConvFinal(num_classes, kernel_size=1, padding="same")(x)
    outputs = layers.Activation(classifier_activation, dtype="float32")(x)

    # built unet
    model = Model(inputs=inputs, outputs=outputs, name=f"UNet{dim}D_{variant}")

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model

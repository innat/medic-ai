from keras import Model, layers

from medicai.layers.decoder import UNetDecoder
from medicai.utils.model_utils import BACKBONE_ZOO, KERAS_APPLICATION, SKIP_CONNECTION_ARGS


def get_unet_backbone(variant: str, input_shape, dim: int, **kwargs):
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


# Main UNet method
def UNet(
    variant,
    num_classes,
    input_shape=(None, None, None, 1),
    input_tensor=None,
    weights=None,
    activation="sigmoid",
    decoder_block_type="upsampling",
    decoder_filters=(256, 128, 64, 32, 16),
):
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
    outputs = layers.Activation(activation, dtype="float32")(x)
    return Model(inputs=inputs, outputs=outputs, name=f"UNet{dim}D_{variant}")

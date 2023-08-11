from tensorflow import keras
from tensorflow.keras import losses, metrics
from tensorflow.keras import layers as nn


from medic.utils.model_utils import BACKBONE, BACKBONE_ARGS
from medic.layers.conv import UpsampleBlock2D


def UNet2D(config):
    input_size = config.dataset.image_size
    backbone = config.model.backbone
    decoder_filters = config.model.decoder_filters
    num_classes = config.dataset.num_classes
    activation = config.dataset.cls_act

    inputs = keras.Input(shape=(input_size, input_size, 3))

    base_model = BACKBONE[backbone](weights=None, include_top=False, input_tensor=inputs)
    selected_layers = BACKBONE_ARGS[backbone]
    skip_layers = [
        base_model.get_layer(name=sl).output
        if isinstance(sl, str)
        else base_model.get_layer(index=sl).output
        for sl in selected_layers
    ]

    # Start Upsampling
    x = base_model.output
    for i in range(len(decoder_filters)):
        if i < len(skip_layers):
            skip = skip_layers[i]
        else:
            skip = None
        x = UpsampleBlock2D(decoder_filters[i])(x, skip)

    # Final layer
    x = nn.Conv2D(filters=num_classes, kernel_size=(3, 3), padding="same")(x)
    final = nn.Activation(activation, dtype="float32")(x)
    model = keras.Model(inputs=inputs, outputs=final, name=f"UNet")
    return model



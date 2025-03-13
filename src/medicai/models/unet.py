from tensorflow import keras
from tensorflow.keras import layers as nn

from medicai.layers.conv import UpsampleBlock2D
from medicai.utils.model_utils import BACKBONE, BACKBONE_ARGS


def UNet2D(
    backbone,
    input_size,
    num_classes,
    class_activation,
    backbone_weight=None,
    freeze_backbone=False,
    decoder_filters=[256, 128, 64, 32, 16],
):
    inputs = keras.Input(shape=(input_size, input_size, 3))
    base_model = BACKBONE[backbone](weights=backbone_weight, include_top=False, input_tensor=inputs)
    base_model.trainable = freeze_backbone
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
    final = nn.Activation(class_activation, dtype="float32")(x)
    model = keras.Model(inputs=inputs, outputs=final, name=f"UNet")
    return model

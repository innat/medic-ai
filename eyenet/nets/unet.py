from tensorflow import keras
from tensorflow.keras import layers as nn
from tensorflow.keras import applications


BACKBONE = {
    "efficientnetb0": applications.EfficientNetB0,
    "resnet50": applications.ResNet50,
    "densenet121": applications.DenseNet121,
    "convnextsmall": applications.ConvNeXtSmall,
}

BACKBONE_ARGS = {
    "efficientnetb0": [
        "block6a_expand_activation",
        "block4a_expand_activation",
        "block3a_expand_activation",
        "block2a_expand_activation",
    ],
    "resnet50": ["conv4_block6_2_relu", "conv3_block4_2_relu", "conv2_block3_2_relu", "conv1_relu"],
    "densenet121": [311, 139, 51, 4],
    "convnextsmall": [268, 51, 26],
}


def Conv3x3BNReLU(filters):
    def apply(input):
        x = nn.Conv2D(
            filters,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )(input)
        x = nn.BatchNormalization()(x)
        x = nn.ReLU()(x)
        return x

    return apply


def UpsampleBlock(filters):
    def apply(x, skip=None):
        x = nn.UpSampling2D((2, 2))(x)
        x = nn.Concatenate(axis=3)([skip, x]) if skip is not None else x
        x = Conv3x3BNReLU(filters)(x)
        x = Conv3x3BNReLU(filters)(x)
        return x

    return apply


def UNet(backbone, input_size, num_classes, activation, decoder_filters=[256, 128, 64, 32, 16]):
    inputs = keras.Input(input_size)

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
        x = UpsampleBlock(decoder_filters[i])(x, skip)

    # Final layer
    x = nn.Conv2D(filters=num_classes, kernel_size=(3, 3), padding="same")(x)
    final = nn.Activation(activation, dtype="float32")(x)
    model = keras.Model(inputs=inputs, outputs=final, name=f"UNet[{backbone}]")

    return model

from keras import layers


def Conv3x3BNReLU(filters):
    def apply(input):
        x = layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )(input)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    return apply


def UpsampleBlock(filters):
    def apply(x, skip=None):
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate(axis=3)([skip, x]) if skip is not None else x
        x = Conv3x3BNReLU(filters)(x)
        x = Conv3x3BNReLU(filters)(x)
        return x

    return apply

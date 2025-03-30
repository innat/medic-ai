from keras import layers


def UnetOutBlock(num_classes, dropout_rate=None, activation=None):
    def wrapper(inputs):
        x = layers.Conv3D(
            num_classes, kernel_size=1, strides=1, use_bias=True, activation=activation
        )(inputs)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)
        return x

    return wrapper

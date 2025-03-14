
from keras import layers

def UnetOutBlock(out_channels, dropout_rate=None):
    def wrapper(inputs):
        x = layers.Conv3D(out_channels, kernel_size=1, strides=1, use_bias=True)(inputs)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)
        return x
    return wrapper
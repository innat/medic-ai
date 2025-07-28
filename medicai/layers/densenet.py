from keras import layers


def DenseLayer3D(x, growth_rate, bn_size, dropout_rate):
    """A single 3D dense layer (BN -> ReLU -> Conv3D(1x1) -> BN -> ReLU -> Conv3D(3x3))"""
    out = layers.BatchNormalization()(x)
    out = layers.Activation("relu")(out)
    out = layers.Conv3D(bn_size * growth_rate, kernel_size=1, use_bias=False)(out)

    out = layers.BatchNormalization()(out)
    out = layers.Activation("relu")(out)
    out = layers.Conv3D(growth_rate, kernel_size=3, padding="same", use_bias=False)(out)

    if dropout_rate > 0.0:
        out = layers.Dropout(dropout_rate)(out)

    return layers.Concatenate(axis=-1)([x, out])


def DenseBlock3D(x, num_layers, growth_rate, bn_size, dropout_rate):
    """A 3D dense block made of multiple DenseLayer3D"""
    for _ in range(num_layers):
        x = DenseLayer3D(x, growth_rate, bn_size, dropout_rate)
    return x


def TransitionLayer3D(x, out_channels):
    """A transition layer with 1x1 conv + avg pool"""
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv3D(out_channels, kernel_size=1, use_bias=False)(x)
    x = layers.AveragePooling3D(pool_size=2, strides=2)(x)
    return x

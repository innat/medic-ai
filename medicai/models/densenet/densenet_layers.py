from keras import layers

from ...utils import get_conv_layer, get_pooling_layer


def DenseLayer(x, growth_rate, bn_size, dropout_rate):
    """A single dense layer (BN -> ReLU -> Conv3D(1x1) -> BN -> ReLU -> Conv3D(3x3))"""
    out = layers.BatchNormalization()(x)
    out = layers.Activation("relu")(out)
    out = get_conv_layer(
        spatial_dims=len(x.shape) - 2,
        layer_type="conv",
        filters=bn_size * growth_rate,
        kernel_size=1,
        use_bias=False,
    )(out)

    out = layers.BatchNormalization()(out)
    out = layers.Activation("relu")(out)
    out = get_conv_layer(
        spatial_dims=len(x.shape) - 2,
        layer_type="conv",
        filters=growth_rate,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(out)

    if dropout_rate > 0.0:
        out = layers.Dropout(dropout_rate)(out)

    return layers.Concatenate(axis=-1)([x, out])


def DenseBlock(x, num_layers, growth_rate, bn_size, dropout_rate):
    """A 3D dense block made of multiple DenseLayer"""
    for _ in range(num_layers):
        x = DenseLayer(x, growth_rate, bn_size, dropout_rate)
    return x


def TransitionLayer(x, out_channels):
    """A transition layer with 1x1 conv + avg pool"""
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = get_conv_layer(
        spatial_dims=len(x.shape) - 2,
        layer_type="conv",
        filters=out_channels,
        kernel_size=1,
        use_bias=False,
    )(x)
    x = get_pooling_layer(spatial_dims=len(x.shape) - 2, layer_type="avg", pool_size=2, strides=2)(
        x
    )
    return x

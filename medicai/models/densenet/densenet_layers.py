from keras import layers

from ...utils import get_conv_layer, get_pooling_layer


def apply_dense_layer(x, growth_rate, bn_size, dropout_rate, block_idx, layer_idx):
    """A single dense layer (BN -> ReLU -> Conv3D(1x1) -> BN -> ReLU -> Conv3D(3x3))"""
    out = layers.BatchNormalization(name=f"block{block_idx}_layer{layer_idx}_bn1")(x)
    out = layers.Activation("relu", name=f"block{block_idx}_layer{layer_idx}_relu1")(out)
    out = get_conv_layer(
        spatial_dims=len(x.shape) - 2,
        layer_type="conv",
        filters=bn_size * growth_rate,
        kernel_size=1,
        use_bias=False,
        name=f"block{block_idx}_layer{layer_idx}_conv1x1",
    )(out)

    out = layers.BatchNormalization(name=f"block{block_idx}_layer{layer_idx}_bn2")(out)
    out = layers.Activation("relu", name=f"block{block_idx}_layer{layer_idx}_relu2")(out)
    out = get_conv_layer(
        spatial_dims=len(x.shape) - 2,
        layer_type="conv",
        filters=growth_rate,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name=f"block{block_idx}_layer{layer_idx}_conv3x3",
    )(out)

    if dropout_rate > 0.0:
        out = layers.Dropout(dropout_rate, name=f"block{block_idx}_layer{layer_idx}_dropout")(out)

    return layers.Concatenate(axis=-1, name=f"block{block_idx}_layer{layer_idx}_concat")([x, out])


def apply_dense_block(x, num_layers, growth_rate, bn_size, dropout_rate, block_idx):
    """A 3D dense block made of multiple DenseLayer"""
    for layer_idx in range(num_layers):
        x = apply_dense_layer(x, growth_rate, bn_size, dropout_rate, block_idx, layer_idx)
    return x


def apply_transition_layer(x, out_channels, block_idx):
    x = layers.BatchNormalization(name=f"block{block_idx}_trans_bn")(x)
    x = layers.Activation("relu", name=f"block{block_idx}_trans_relu")(x)
    x = get_conv_layer(
        spatial_dims=len(x.shape) - 2,
        layer_type="conv",
        filters=out_channels,
        kernel_size=1,
        use_bias=False,
        name=f"block{block_idx}_trans_conv1x1",
    )(x)
    x = get_pooling_layer(
        spatial_dims=len(x.shape) - 2,
        layer_type="avg",
        pool_size=2,
        strides=2,
        name=f"block{block_idx}_trans_pool",
    )(x)
    return x

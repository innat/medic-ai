from keras import layers

from medicai.utils import get_act_layer, get_norm_layer


def UnetBasicBlock(out_channels, kernel_size=3, stride=1, norm_name="instance", dropout_rate=None):
    def apply(inputs):
        x = layers.Conv3D(out_channels, kernel_size, strides=stride, use_bias=False)(inputs)
        x = get_norm_layer(norm_name)(x)
        x = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))(x)

        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        x = layers.Conv3D(out_channels, kernel_size, strides=1, use_bias=False)(x)
        x = get_norm_layer(norm_name)(x)
        x = get_act_layer(("leakyrelu", {"inplace": True, "negative_slope": 0.01}))(x)

        return x

    return apply

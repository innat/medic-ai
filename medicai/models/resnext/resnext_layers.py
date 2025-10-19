from keras import layers

from medicai.utils import get_act_layer, get_conv_layer, get_norm_layer, get_reshaping_layer


def conv_block(
    filters, strides, groups=32, normalization="batch", activation="relu", name="conv_block"
):
    """ResNeXt convolutional block with bottleneck and shortcut connection.

    Args:
        filters: Integer, number of output filters in the first convolution
        strides: Integer, stride length for the spatial convolution
        groups: Integer, number of groups for grouped convolution (cardinality)
        normalization: String, type of normalization to use
        activation: String, activation function to use
        name: String, base name for the layers
    """

    def apply(inputs):
        spatial_dims = len(inputs.shape) - 2

        # First 1x1 convolution
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=False,
            name=f"{name}_1x1_conv",
        )(inputs)
        x = get_norm_layer(
            layer_type=normalization,
            name=f"{name}_1x1_norm",
        )(x)
        x = get_act_layer(layer_type=activation, name=f"{name}_1x1_act")(x)

        # 3x3 grouped convolution with padding
        x = get_reshaping_layer(
            spatial_dims=spatial_dims,
            layer_type="padding",
            padding=1,
            name=f"{name}_padding",
        )(x)

        # Group conv
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=filters,
            kernel_size=3,
            strides=strides,
            groups=groups,
            padding="valid",
            use_bias=False,
            name=f"{name}_3x3_group_conv",
        )(x)
        x = get_norm_layer(
            layer_type=normalization,
            name=f"{name}_3x3_norm",
        )(x)
        x = get_act_layer(layer_type=activation, name=f"{name}_3x3_act")(x)

        # Final 1x1 convolution
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=2 * filters,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=False,
            name=f"{name}_1x1_final",
        )(x)
        x = get_norm_layer(
            layer_type=normalization,
            name=f"{name}_final_norm",
        )(x)

        # Shortcut connection
        shortcut = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=2 * filters,
            kernel_size=1,
            strides=strides,
            padding="valid",
            use_bias=False,
            name=f"{name}_shortcut_conv",
        )(inputs)
        shortcut = get_norm_layer(
            layer_type=normalization,
            name=f"{name}_shortcut_norm",
        )(shortcut)

        # Add and activate
        x = layers.Add(name=f"{name}_add")([x, shortcut])
        x = get_act_layer(layer_type=activation, name=f"{name}_output_act")(x)
        return x

    return apply


def identity_block(
    filters, groups=32, normalization="batch", activation="relu", name="identity_block"
):
    """ResNeXt identity block with bottleneck architecture.

    Args:
        filters: Integer, number of output filters in the first convolution
        groups: Integer, number of groups for grouped convolution (cardinality)
        normalization: String, type of normalization to use
        activation: String, activation function to use
        name: String, base name for the layers
    """

    def apply(inputs):
        spatial_dims = len(inputs.shape) - 2

        # First 1x1 convolution
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=False,
            name=f"{name}_1x1_conv",
        )(inputs)
        x = get_norm_layer(
            layer_type=normalization,
            name=f"{name}_1x1_norm",
        )(x)
        x = get_act_layer(layer_type=activation, name=f"{name}_1x1_act")(x)

        # 3x3 grouped convolution with padding
        x = get_reshaping_layer(
            spatial_dims=spatial_dims,
            layer_type="padding",
            padding=1,
            name=f"{name}_padding",
        )(x)
        # Group conv
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=filters,
            kernel_size=3,
            strides=1,
            groups=groups,
            padding="valid",
            use_bias=False,
            name=f"{name}_3x3_group_conv",
        )(x)
        x = get_norm_layer(
            layer_type=normalization,
            name=f"{name}_3x3_norm",
        )(x)
        x = get_act_layer(layer_type=activation, name=f"{name}_3x3_act")(x)

        # Final 1x1 convolution
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=2 * filters,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=False,
            name=f"{name}_1x1_final",
        )(x)
        x = get_norm_layer(
            layer_type=normalization,
            name=f"{name}_final_norm",
        )(x)

        # Add and activate (identity shortcut)
        x = layers.Add(name=f"{name}_add")([x, inputs])
        x = get_act_layer(layer_type=activation, name=f"{name}_output_act")(x)
        return x

    return apply

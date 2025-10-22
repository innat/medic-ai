from keras import layers, ops

from medicai.layers import AtrousSpatialPyramidPooling, ResizingND
from medicai.utils import get_act_layer, get_conv_layer, get_norm_layer


def DeepLabV3PlusDecoder(
    spatial_dims,
    decoder_channels=256,
    decoder_dilation_rates=(12, 24, 36),
    decoder_aspp_separable=True,
    decoder_aspp_dropout=0.0,
    decoder_normalization="batch",
    decoder_activation="relu",
    projection_filters=48,
):
    """
    Implements the DeepLabV3+ decoder path, fusing deep features with low-level features.

    This function constructs the decoder logic, which consists of:
    1. Processing the deep feature map through the Atrous Spatial Pyramid Pooling (ASPP) module.
    2. Upsampling the ASPP output to match the low-level feature resolution.
    3. Projecting the low-level feature map (usually P2, $\\frac{1}{4}$ res) using a 1x1x1 convolution.
    4. Concatenating the upsampled ASPP output and the projected low-level feature.
    5. Applying final decoder convolutions (usually separable).

    Args:
        spatial_dims (int): The number of spatial dimensions (2 for 2D, 3 for 3D).
        decoder_channels (int, optional): The channel count for ASPP output and final decoder conv.
            Defaults to 256.
        decoder_dilation_rates (tuple of ints, optional): Dilation rates for the ASPP module.
            Defaults to (12, 24, 36).
        decoder_aspp_separable (bool, optional): If True, use separable convolutions in ASPP.
            Defaults to True.
        decoder_aspp_dropout (float, optional): Dropout rate after ASPP projection.
            Defaults to 0.0.
        decoder_normalization (str, optional): Normalization type used (e.g., 'batch').
            Defaults to "batch".
        decoder_activation (str, optional): Activation function used in decoder blocks.
            Defaults to "relu".
        projection_filters (int, optional): Number of channels to project low-level features to.
            Defaults to 48.
    """

    def apply(deep_feature, low_level_feature=None):

        if low_level_feature is None:
            raise ValueError("`low_level_feature` must be provided to DeepLabV3PlusDecoder.")

        # ASPP module on deep features
        aspp_output = AtrousSpatialPyramidPooling(
            dilation_rates=decoder_dilation_rates,
            num_channels=decoder_channels,
            activation=decoder_activation,
            separable=decoder_aspp_separable,
            dropout=decoder_aspp_dropout,
            name="aspp",
        )(deep_feature)

        # Upsample ASPP output
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="separable_conv",
            filters=decoder_channels,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="aspp_projection",
        )(aspp_output)
        x = get_norm_layer(decoder_normalization, name="aspp_projection_bn")(x)
        x = get_act_layer(decoder_activation, name="aspp_projection_activation")(x)

        # Upsampling
        target_size = tuple(ops.shape(low_level_feature)[i] for i in range(1, 1 + spatial_dims))
        x = ResizingND(
            target_shape=target_size, interpolation="bilinear" if spatial_dims == 2 else "trilinear"
        )(x)

        # Project low-level features
        low_level_proj = get_conv_layer(
            spatial_dims,
            layer_type="conv",
            filters=projection_filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name="low_level_projection",
        )(low_level_feature)
        low_level_proj = get_norm_layer(decoder_normalization, name="low_level_projection_bn")(
            low_level_proj
        )
        low_level_proj = get_act_layer(decoder_activation, name="low_level_projection_activation")(
            low_level_proj
        )

        # Concatenate with upsampled ASPP features
        x = layers.Concatenate(name="decoder_concat")([x, low_level_proj])

        # Final decoder convolutions
        x = get_conv_layer(
            spatial_dims,
            layer_type="separable_conv",
            filters=decoder_channels,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="decoder_conv1",
        )(x)
        x = get_norm_layer(decoder_normalization, name="decoder_conv_bn")(x)
        x = get_act_layer(decoder_activation, name="decoder_conv_activation")(x)
        return x

    return apply

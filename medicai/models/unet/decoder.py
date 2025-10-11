from keras import layers

from medicai.layers import AttentionGate
from medicai.utils import get_act_layer, get_conv_layer, get_norm_layer, get_reshaping_layer

INVALID_BLOCK_TYPE_MSG = "Invalid block_type '{}'. Must be 'upsampling' or 'transpose'."


def Conv3x3BnReLU(spatial_dims, filters, use_batchnorm=True, name_prefix=""):
    """
    Builds a 3x3 convolutional block followed by optional BatchNormalization and ReLU activation.

    Args:
        spatial_dims (int): Dimensionality of the convolution. Use 2 for Conv2D or 3 for Conv3D.
        filters (int): Number of output filters.
        use_batchnorm (bool): Whether to include BatchNormalization after the convolution.

    Returns:
        function: A function that applies the convolutional block to an input tensor.
    """
    dim_str = f"{spatial_dims}D"

    def apply(x):
        conv_name = f"{name_prefix}_conv3x3_{dim_str}"
        bn_name = f"{name_prefix}_bn_{dim_str}"
        relu_name = f"{name_prefix}_relu"
        x = get_conv_layer(
            spatial_dims,
            layer_type="conv",
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=not use_batchnorm,
            name=conv_name,
        )(x)
        if use_batchnorm:
            x = get_norm_layer(layer_type="batch", axis=-1, name=bn_name)(x)
        x = get_act_layer(layer_type="relu", name=relu_name)(x)
        return x

    return apply


def DecoderBlock(
    spatial_dims,
    filters,
    block_type="upsampling",
    use_batchnorm=True,
    use_attention=False,
    stage_idx=0,
):
    """
    Builds a decoder block that upsamples the feature map and optionally merges with a skip connection.

    Args:
        spatial_dims (int): Dimensionality of the operation (2 for 2D, 3 for 3D).
        filters (int): Number of filters for convolutional layers.
        block_type (str): Type of upsampling — 'upsampling' (interpolation) or 'transpose' (learned).
        use_batchnorm (bool): Whether to include BatchNormalization layers.
        use_attention (bool): Whether to apply an attention gate on the skip connection.
        stage_idx (int): Index for naming the decoder stage.

    Returns:
        function: A callable that applies the decoder block to an input and optional skip tensor.
    """
    dim_str = f"{spatial_dims}D"
    stage_prefix = f"decoder_stage{stage_idx}"

    def upsample_block(x):
        return get_reshaping_layer(
            spatial_dims,
            layer_type="upsampling",
            size=2,
            name=f"{stage_prefix}_upsampling_{dim_str}",
        )(x)

    def transpose_block(x):
        x = get_conv_layer(
            spatial_dims,
            layer_type="conv_transpose",
            filters=filters,
            kernel_size=4,
            strides=2,
            padding="same",
            name=f"{stage_prefix}_transpose_conv_{dim_str}",
        )(x)
        if use_batchnorm:
            x = get_norm_layer(layer_type="batch", axis=-1, name=f"{stage_prefix}_bn_{dim_str}")(x)
        return get_act_layer(layer_type="relu", name=f"{stage_prefix}_relu")(x)

    def apply(x, skip=None):
        # Apply attention
        if skip is not None and use_attention:
            skip = AttentionGate(filters=filters, name=f"{stage_prefix}_attention_gate")([skip, x])

        # Upsample path
        if block_type == "transpose":
            x = transpose_block(x)
        elif block_type == "upsampling":
            x = upsample_block(x)
        else:
            raise ValueError(INVALID_BLOCK_TYPE_MSG.format(block_type))

        # Skip connection
        if skip is not None:
            x = layers.Concatenate(axis=-1, name=f"{stage_prefix}_concat")([x, skip])

        # Double convolution block
        for i in range(1, 3):
            x = Conv3x3BnReLU(
                spatial_dims=spatial_dims,
                filters=filters,
                use_batchnorm=use_batchnorm,
                name_prefix=f"{stage_prefix}_conv_{i}",
            )(x)
            if block_type == "transpose":
                return x

        return x

    return apply


def UNetDecoder(
    spatial_dims,
    skip_layers,
    decoder_filters,
    block_type="upsampling",
    use_attention=False,
    use_batchnorm=True,
):
    """
    Constructs the full decoder path of the UNet using a series of DecoderBlocks.

    Args:
        skip_layers (list): List of skip connection tensors from the encoder, ordered deepest to shallowest.
        decoder_filters (list or tuple): Number of filters for each decoder stage.
        dim (int): Dimensionality of the model — 2 for 2D or 3 for 3D.
        block_type (str): Decoder block type, either 'upsampling' or 'transpose'.
        use_batchnorm (bool): Whether to include BatchNormalization layers.
        use_attention (bool): Whether to apply an attention gate on the skip connection.

    Returns:
        function: A decoder function that takes the encoder output and returns the final decoded tensor.
    """

    def decoder(x):
        num_stages = len(decoder_filters)

        for i, filters in enumerate(decoder_filters):
            stage_idx = num_stages - i
            skip = skip_layers[i] if i < len(skip_layers) else None
            x = DecoderBlock(
                spatial_dims,
                filters,
                block_type,
                use_batchnorm,
                use_attention,
                stage_idx=stage_idx,
            )(x, skip)
        return x

    return decoder

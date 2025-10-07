from keras import layers, ops

from medicai.layers import AttentionGate, SpatialResize
from medicai.utils import get_conv_layer, get_reshaping_layer, resize_volumes


def Conv3x3BnReLU(spatial_dims, filters, use_batchnorm=True, name_prefix=""):
    """
    Builds a 3x3 convolutional block followed by optional BatchNormalization and ReLU activation.

    Args:
        filters (int): Number of output filters.
        dim (int): Dimensionality of the convolution. Use 2 for Conv2D or 3 for Conv3D.
        use_batchnorm (bool): Whether to include BatchNormalization after the convolution.

    Returns:
        function: A function that applies the convolutional block to an input tensor.
    """
    BatchNorm = layers.BatchNormalization
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
            x = BatchNorm(axis=-1, name=bn_name)(x)
        x = layers.Activation("relu", name=relu_name)(x)
        return x

    return apply


def DecoderBlock(
    filters,
    spatial_dims=2,
    block_type="upsampling",
    use_batchnorm=True,
    use_attention=False,
    interpolation="nearest",
    stage_idx=0,
):
    """
    Builds a decoder block that upsamples an input tensor and optionally concatenates with a skip connection.

    Args:
        filters (int): Number of filters for the convolutional layers.
        spatial_dims (int): Dimensionality of the operation (2 for 2D, 3 for 3D).
        block_type (str): Upsampling strategy — either 'upsample' (interpolation) or 'transpose' (learned).
        use_batchnorm (bool): Whether to use BatchNormalization in convolutional blocks.

    Returns:
        function: A function that applies the decoder block to a pair of input and optional skip tensors.
    """
    dim_str = f"{spatial_dims}D"
    stage_prefix = f"dec_stage{stage_idx}"

    def apply(x, skip=None):
        if block_type == "transpose":
            x = get_conv_layer(
                spatial_dims,
                layer_type="conv_transpose",
                filters=filters,
                kernel_size=4,
                strides=2,
                padding="same",
                name=f"{stage_prefix}_upconv_{dim_str}",
            )(x)
            if use_batchnorm:
                x = layers.BatchNormalization(axis=-1, name=f"{stage_prefix}_bn_{dim_str}")(x)
            x = layers.Activation("relu", name=f"{stage_prefix}_relu")(x)
        else:
            x = get_reshaping_layer(spatial_dims, layer_type="upsampling", size=2)(x)

        if skip is not None:
            # Resize skip if spatial shapes don’t match
            if x.shape[1:-1] != skip.shape[1:-1]:
                skip = SpatialResize(
                    target_shape=ops.shape(x)[1:-1],
                    interpolation=interpolation,
                    name=f"{stage_prefix}_skip_resize",
                )(skip)

            # Make Attention-UNet.
            if use_attention:
                attention_name = f"{stage_prefix}_attention_gate"
                skip = AttentionGate(filters, spatial_dims, name=attention_name)(
                    skip, x
                )  # gating signal = current decoder x
            x = layers.Concatenate(axis=-1, name=f"{stage_prefix}_concat")([x, skip])

        x = Conv3x3BnReLU(
            spatial_dims=spatial_dims,
            filters=filters,
            use_batchnorm=use_batchnorm,
            name_prefix=f"{stage_prefix}_conv_1",
        )(x)
        x = Conv3x3BnReLU(
            spatial_dims=spatial_dims,
            filters=filters,
            use_batchnorm=use_batchnorm,
            name_prefix=f"{stage_prefix}_conv_2",
        )(x)
        return x

    return apply


def UNetDecoder(
    skip_layers,
    decoder_filters,
    spatial_dims,
    block_type="upsampling",
    use_attention=False,
    use_batchnorm=True,
    interpolation="nearest",
):
    """
    Constructs the full decoder path of the UNet using a series of DecoderBlocks.

    Args:
        skip_layers (list): List of skip connection tensors from the encoder, ordered deepest to shallowest.
        decoder_filters (list or tuple): Number of filters for each decoder stage.
        dim (int): Dimensionality of the model — 2 for 2D or 3 for 3D.
        block_type (str): Decoder block type, either 'upsampling' or 'transpose'.

    Returns:
        function: A decoder function that takes the encoder output and returns the final decoded tensor.
    """

    def decoder(x):
        num_stages = len(decoder_filters)

        for i, filters in enumerate(decoder_filters):
            stage_idx = num_stages - i
            skip = skip_layers[i] if i < len(skip_layers) else None
            x = DecoderBlock(
                filters,
                spatial_dims,
                block_type,
                use_batchnorm,
                use_attention,
                interpolation,
                stage_idx=stage_idx,
            )(x, skip)
        return x

    return decoder

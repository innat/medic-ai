from keras import layers


def Conv3x3BnReLU(filters, dim=2, use_batchnorm=True):
    """
    Builds a 3x3 convolutional block followed by optional BatchNormalization and ReLU activation.

    Args:
        filters (int): Number of output filters.
        dim (int): Dimensionality of the convolution. Use 2 for Conv2D or 3 for Conv3D.
        use_batchnorm (bool): Whether to include BatchNormalization after the convolution.

    Returns:
        function: A function that applies the convolutional block to an input tensor.
    """
    Conv = layers.Conv3D if dim == 3 else layers.Conv2D
    BatchNorm = layers.BatchNormalization

    def apply(x):
        x = Conv(
            filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=not use_batchnorm,
        )(x)
        if use_batchnorm:
            x = BatchNorm()(x)
        x = layers.Activation("relu")(x)
        return x

    return apply


def DecoderBlock(filters, dim=2, block_type="upsampling", use_batchnorm=True):
    """
    Builds a decoder block that upsamples an input tensor and optionally concatenates with a skip connection.

    Args:
        filters (int): Number of filters for the convolutional layers.
        dim (int): Dimensionality of the operation (2 for 2D, 3 for 3D).
        block_type (str): Upsampling strategy — either 'upsample' (interpolation) or 'transpose' (learned).
        use_batchnorm (bool): Whether to use BatchNormalization in convolutional blocks.

    Returns:
        function: A function that applies the decoder block to a pair of input and optional skip tensors.
    """
    Transpose = layers.Conv3DTranspose if dim == 3 else layers.Conv2DTranspose
    UpSampling = layers.UpSampling3D if dim == 3 else layers.UpSampling2D

    def apply(x, skip=None):
        if block_type == "transpose":
            x = Transpose(filters, kernel_size=4, strides=2, padding="same")(x)
        else:
            x = UpSampling(size=2)(x)

        if skip is not None:
            x = layers.Concatenate(axis=-1)([x, skip])

        x = Conv3x3BnReLU(filters, dim=dim, use_batchnorm=use_batchnorm)(x)
        x = Conv3x3BnReLU(filters, dim=dim, use_batchnorm=use_batchnorm)(x)
        return x

    return apply


def UNetDecoder(skip_layers, decoder_filters, dim, block_type="upsampling"):
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
        for i, filters in enumerate(decoder_filters):
            skip = skip_layers[i] if i < len(skip_layers) else None
            x = DecoderBlock(filters, dim, block_type)(x, skip)
        return x

    return decoder

from keras import layers


def Conv3x3BnReLU(filters, dim=2, use_batchnorm=True):
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


def DecoderBlock(filters, dim=2, block_type="upsample", use_batchnorm=True, stage=None):
    """Decoder block supporting both upsampling and transpose mode with optional skip connection"""
    Conv = layers.Conv3D if dim == 3 else layers.Conv2D
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
    def decoder(x):
        for i, filters in enumerate(decoder_filters):
            skip = skip_layers[i] if i < len(skip_layers) else None
            x = DecoderBlock(filters, dim, block_type)(x, skip)
        return x

    return decoder

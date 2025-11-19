import keras
from keras import layers

from medicai.models.unetr_plus_plus.encoder_layers import UNETRPlusPlusTransformer
from medicai.utils import get_conv_layer

from .unet import UNetResBlock


def UNETRPlusPlusUpBlock(
    spatial_dims,
    out_channels,
    kernel_size=3,
    upsample_kernel_size=2,
    norm_name="instance",
    proj_size=64,
    num_heads=4,
    sequence_length=0,
    depth=3,
    conv_decoder=False,
    dropout_rate=0.1,
):
    stride = upsample_kernel_size

    def wrapper(inputs, skip):
        x = get_conv_layer(
            spatial_dims,
            layer_type="conv_transpose",
            filters=out_channels,
            kernel_size=upsample_kernel_size,
            strides=stride,
            use_bias=False,
        )(inputs)

        x = layers.Add()([x, skip])

        # Decoder block
        if conv_decoder:
            # Pure convolutional decoder
            x = UNetResBlock(
                filters=out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )(x)
        else:
            for i in range(depth):
                uid = keras.backend.get_uid(prefix="UNETRPlusPlusTransformer")
                x = UNETRPlusPlusTransformer(
                    sequence_length=sequence_length,
                    hidden_size=out_channels,
                    proj_size=proj_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    pos_embed=True,
                    name=f"trans_block_{uid}_{i}",
                )(x)
        return x

    return wrapper


class UNETRPlusPlusUpBlock(layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name="instance",
        proj_size=64,
        num_heads=4,
        sequence_length=0,
        depth=3,
        conv_decoder=False,
        dropout_rate=0.1,
        name="unetr_pp_block",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.proj_size = proj_size
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.depth = depth
        self.conv_decoder = conv_decoder
        self.dropout_rate = dropout_rate
        self.prefix = name

    def build(self, input_shape):
        x_shape, skip_shape = input_shape
        spatial_dims = len(x_shape) - 2

        # Upsampling layer
        self.up = get_conv_layer(
            spatial_dims,
            layer_type="conv_transpose",
            filters=self.filters,
            kernel_size=self.upsample_kernel_size,
            strides=self.upsample_kernel_size,
            use_bias=False,
        )

        self.add = layers.Add()

        # Decoder block(s): conv OR transformer(s)
        self.blocks = []

        if self.conv_decoder:
            self.blocks.append(
                UNetResBlock(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                    norm_name=self.norm_name,
                )
            )
        else:
            # Transformer decoder (multiple layers)
            for i in range(self.depth):
                block = UNETRPlusPlusTransformer(
                    sequence_length=self.sequence_length,
                    hidden_size=self.filters,
                    proj_size=self.proj_size,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    pos_embed=True,
                    name=f"{self.prefix}_{i}",
                )
                self.blocks.append(block)

        x_shape, skip_shape = input_shape
        self.up.build(x_shape)
        for block in self.blocks:
            block.build(skip_shape)
        self.built = True

    def call(self, inputs, training=None):
        x, skip = inputs

        x = self.up(x, training=training)
        x = self.add([x, skip])

        for block in self.blocks:
            x = block(x, training=training)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "upsample_kernel_size": self.upsample_kernel_size,
                "norm_name": self.norm_name,
                "proj_size": self.proj_size,
                "num_heads": self.num_heads,
                "sequence_length": self.sequence_length,
                "depth": self.depth,
                "conv_decoder": self.conv_decoder,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

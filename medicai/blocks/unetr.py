from keras import layers

from medicai.blocks import UNetBasicBlock, UNetResBlock
from medicai.utils import get_conv_layer


class UNETRBasicBlock(layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        norm_name="instance",
        res_block=True,
        name="unetr_basic_block",
        *kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_name = norm_name
        self.res_block = res_block

        # child block
        if res_block:
            self.block = UNetResBlock(
                filters=filters,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )
        else:
            self.block = UNetBasicBlock(
                filters=filters,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )

    def build(self, input_shape):
        self.block.build(input_shape)
        self.built = True

    def call(self, inputs, training=None):
        return self.block(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "norm_name": self.norm_name,
                "res_block": self.res_block,
            }
        )
        return config


class UNETRUpsamplingBlock(layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        upsample_kernel_size=2,
        norm_name="instance",
        res_block=True,
        name="unetr_upsampling_block",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.res_block = res_block

    def build(self, input_shape):
        x_shape, skip_shape = input_shape
        spatial_dims = len(x_shape) - 2

        # (upsample)
        self.up = get_conv_layer(
            spatial_dims,
            layer_type="conv_transpose",
            filters=self.filters,
            kernel_size=self.upsample_kernel_size,
            strides=self.upsample_kernel_size,
            use_bias=False,
            name="unetr_up_conv_transpose",
        )

        # concat layer
        self.concat = layers.Concatenate(axis=-1)

        # conv block (residual or basic)
        if self.res_block:
            self.block = UNetResBlock(
                filters=self.filters,
                kernel_size=self.kernel_size,
                stride=1,
                norm_name=self.norm_name,
            )
        else:
            self.block = UNetBasicBlock(
                filters=self.filters,
                kernel_size=self.kernel_size,
                stride=1,
                norm_name=self.norm_name,
            )

        self.up.build(x_shape)
        concat_channels = self.filters + skip_shape[-1]
        block_input_shape = (*x_shape[:-1], concat_channels)
        self.block.build(block_input_shape)
        self.built = True

    def call(self, inputs, training=None):
        x, skip = inputs
        x = self.up(x, training=training)
        x = self.concat([x, skip])
        x = self.block(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "upsample_kernel_size": self.upsample_kernel_size,
                "norm_name": self.norm_name,
                "res_block": self.res_block,
            }
        )
        return config


class UNETRPreUpsamplingBlock(layers.Layer):
    def __init__(
        self,
        filters,
        num_layer,
        kernel_size,
        stride,
        upsample_kernel_size,
        conv_block=False,
        res_block=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.stride = stride
        self.upsample_kernel_size = upsample_kernel_size
        self.conv_block = conv_block
        self.res_block = res_block

    def build(self, input_shape):
        spatial_dims = len(input_shape) - 2

        self.transp_conv_init = get_conv_layer(
            spatial_dims,
            layer_type="conv_transpose",
            filters=self.filters,
            kernel_size=self.upsample_kernel_size,
            strides=self.upsample_kernel_size,
            padding="same",
        )

        self.blocks = []
        for _ in range(self.num_layer):

            # Always start with transpose conv
            up_layer = get_conv_layer(
                spatial_dims,
                layer_type="conv_transpose",
                filters=self.filters,
                kernel_size=self.upsample_kernel_size,
                strides=self.upsample_kernel_size,
                padding="same",
            )

            if self.conv_block:
                if self.res_block:
                    conv_layer = UNetResBlock(
                        filters=self.filters,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        norm_name="instance",
                    )
                else:
                    conv_layer = UNetBasicBlock(
                        filters=self.filters,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        norm_name="instance",
                    )
                self.blocks.append((up_layer, conv_layer))
            else:
                # Only transpose conv
                self.blocks.append((up_layer, None))

        # Initial transpose conv
        self.transp_conv_init.build(input_shape)
        input_shape = list(input_shape)
        input_shape[-1] = self.filters

        # Build each block layer by layer
        for up_layer, conv_layer in self.blocks:
            up_layer.build(tuple(input_shape))
            input_shape[-1] = self.filters
            if conv_layer is not None:
                conv_layer.build(tuple(input_shape))

        self.built = True

    def call(self, inputs, training=None):
        x = inputs

        # Initial upsample
        x = self.transp_conv_init(x, training=training)

        # Sequential blocks
        for up_layer, conv_layer in self.blocks:
            x = up_layer(x, training=training)
            if conv_layer is not None:
                x = conv_layer(x, training=training)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "num_layer": self.num_layer,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "upsample_kernel_size": self.upsample_kernel_size,
                "conv_block": self.conv_block,
                "res_block": self.res_block,
            }
        )
        return config

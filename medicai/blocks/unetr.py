from keras import layers

from medicai.blocks import UNetBasicBlock, UNetResBlock
from medicai.utils import get_conv_layer


class UNETRBasicBlock(layers.Layer):
    """
    A basic building block for a 3D UNet, consisting of two convolutional layers
    with normalization and LeakyReLU activation, and optional dropout.

    Args:
        filters (int): The number of output channels for both convolutional layers.
        kernel_size (int): The size of the convolutional kernel in all spatial dimensions
            (default: 3).
        stride (int): The stride of the first convolutional layer in all spatial dimensions
            (default: 1).
        norm_name (Optional[str]): The name of the normalization layer to use.
            Options are "instance" (requires tensorflow-addons), "batch", or None for no
            normalization (default: "instance").
        dropout_rate (Optional[float]): The dropout rate (between 0 and 1). If None, no
            dropout is applied (default: None).

    Returns:
        Callable: A function that takes an input tensor and returns the output
            tensor after applying the basic block.
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        norm_name="instance",
        res_block=True,
        name="unetr_basic_block",
        **kwargs
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
    """
    A UNETR Decoder block that performs feature upsampling, concatenation with
    a skip connection, and subsequent feature processing.


    This block typically takes two inputs: the feature map to be upsampled
    from the previous decoder stage (x_in) and the corresponding skip connection
    from the encoder (x_skip).

    Args:
        filters (int): The number of output channels after feature processing.
        kernel_size (int or tuple): The kernel size for the internal convolutional processing
            block (Basic or Residual). (default: 3).
        stride (int or tuple): The stride for the internal convolutional processing block.
            (default: 1).
        upsample_kernel_size (int or tuple): Kernel size/stride for the transpose convolution
            layer used for upsampling. (default: 2).
        norm_name (Optional[str]): The name of the normalization layer to use in the internal
            processing block (default: "instance").
        res_block (bool): If True, uses a residual block (`UnetResBlock`) for feature
            processing; otherwise, uses a basic block (`UnetBasicBlock`). (default: True).
    """

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
        up_output_shape = self.up.compute_output_shape(x_shape)
        block_input_shape = (*up_output_shape[:-1], concat_channels)
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
    """
    A Keras Layer that implements the UNETR Projection Upsampling block.

    This block performs an initial transpose convolution to upsample features,
    followed by a sequence of 'num_layer' blocks. Each subsequent block consists
    of another transpose convolution (upsample) and an optional convolutional
    sub-block (basic or residual).

    Args:
        filters (int): Number of output channels for the block.
        num_layer (int): Number of repeated (Upsample + Conv) stages after the initial upsample.
        kernel_size (int or tuple): Kernel size for the optional convolutional sub-blocks.
        stride (int or tuple): Stride for the optional convolutional sub-blocks.
        upsample_kernel_size (int or tuple): Kernel/stride size for all transpose convolutions.
        conv_block (bool): If True, applies convolutional/residual sub-blocks after transpose convs.
        res_block (bool): If True and conv_block is True, uses UnetResBlock; otherwise,
            uses UnetBasicBlock.

    Example:
        # Create a 3D UNETR upsampling block
        up_block = UNETRPreUpsamplingBlock(
            filters=128,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            conv_block=True,
            res_block=True,
        )
        # Assuming input x is 3D, e.g., (B, 16, 16, 16, C)
        # This block will perform 1 + 2 = 3 upsampling steps.
        y = up_block(x)
    """

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
        self.transp_conv_init.build(input_shape)

        self.blocks = []
        current_shape = self.transp_conv_init.compute_output_shape(input_shape)

        for _ in range(self.num_layer):
            # Transpose Conv layer (instantiated inside the loop
            up_layer = get_conv_layer(
                spatial_dims,
                layer_type="conv_transpose",
                filters=self.filters,
                kernel_size=self.upsample_kernel_size,
                strides=self.upsample_kernel_size,
                padding="same",
            )
            up_layer.build(current_shape)
            current_shape = up_layer.compute_output_shape(current_shape)

            conv_layer = None
            if self.conv_block:
                # Convolutional/Residual layer
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

                # Build the conv layer
                conv_layer.build(current_shape)
                current_shape = conv_layer.compute_output_shape(current_shape)

            self.blocks.append((up_layer, conv_layer))

        self.built = True

    def call(self, inputs, training=None):
        x = inputs

        # 1. Initial upsample
        x = self.transp_conv_init(x, training=training)

        # 2. Sequential blocks (Up+Conv/Res)
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

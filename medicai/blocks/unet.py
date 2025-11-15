import numpy as np
from keras import layers

from medicai.utils import get_act_layer, get_conv_layer, get_norm_layer


def UnetBasicBlock(
    spatial_dims, out_channels, kernel_size=3, stride=1, norm_name="instance", dropout_rate=None
):
    """
    A basic building block for a UNet (2D or 3D), consisting of two convolutional layers
    with normalization and LeakyReLU activation, and optional dropout.

    Args:
        out_channels (int): The number of output channels for both convolutional layers.
        kernel_size (int): The size of the convolutional kernel in all spatial dimensions (default: 3).
        stride (int): The stride of the first convolutional layer in all spatial dimensions (default: 1).
        norm_name (Optional[str]): The name of the normalization layer to use.
            Options are "instance" (requires tensorflow-addons), "batch", or None for no normalization (default: "instance").
        dropout_rate (Optional[float]): The dropout rate (between 0 and 1). If None, no dropout is applied (default: None).

    Returns:
        Callable: A function that takes an input tensor and returns the output
            tensor after applying the basic block.
    """

    def apply(inputs):
        x = get_conv_layer(
            spatial_dims,
            layer_type="conv",
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            use_bias=False,
        )(inputs)
        x = get_norm_layer(norm_name)(x)
        x = get_act_layer("leaky_relu", negative_slope=0.01)(x)

        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        x = get_conv_layer(
            spatial_dims,
            layer_type="conv",
            filters=out_channels,
            kernel_size=kernel_size,
            strides=1,
            use_bias=False,
        )(x)
        x = get_norm_layer(norm_name)(x)
        x = get_act_layer("leaky_relu", negative_slope=0.01)(x)

        return x

    return apply


def UnetOutBlock(spatial_dims, num_classes, dropout_rate=None, activation=None):
    """The output block of a 3D UNet, consisting of a 1x1x1 convolutional layer
    to map the features to the desired number of classes, with optional dropout
    and a final activation function.

    Args:
        num_classes (int): The number of output classes for the segmentation task.
            This determines the number of output channels of the convolutional layer.
        dropout_rate (Optional[float]): The dropout rate (between 0 and 1). If None,
            no dropout is applied (default: None).
        activation (Optional[Union[str, layers.Activation]]): The activation function
            to apply to the output of the convolutional layer. This can be a string
            (e.g., 'softmax', 'sigmoid') or a Keras activation layer instance.
            If None, no activation is applied (default: None).

    Returns:
        Callable: A function that takes an input tensor and returns the output
            tensor after applying the 1x1x1 convolution and optional dropout
            and activation.
    """

    def wrapper(inputs):

        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        x = get_conv_layer(
            spatial_dims,
            layer_type="conv",
            filters=num_classes,
            kernel_size=1,
            strides=1,
            use_bias=True,
            activation=activation,
            dtype="float32",
        )(inputs)
        return x

    return wrapper


def UnetResBlock(
    spatial_dims,
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    norm_name="instance",
    dropout_rate=None,
):
    """
    A residual building block for a 3D UNet, consisting of two convolutional layers
    with normalization, LeakyReLU activation, optional dropout, and a skip connection.

    Args:
        in_channels (int): The number of input channels to the block.
        out_channels (int): The number of output channels for the convolutional layers.
        kernel_size (int): The size of the convolutional kernel in all spatial dimensions (default: 3).
        stride (int): The stride of the first convolutional layer in all spatial dimensions (default: 1).
        norm_name (Optional[str]): The name of the normalization layer to use.
            Options are "instance" (requires tensorflow-addons), "batch", or None for no normalization (default: "instance").
        dropout_rate (Optional[float]): The dropout rate (between 0 and 1). If None, no dropout is applied (default: None).

    Returns:
        Callable: A function that takes an input tensor and returns the output
            tensor after applying the residual block.
    """

    def wrapper(inputs):
        # first convolution
        x = get_conv_layer(
            spatial_dims,
            layer_type="conv",
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            use_bias=False,
        )(inputs)
        x = get_norm_layer(norm_name)(x)
        x = get_act_layer("leaky_relu", negative_slope=0.01)(x)

        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        # second convolution
        x = get_conv_layer(
            spatial_dims,
            layer_type="conv",
            filters=out_channels,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            use_bias=False,
        )(x)
        x = get_norm_layer(norm_name)(x)

        # residual
        residual = inputs
        downsample = (in_channels != out_channels) or (np.atleast_1d(stride) != 1).any()
        if downsample:
            residual = get_conv_layer(
                spatial_dims,
                layer_type="conv",
                filters=out_channels,
                kernel_size=1,
                strides=stride,
                padding="same",
                use_bias=False,
            )(residual)
            residual = get_norm_layer(norm_name)(residual)

        # add residual connection
        x = layers.Add()([x, residual])
        x = get_act_layer("leaky_relu", negative_slope=0.01)(x)

        return x

    return wrapper


class UNetResBlock(layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        norm_name="instance",
        dropout_rate=None,
        name="unet_residual_block",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_name = norm_name
        self.dropout_rate = dropout_rate
        self.use_dropout = dropout_rate is not None
        self.prefix = name

    def build(self, input_shape):
        spatial_dims = len(input_shape) - 2
        in_channels = input_shape[-1]

        self.conv1 = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="same",
            use_bias=False,
            name=f"{self.prefix}_conv1",
        )
        self.conv1.build(input_shape)

        self.norm1 = get_norm_layer(layer_type=self.norm_name, name=f"{self.prefix}_norm1")
        self.norm1.build(self.conv1.compute_output_shape(input_shape))
        self.act1 = get_act_layer(
            layer_type="leaky_relu", negative_slope=0.01, name=f"{self.prefix}_act1"
        )

        if self.use_dropout:
            self.dropout = layers.Dropout(self.dropout_rate, name=f"{self.prefix}_dropout")

        conv1_out_shape = self.norm1.compute_output_shape(
            self.conv1.compute_output_shape(input_shape)
        )
        self.conv2 = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"{self.prefix}_conv2",
        )
        self.conv2.build(conv1_out_shape)

        self.norm2 = get_norm_layer(layer_type=self.norm_name, name=f"{self.prefix}_norm2")
        self.norm2.build(self.conv2.compute_output_shape(conv1_out_shape))

        needs_res_conv = (in_channels != self.filters) or (self.stride != 1)
        if needs_res_conv:
            self.res_conv = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=self.filters,
                kernel_size=1,
                strides=self.stride,
                padding="same",
                use_bias=False,
                name=f"{self.prefix}_res_conv",
            )
            self.res_conv.build(input_shape)
            self.res_norm = get_norm_layer(
                layer_type=self.norm_name, name=f"{self.prefix}_res_norm"
            )
            self.res_norm.build(self.res_conv.compute_output_shape(input_shape))
        else:
            self.res_conv = None
            self.res_norm = None

        self.act2 = get_act_layer(
            layer_type="leaky_relu", negative_slope=0.01, name=f"{self.prefix}_act2"
        )

        self.built = True

    def call(self, x, training=False):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out, training=training)
        out = self.act1(out)

        if self.use_dropout:
            out = self.dropout(out, training=training)

        out = self.conv2(out)
        out = self.norm2(out, training=training)

        # Handle residual connection
        if self.res_conv is not None:
            skip = self.res_conv(identity)
            skip = self.res_norm(skip, training=training)
        else:
            skip = identity

        out = layers.add([out, skip])
        out = self.act2(out)

        return out

    def compute_output_shape(self, input_shape):
        spatial_dims = len(input_shape) - 2
        batch_size = input_shape[0]

        spatial_shape = []
        for i in range(1, spatial_dims + 1):
            spatial_dim = input_shape[i]
            if self.stride > 1:
                # For 'same' padding with stride, output size is ceil(input_size / stride)
                spatial_dim = (spatial_dim + self.stride - 1) // self.stride
            spatial_shape.append(spatial_dim)

        output_shape = [batch_size] + spatial_shape + [self.out_channels]
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_channels": self.out_channels,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "norm_name": self.norm_name,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

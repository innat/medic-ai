import numpy as np
from keras import layers

from medicai.utils import get_act_layer, get_norm_layer


def UnetResBlock(
    in_channels, out_channels, kernel_size=3, stride=1, norm_name="instance", dropout_rate=None
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
        x = layers.Conv3D(
            out_channels, kernel_size, strides=stride, padding="same", use_bias=False
        )(inputs)
        x = get_norm_layer(norm_name)(x)
        x = get_act_layer(("leakyrelu", {"negative_slope": 0.01}))(x)

        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        # second convolution
        x = layers.Conv3D(out_channels, kernel_size, strides=1, padding="same", use_bias=False)(x)
        x = get_norm_layer(norm_name)(x)

        # residual
        residual = inputs
        downsample = (in_channels != out_channels) or (np.atleast_1d(stride) != 1).any()
        if downsample:
            residual = layers.Conv3D(
                out_channels, kernel_size=1, strides=stride, padding="same", use_bias=False
            )(residual)
            residual = get_norm_layer(norm_name)(residual)

        # add residual connection
        x = layers.Add()([x, residual])
        x = get_act_layer(("leakyrelu", {"negative_slope": 0.01}))(x)

        return x

    return wrapper

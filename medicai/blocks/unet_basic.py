from keras import layers

from medicai.utils import get_act_layer, get_norm_layer


def UnetBasicBlock(out_channels, kernel_size=3, stride=1, norm_name="instance", dropout_rate=None):
    """
    A basic building block for a 3D UNet, consisting of two convolutional layers
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
        x = layers.Conv3D(out_channels, kernel_size, strides=stride, use_bias=False)(inputs)
        x = get_norm_layer(norm_name)(x)
        x = get_act_layer(("leakyrelu", {"negative_slope": 0.01}))(x)

        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        x = layers.Conv3D(out_channels, kernel_size, strides=1, use_bias=False)(x)
        x = get_norm_layer(norm_name)(x)
        x = get_act_layer(("leakyrelu", {"negative_slope": 0.01}))(x)

        return x

    return apply

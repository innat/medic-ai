from medicai.blocks import UnetBasicBlock, UnetResBlock


def UnetrBasicBlock(out_channels, kernel_size=3, stride=1, norm_name="instance", res_block=True):
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

    def wrapper(inputs):
        if res_block:
            x = UnetResBlock(
                in_channels=inputs.shape[-1],
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )(inputs)
        else:
            x = UnetBasicBlock(
                out_channels,
                kernel_size,
                stride,
                norm_name,
            )(inputs)
        return x

    return wrapper

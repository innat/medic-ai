from medicai.blocks import UnetBasicBlock, UnetResBlock
from medicai.utils import get_conv_layer


def UnetrPrUpBlock(
    spatial_dims,
    out_channels,
    num_layer,
    kernel_size,
    stride,
    upsample_kernel_size,
    conv_block=False,
    res_block=False,
):
    """
    Functional closure version of a UNETR projection upsampling block.

    This block performs upsampling using a transpose convolution followed
    by optional convolutional or residual sub-blocks. It returns a callable
    function `apply(x)` that applies the block to an input tensor.

    Args:
        spatial_dims (int): Number of spatial dimensions (2 or 3).
        out_channels (int): Number of output channels for the upsampled feature.
        num_layer (int): Number of convolutional/residual sub-blocks after the initial transpose convolution.
        kernel_size (int or tuple): Kernel size for the convolutional sub-blocks.
        stride (int or tuple): Stride for the convolutional sub-blocks.
        upsample_kernel_size (int or tuple): Kernel size / stride for the initial transpose convolution.
        conv_block (bool): If True, apply convolutional sub-blocks after transpose conv.
        res_block (bool): If True and conv_block is True, use residual blocks (`UnetResBlock`)
                        instead of basic convolutional blocks (`UnetBasicBlock`).

    Returns:
        function: A callable `apply(x)` function that applies the UNETR projection upsampling block
                to a tensor `x`.

    Example:
        # Create a 3D UNETR upsampling block
        up_block = UnetrPrUpBlock(
            spatial_dims=3,
            out_channels=128,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            conv_block=True,
            res_block=True,
        )

        # Apply it to a feature tensor `x`
        y = up_block(x)
    """

    # Base transpose conv initializer
    transp_conv_init = get_conv_layer(
        spatial_dims,
        transpose=True,
        filters=out_channels,
        kernel_size=upsample_kernel_size,
        strides=upsample_kernel_size,
        padding="same",
    )

    block_fns = []
    for _ in range(num_layer):
        if conv_block:
            if res_block:

                def block_fn(x):
                    x = get_conv_layer(
                        spatial_dims,
                        transpose=True,
                        filters=out_channels,
                        kernel_size=upsample_kernel_size,
                        strides=upsample_kernel_size,
                        padding="same",
                    )(x)
                    x = UnetResBlock(
                        spatial_dims,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        norm_name="instance",
                    )(x)
                    return x

            else:

                def block_fn(x):
                    x = get_conv_layer(
                        spatial_dims,
                        transpose=True,
                        filters=out_channels,
                        kernel_size=upsample_kernel_size,
                        strides=upsample_kernel_size,
                        padding="same",
                    )(x)
                    x = UnetBasicBlock(
                        spatial_dims,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        norm_name="instance",
                    )(x)
                    return x

        else:

            def block_fn(x):
                x = get_conv_layer(
                    spatial_dims,
                    transpose=True,
                    filters=out_channels,
                    kernel_size=upsample_kernel_size,
                    strides=upsample_kernel_size,
                    padding="same",
                )(x)
                return x

        block_fns.append(block_fn)

    def apply(x):
        x = transp_conv_init(x)
        for fn in block_fns:
            x = fn(x)
        return x

    return apply

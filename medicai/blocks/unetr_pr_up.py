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
    Functional closure version of UNETR projection upsampling block.
    Returns a callable `apply(x)` function.
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

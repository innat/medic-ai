from keras import layers

from medicai.blocks import UnetBasicBlock, UnetResBlock


def UnetrUpBlock(
    out_channels,
    kernel_size=3,
    stride=1,
    upsample_kernel_size=2,
    norm_name="instance",
    res_block=True,
):
    stride = upsample_kernel_size

    def wrapper(inputs, skip):
        x = layers.Conv3DTranspose(
            out_channels,
            kernel_size=upsample_kernel_size,
            strides=stride,
            use_bias=False,
        )(inputs)

        # Concatenate with skip connection
        x = layers.Concatenate(axis=-1)([x, skip])

        # Apply the convolutional block
        if res_block:
            x = UnetResBlock(
                in_channels=x.shape[-1],
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )(x)
        else:
            x = UnetBasicBlock(
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )(x)
        return x

    return wrapper

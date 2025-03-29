from medicai.blocks import UnetBasicBlock, UnetResBlock


def UnetrBasicBlock(out_channels, kernel_size=3, stride=1, norm_name="instance", res_block=True):
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

import keras
from keras import layers

from medicai.models.unetr_plus_plus.encoder_layers import UNETRPlusPlusTransformer
from medicai.utils import get_conv_layer

from .unet import UnetResBlock


def UNETRPlusPlusUpBlock(
    spatial_dims,
    out_channels,
    kernel_size=3,
    upsample_kernel_size=2,
    norm_name="instance",
    proj_size=64,
    num_heads=4,
    sequence_length=0,
    depth=3,
    conv_decoder=False,
    dropout_rate=0.1,
):
    stride = upsample_kernel_size

    def wrapper(inputs, skip):
        x = get_conv_layer(
            spatial_dims,
            layer_type="conv_transpose",
            filters=out_channels,
            kernel_size=upsample_kernel_size,
            strides=stride,
            use_bias=False,
        )(inputs)

        x = layers.Add()([x, skip])

        # Decoder block
        if conv_decoder:
            # Pure convolutional decoder
            x = UnetResBlock(
                spatial_dims,
                in_channels=x.shape[-1],
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )(x)
        else:
            for i in range(depth):
                uid = keras.backend.get_uid(prefix="UNETRPlusPlusTransformer")
                x = UNETRPlusPlusTransformer(
                    sequence_length=sequence_length,
                    hidden_size=out_channels,
                    proj_size=proj_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    pos_embed=True,
                    name=f"trans_block_{uid}_{i}",
                )(x)
        return x

    return wrapper

import keras
from keras import layers, ops

from medicai.layers import ConvBnAct, ResizingND
from medicai.utils import get_act_layer, get_pooling_layer


def PyramidPoolingModule(
    out_channels,
    pool_scales=(1, 2, 3, 6),
    decoder_normalization="batchnorm",
):
    # Create pooling blocks
    blocks = []
    for size in pool_scales:

        def create_pooling_block(size=size):
            def apply(x):
                spatial_dims = len(x.shape) - 2
                x = get_pooling_layer(
                    spatial_dims=spatial_dims, layer_type="adaptive_avg", output_size=size
                )(x)
                x = ConvBnAct(
                    out_channels,
                    kernel_size=1,
                    padding="valid",
                    activation="relu",
                    normalization=decoder_normalization,
                )(x)
                return x

            return apply

        blocks.append(create_pooling_block(size))

    # Create output convolution
    out_conv = ConvBnAct(
        out_channels,
        kernel_size=3,
        padding="same",
        activation="relu",
        normalization=decoder_normalization,
    )

    def apply(feature):
        batch_size, height, width, channels = ops.shape(feature)

        pyramid_features = [feature]

        # Apply pooling blocks and resize back to original size
        for block in blocks:
            pooled_feature = block(feature)

            # Resize back to original spatial dimensions
            resized_feature = layers.UpSampling2D(
                size=(
                    height // ops.shape(pooled_feature)[1],
                    width // ops.shape(pooled_feature)[2],
                ),
                interpolation="bilinear",
            )(pooled_feature)

            pyramid_features.append(resized_feature)

        # Concatenate all pyramid features
        fused_feature = layers.Concatenate(axis=-1)(pyramid_features)

        # Apply output convolution
        output = out_conv(fused_feature)

        return output

    return apply

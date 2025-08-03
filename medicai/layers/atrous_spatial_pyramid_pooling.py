
import keras
from keras import layers
from keras import ops

def SpatialPyramidPooling3D(
    dilation_rates,
    num_channels=256,
    activation="relu",
    dropout=0.0,
):
    """Implements the Atrous Spatial Pyramid Pooling for 3D data using a closure.

    This is the 3D equivalent of the original DeepLabV3+ ASPP module.
    It processes 3D feature maps (depth, height, width, channels) using
    parallel 3D dilated convolutions and a global average pooling branch.

    Args:
        dilation_rates: list of ints. The dilation rate for parallel dilated conv.
            Usually a sample choice of rates are `[6, 12, 18]`.
        num_channels: int. The number of output channels, defaults to `256`.
        activation: str. Activation to be used, defaults to `relu`.
        dropout: float. The dropout rate of the final projection output after the
            activations and batch norm, defaults to `0.0`, which means no dropout is
            applied to the output.

    Returns:
        A function `apply(inputs)` that takes an input tensor and returns the
        ASPP output tensor.

    Example:
    ```python
    inp = keras.layers.Input((64, 128, 128, 1))
    feature_map = keras.layers.Conv3D(256, 1)(inp)
    aspp_layer = SpatialPyramidPooling3D(dilation_rates=[6, 12, 18])
    output = aspp_layer(feature_map)
    ```
    """
    def apply(inputs):
        """
        Applies the 3D ASPP module to an input tensor.

        Args:
            inputs: A tensor of shape [batch, depth, height, width, channels]

        Returns:
            A tensor of shape [batch, depth, height, width, num_channels]
        """
        data_format = "channels_last"
        channel_axis = -1

        # Get the input shape dynamically from the input tensor
        image_shape = ops.shape(inputs)
        depth, height, width, channels = image_shape[1], image_shape[2], image_shape[3], image_shape[4]

        aspp_branches = []

        # Branch 1: 1x1x1 Conv3D
        aspp_branches.append(
            keras.Sequential([
                layers.Conv3D(
                    num_channels, 
                    (1, 1, 1), 
                    use_bias=False, 
                    data_format=data_format, 
                    name="aspp_conv_1"
                ),
                layers.BatchNormalization(axis=channel_axis, name="aspp_bn_1"),
                layers.Activation(activation, name="aspp_activation_1"),
            ])(inputs)
        )

        # Branches 2 to N: Dilated Conv3D with 3x3x3 kernel size
        for i, rate in enumerate(dilation_rates):
            aspp_branches.append(
                keras.Sequential([
                    layers.Conv3D(
                        num_channels, 
                        (3, 3, 3), 
                        padding="same", 
                        dilation_rate=(rate, rate, rate), 
                        use_bias=False, 
                        data_format=data_format, 
                        name=f"aspp_conv_{i+2}"
                    ),
                    layers.BatchNormalization(axis=channel_axis, name=f"aspp_bn_{i+2}"),
                    layers.Activation(activation, name=f"aspp_activation_{i+2}"),
                ])(inputs)
            )

        # Last branch: Global Average Pooling and upsampling
        global_pool_branch = keras.Sequential([
            layers.GlobalAveragePooling3D(data_format=data_format, name="average_pooling"),
            layers.Reshape((1, 1, 1, channels), name="reshape"),
            layers.Conv3D(
                num_channels, 
                (1, 1, 1), 
                use_bias=False, 
                data_format=data_format, 
                name="conv_pooling"
            ),
            layers.BatchNormalization(axis=channel_axis, name="bn_pooling"),
            layers.Activation(activation, name="activation_pooling"),
            # Upsample to match the input dimensions
            layers.UpSampling3D(
                size=(depth, height, width), 
                data_format=data_format, 
                name="upsample_pooling"
            )
        ])(inputs)
        
        aspp_branches.append(global_pool_branch)

        # Concatenate all parallel branch outputs
        concatenated_features = layers.Concatenate(
            axis=channel_axis, name="aspp_concat"
        )(aspp_branches)

        # Final projection layers
        projection = keras.Sequential([
            layers.Conv3D(
                num_channels, 
                (1, 1, 1), 
                use_bias=False, 
                data_format=data_format, 
                name="conv_projection"
            ),
            layers.BatchNormalization(axis=channel_axis, name="bn_projection"),
            layers.Activation(activation, name="activation_projection"),
            layers.Dropout(rate=dropout, name="dropout")
        ])(concatenated_features)

        return projection

    return apply
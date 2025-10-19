import keras
from keras import layers

from medicai.utils import (
    get_act_layer,
    get_conv_layer,
    get_norm_layer,
    get_pooling_layer,
    get_reshaping_layer,
    parse_model_inputs,
)

from .resnext_layers import conv_block, identity_block


class ResNeXtBackbone(keras.Model):
    """ResNeXt backbone network for feature extraction.

    Args:
        blocks: List of integers, number of blocks in each stage
        input_shape: Tuple, shape of input tensor
        input_tensor: Optional tensor to use as model input
        groups: Integer, number of groups for grouped convolutions (cardinality)
        normalization: String, type of normalization to use
        activation: String, activation function to use
        include_rescaling: Boolean, whether to include input rescaling
        name: String, name of the model
    """

    def __init__(
        self,
        blocks,
        input_shape,
        input_tensor=None,
        groups=32,
        normalization="batch",
        activation="relu",
        include_rescaling=False,
        name=None,
    ):
        spatial_dims = len(input_shape) - 1
        inputs = parse_model_inputs(input_shape, input_tensor, name="input_spec")

        x = inputs
        if include_rescaling:
            x = layers.Rescaling(1.0 / 255, name="rescaling")(x)

        # Initial stems
        x = get_norm_layer(layer_type=normalization, name="initial_norm")(x)
        x = get_reshaping_layer(
            spatial_dims=spatial_dims,
            layer_type="padding",
            padding=3,
            name="pre_conv_padding",
        )(x)
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=64,
            kernel_size=7,
            strides=2,
            padding="valid",
            use_bias=False,
            name="downsample_conv_7x7",
        )(x)
        x = get_norm_layer(
            layer_type=normalization,
            name="post_conv_norm",
        )(x)
        x = get_act_layer(
            layer_type=activation,
            name="post_conv_act",
        )(x)

        pyramid_outputs = {}
        pyramid_outputs["P1"] = x

        x = get_reshaping_layer(
            spatial_dims=spatial_dims,
            layer_type="padding",
            padding=1,
            name="maxpool_padding",
        )(x)

        x = get_pooling_layer(
            spatial_dims=spatial_dims,
            layer_type="max",
            pool_size=3,
            strides=2,
            padding="valid",
            name="initial_maxpool",
        )(x)

        # Residual stages
        init_filters = 128

        for stage, rep in enumerate(blocks):
            for block in range(rep):
                filters = init_filters * (2**stage)
                block_name = f"stage_{stage}_block_{block}"

                # First block of first stage without strides because we have maxpooling before
                if stage == 0 and block == 0:
                    x = conv_block(
                        filters,
                        strides=1,
                        groups=groups,
                        normalization=normalization,
                        activation=activation,
                        name=block_name,
                    )(x)
                elif block == 0:
                    x = conv_block(
                        filters,
                        strides=2,
                        groups=groups,
                        normalization=normalization,
                        activation=activation,
                        name=block_name,
                    )(x)
                else:
                    x = identity_block(
                        filters,
                        groups=groups,
                        normalization=normalization,
                        activation=activation,
                        name=block_name,
                    )(x)

            # Store output for feature pyramid
            pyramid_outputs[f"P{stage+2}"] = x

        super().__init__(
            inputs=inputs,
            outputs=x,
            name=name or f"ResNeXtBackbone{spatial_dims}D",
        )

        self.blocks = blocks
        self.groups = groups
        self.normalization = normalization
        self.activation = activation
        self.include_rescaling = include_rescaling
        self.pyramid_outputs = pyramid_outputs
        self.name = name

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "blocks": self.blocks,
            "groups": self.groups,
            "normalization": self.normalization,
            "activation": self.activation,
            "include_rescaling": self.include_rescaling,
            "name": self.name,
        }
        return config

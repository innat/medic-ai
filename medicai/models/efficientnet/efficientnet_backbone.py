import copy
import math

import keras
from keras import layers

from medicai.utils import get_conv_layer, get_reshaping_layer, parse_model_inputs

from .efficientnet_layers import (
    DEFAULT_BLOCKS_ARGS,
    DENSE_KERNEL_INITIALIZER,
    EfficientNetBlock,
    correct_pad,
)


class EfficientNetBackbone(keras.Model):
    def __init__(
        self,
        width_coefficient,
        depth_coefficient,
        input_tensor=None,
        input_shape=None,
        drop_connect_rate=0.2,
        depth_divisor=8,
        activation="swish",
        blocks_args="default",
        include_rescaling=False,
        name="efficientnet",
        **kwargs,
    ):
        if blocks_args == "default":
            blocks_args = DEFAULT_BLOCKS_ARGS

        # Input
        inputs = parse_model_inputs(input_shape, input_tensor)
        spatial_dims = len(input_shape) - 1

        # Stem
        x = inputs

        if include_rescaling:
            x = layers.Rescaling(1.0 / 255.0)(x)
            x = layers.Normalization(axis=-1)(x)

        x = get_reshaping_layer(
            spatial_dims=spatial_dims,
            layer_type="padding",
            padding=correct_pad(x, 3),
            name="stem_conv_pad",
        )(x)
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=self.round_filters(
                32, divisor=depth_divisor, width_coefficient=width_coefficient
            ),
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            kernel_initializer="he_normal",
            name="stem_conv",
        )(x)
        x = layers.BatchNormalization(axis=-1, name="stem_bn")(x)
        x = layers.Activation(activation, name="stem_activation")(x)

        # Blocks
        blocks_args = copy.deepcopy(blocks_args)
        b = 0
        blocks = float(
            sum(self.round_repeats(args["repeats"], depth_coefficient) for args in blocks_args)
        )
        for i, args in enumerate(blocks_args):
            assert args["repeats"] > 0
            args["filters_in"] = self.round_filters(
                args["filters_in"], divisor=depth_divisor, width_coefficient=width_coefficient
            )
            args["filters_out"] = self.round_filters(
                args["filters_out"], divisor=depth_divisor, width_coefficient=width_coefficient
            )

            for j in range(self.round_repeats(args.pop("repeats"), depth_coefficient)):
                if j > 0:
                    args["strides"] = 1
                    args["filters_in"] = args["filters_out"]
                x = EfficientNetBlock(
                    activation,
                    drop_connect_rate * b / blocks,
                    name=f"block{i + 1}{chr(j + 97)}_",
                    **args,
                )(x)
                b += 1

        # Top
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=self.round_filters(
                1280, divisor=depth_divisor, width_coefficient=width_coefficient
            ),
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name="top_conv",
        )(x)
        x = layers.BatchNormalization(axis=-1, name="top_bn")(x)
        x = layers.Activation(activation, name="top_activation")(x)

        super().__init__(
            inputs=inputs, outputs=x, name=name or f"EfficientNetBackbone{spatial_dims}D", **kwargs
        )

    def round_filters(self, filters, divisor, width_coefficient):
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(self, repeats, depth_coefficient):
        return int(math.ceil(depth_coefficient * repeats))

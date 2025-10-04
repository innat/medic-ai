import copy
import math

import keras
from keras import layers

from medicai.utils import get_conv_layer, get_reshaping_layer, parse_model_inputs

from .efficientnet_layers import (
    CONV_KERNEL_INITIALIZER,
    DEFAULT_BLOCKS_ARGS,
    EfficientNetBlock,
    correct_pad,
)


class EfficientNetBackbone(keras.Model):
    """
    EfficientNet backbone model supporting both 2D and 3D inputs.

    This class implements the feature extraction (backbone) part of the EfficientNet architecture,
    which scales width, depth, and resolution uniformly using compound scaling.
    It can operate on 2D inputs (e.g., images of shape `(H, W, C)`) or 3D inputs
    (e.g., volumetric data of shape `(D, H, W, C)`).

    The backbone produces multi-scale feature maps that can be used for downstream
    tasks such as classification, detection, or segmentation.

    References:
        - Tan, M. and Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
          ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)

    Example (2D):
        >>> model = EfficientNetBackbone(
        ...     width_coefficient=1.0,
        ...     depth_coefficient=1.0,
        ...     input_shape=(224, 224, 3),
        ...     include_rescaling=True,
        ... )
        >>> x = tf.random.normal((1, 224, 224, 3))
        >>> y = model(x)
        >>> y.shape
        TensorShape([1, 7, 7, 1280])

    Example (3D):
        >>> model = EfficientNetBackbone(
        ...     width_coefficient=1.0,
        ...     depth_coefficient=1.0,
        ...     input_shape=(32, 224, 224, 3),
        ...     include_rescaling=True,
        ... )
        >>> x = tf.random.normal((1, 32, 224, 224, 3))
        >>> y = model(x)
        >>> y.shape
        TensorShape([1, 1, 7, 7, 1280])
    '''
    """

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
        """
        Initializes the EfficientNetBackbone model.

        Args:
            width_coefficient (float):
                Scaling coefficient for network width (number of filters per layer).
            depth_coefficient (float):
                Scaling coefficient for network depth (number of repeated blocks).
            input_tensor (tf.Tensor, optional):
                Optional tensor to use as model input. If None, a new input tensor is created.
            input_shape (tuple, optional):
                Shape of the input tensor. Must include the channel dimension.
                For 2D: `(H, W, C)`, for 3D: `(D, H, W, C)`.
            drop_connect_rate (float, default=0.2):
                Drop connect (stochastic depth) rate applied within MBConv blocks.
            depth_divisor (int, default=8):
                Ensures the number of filters is divisible by this value after scaling.
            activation (str, default="swish"):
                Activation function used throughout the model.
            blocks_args (list or str, default="default"):
                Configuration for the sequence of MBConv blocks. If "default", uses the
                standard EfficientNet block structure.
            include_rescaling (bool, default=False):
                If True, includes input rescaling (1/255) and normalization layers.
            name (str, default="efficientnet"):
                Model name.
        """
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
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name="stem_conv",
        )(x)
        x = layers.BatchNormalization(axis=-1, name="stem_bn")(x)
        x = layers.Activation(activation, name="stem_activation")(x)

        # Blocks
        blocks_args = copy.deepcopy(blocks_args)
        pyramid_outputs = {"P1": x}
        pyramid_level = 2

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

                if args["strides"] != 1:
                    pyramid_outputs[f"P{pyramid_level}"] = x
                    pyramid_level += 1

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
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name="top_conv",
        )(x)
        x = layers.BatchNormalization(axis=-1, name="top_bn")(x)
        x = layers.Activation(activation, name="top_activation")(x)
        super().__init__(
            inputs=inputs, outputs=x, name=name or f"EfficientNetBackbone{spatial_dims}D", **kwargs
        )

        self.pyramid_outputs = pyramid_outputs
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.drop_connect_rate = drop_connect_rate
        self.depth_divisor = depth_divisor
        self.activation = activation
        self.blocks_args = blocks_args
        self.include_rescaling = include_rescaling
        self.name = name

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "width_coefficient": self.width_coefficient,
            "depth_coefficient": self.depth_coefficient,
            "drop_connect_rate": self.drop_connect_rate,
            "depth_divisor": self.depth_divisor,
            "activation": self.activation,
            "blocks_args": self.blocks_args,
            "include_rescaling": self.include_rescaling,
            "name": self.name,
        }
        return config

    def round_filters(self, filters, divisor, width_coefficient):
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(self, repeats, depth_coefficient):
        return int(math.ceil(depth_coefficient * repeats))

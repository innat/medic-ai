import copy
import math

import keras
from keras import layers

from medicai.utils import get_conv_layer, get_reshaping_layer, parse_model_inputs

from .efficientnet_layers import (
    CONV_KERNEL_INITIALIZER,
    DEFAULT_BLOCKS_ARGS_V1,
    DEFAULT_BLOCKS_ARGS_V2,
    EfficientNetV1Block,
    FusedMBConvBlock,
    MBConvBlock,
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

    '''
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
        name=None,
        **kwargs,
    ):
        if blocks_args == "default":
            blocks_args = DEFAULT_BLOCKS_ARGS_V1

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

                x = EfficientNetV1Block(
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
            inputs=inputs,
            outputs=x,
            name=name or f"EfficientNetV1Backbone{spatial_dims}D",
            **kwargs,
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

    @staticmethod
    def round_filters(filters, divisor, width_coefficient):
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    @staticmethod
    def round_repeats(repeats, depth_coefficient):
        return int(math.ceil(depth_coefficient * repeats))


class EfficientNetBackboneV2(keras.Model):
    def __init__(
        self,
        width_coefficient,
        depth_coefficient,
        input_tensor=None,
        input_shape=None,
        drop_connect_rate=0.2,
        depth_divisor=8,
        min_depth=8,
        bn_momentum=0.9,
        activation="swish",
        blocks_args="efficientnetv2-s",
        include_rescaling=False,
        name="efficientnet_v2",
        **kwargs,
    ):

        # Handle blocks_args as config key
        if isinstance(blocks_args, str):
            if blocks_args in DEFAULT_BLOCKS_ARGS_V2:
                blocks_args = DEFAULT_BLOCKS_ARGS_V2[blocks_args]
            else:
                available_keys = list(DEFAULT_BLOCKS_ARGS_V2.keys())
                raise ValueError(
                    f"Unknown blocks_args configuration: '{blocks_args}'. "
                    f"Available configurations: {available_keys}"
                )
        # If blocks_args is already a list (custom config)
        elif isinstance(blocks_args, list):
            pass
        else:
            raise TypeError(
                f"blocks_args must be str (config key) or list (custom config). "
                f"Got {type(blocks_args)}"
            )

        # Input
        inputs = parse_model_inputs(input_shape, input_tensor)
        spatial_dims = len(input_shape) - 1

        # Stem
        x = inputs

        if include_rescaling:
            x = layers.Rescaling(1.0 / 255.0)(x)
            x = layers.Normalization(axis=-1)(x)

        stem_filters = self.round_filters(
            filters=blocks_args[0]["input_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )

        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=stem_filters,
            kernel_size=3,
            strides=2,
            padding="same",
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
        blocks = float(sum(args["num_repeat"] for args in blocks_args))

        for i, args in enumerate(blocks_args):
            assert args["num_repeat"] > 0

            # Update block input and output filters based on depth multiplier.
            args["input_filters"] = self.round_filters(
                filters=args["input_filters"],
                width_coefficient=width_coefficient,
                min_depth=min_depth,
                depth_divisor=depth_divisor,
            )
            args["output_filters"] = self.round_filters(
                filters=args["output_filters"],
                width_coefficient=width_coefficient,
                min_depth=min_depth,
                depth_divisor=depth_divisor,
            )

            # Determine which conv type to use:
            block = {0: MBConvBlock, 1: FusedMBConvBlock}[args.pop("conv_type")]
            repeats = self.round_repeats(
                repeats=args.pop("num_repeat"), depth_coefficient=depth_coefficient
            )

            for j in range(repeats):
                if j > 0:
                    args["strides"] = 1
                    args["input_filters"] = args["output_filters"]

                x = block(
                    activation=activation,
                    bn_momentum=bn_momentum,
                    survival_probability=drop_connect_rate * b / blocks,
                    name=f"block{i + 1}{chr(j + 97)}_",
                    **args,
                )(x)

                if args["strides"] != 1:
                    pyramid_outputs[f"P{pyramid_level}"] = x
                    pyramid_level += 1

                b += 1

        # Top
        top_filters = self.round_filters(
            filters=1280,
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=top_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name="top_conv",
        )(x)
        x = layers.BatchNormalization(axis=-1, name="top_bn")(x)
        x = layers.Activation(activation, name="top_activation")(x)
        super().__init__(
            inputs=inputs,
            outputs=x,
            name=name or f"EfficientNetBackboneV2{spatial_dims}D",
            **kwargs,
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

    @staticmethod
    def round_filters(filters, width_coefficient, min_depth, depth_divisor):
        filters *= width_coefficient
        minimum_depth = min_depth or depth_divisor
        new_filters = max(
            minimum_depth,
            int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
        )
        return int(new_filters)

    @staticmethod
    def round_repeats(repeats, depth_coefficient):
        return int(math.ceil(depth_coefficient * repeats))


EfficientNetBackbone_DOCSTRING = """
{name} model supporting both 2D and 3D inputs.

This class implements the backbone part of the EfficientNet architecture,
which scales width, depth, and resolution uniformly using compound scaling.
It can operate on 2D inputs (e.g., images of shape `(H, W, C)`) or 3D inputs
(e.g., volumetric data of shape `(D, H, W, C)`).

The backbone produces multi-scale feature maps that can be used for downstream
tasks such as classification, detection, or segmentation.

References:
    - Tan, M. and Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".
      ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)

Example (2D):
    >>> model = {name}(
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
    >>> model = {name}(
    ...     width_coefficient=1.0,
    ...     depth_coefficient=1.0,
    ...     input_shape=(32, 224, 224, 3),
    ...     include_rescaling=True,
    ... )
    >>> x = tf.random.normal((1, 32, 224, 224, 3))
    >>> y = model(x)
    >>> y.shape
    TensorShape([1, 1, 7, 7, 1280])

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
    blocks_args (list or str, default="{default_blocks}"):
        Configuration for the sequence of MBConv/FusedMBConv blocks.
        - If "default", uses the standard EfficientNet block structure.
        - If a string key (e.g., "efficientnetv2-s"), loads the corresponding preset from
          `DEFAULT_BLOCKS_ARGS_V2`.
        - If a list, expects a custom block configuration dict.
    include_rescaling (bool, default=False):
        If True, includes input rescaling (1/255) and normalization layers.
    name (str, default="{default_name}"):
        Model name.
{extra_args}
"""

# --- version-specific extensions ---
EfficientNetBackboneV1_DOCSTRING = EfficientNetBackbone_DOCSTRING.format(
    name="EfficientNetBackbone",
    default_blocks="default",
    default_name="efficientnet",
    extra_args=""
)

EfficientNetBackboneV2_DOCSTRING = EfficientNetBackbone_DOCSTRING.format(
    name="EfficientNetBackboneV2",
    default_blocks="efficientnetv2-s",
    default_name="efficientnet_v2",
    extra_args="""
    min_depth (int, default=8):
        Minimum number of filters in any layer after scaling.
    bn_momentum (float, default=0.9):
        Momentum used for batch normalization layers.
"""
)

# attach
EfficientNetBackbone.__doc__ = EfficientNetBackboneV1_DOCSTRING
EfficientNetBackboneV2.__doc__ = EfficientNetBackboneV2_DOCSTRING

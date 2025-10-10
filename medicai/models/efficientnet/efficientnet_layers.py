import keras
from keras import layers

from medicai.layers import DropPath
from medicai.utils import get_conv_layer, get_pooling_layer, get_reshaping_layer

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    },
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 1.0 / 3.0,
        "mode": "fan_out",
        "distribution": "uniform",
    },
}


def EfficientNetV1Block(
    activation="swish",
    drop_rate=0.0,
    filters_in=32,
    filters_out=16,
    kernel_size=3,
    strides=1,
    expand_ratio=1,
    se_ratio=0.0,
    id_skip=True,
    name="",
):

    def apply(inputs):
        spatial_dims = len(inputs.shape) - 2

        # Expansion phase
        filters = filters_in * expand_ratio
        if expand_ratio != 1:
            x = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=filters,
                kernel_size=1,
                padding="same",
                use_bias=False,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "expand_conv",
            )(inputs)
            x = layers.BatchNormalization(axis=-1, name=name + "expand_bn")(x)
            x = layers.Activation(activation, name=name + "expand_activation")(x)
        else:
            x = inputs

        # Depthwise Convolution
        if strides == 2:
            x = get_reshaping_layer(
                spatial_dims=spatial_dims,
                layer_type="padding",
                padding=correct_pad(x, kernel_size),
                name=name + "dwconv_pad",
            )(x)
            conv_pad = "valid"
        else:
            conv_pad = "same"

        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="depthwise_conv",
            kernel_size=kernel_size,
            strides=strides,
            padding=conv_pad,
            use_bias=False,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "dwconv",
        )(x)
        x = layers.BatchNormalization(axis=-1, name=name + "bn")(x)
        x = layers.Activation(activation, name=name + "activation")(x)

        # Squeeze and Excitation phase
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(filters_in * se_ratio))
            se = get_pooling_layer(
                spatial_dims=spatial_dims,
                layer_type="avg",
                global_pool=True,
                name=name + "se_squeeze",
            )(x)
            se_shape = (1,) * spatial_dims + (filters,)
            se = layers.Reshape(se_shape, name=name + "se_reshape")(se)
            se = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=filters_se,
                kernel_size=1,
                padding="same",
                activation=activation,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_reduce",
            )(se)
            se = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=filters,
                kernel_size=1,
                padding="same",
                activation="sigmoid",
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_expand",
            )(se)
            x = layers.multiply([x, se], name=name + "se_excite")

        # Output phase
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=filters_out,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "project_conv",
        )(x)
        x = layers.BatchNormalization(axis=-1, name=name + "project_bn")(x)

        if id_skip and strides == 1 and filters_in == filters_out:
            if drop_rate > 0:
                x = DropPath(rate=drop_rate, name=name + "drop")(x)
            x = layers.add([x, inputs], name=name + "add")

        return x

    return apply


def MBConvBlock(
    input_filters,
    output_filters,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    bn_momentum=0.9,
    activation="swish",
    drop_rate=0.0,
    name=None,
):
    if name is None:
        name = keras.backend.get_uid("block0")

    def apply(inputs):
        # ndim
        spatial_dims = len(inputs.shape) - 2

        # Expansion phase
        filters = input_filters * expand_ratio
        if expand_ratio != 1:
            x = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=filters,
                kernel_size=1,
                strides=1,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                padding="same",
                use_bias=False,
                name=name + "expand_conv",
            )(inputs)
            x = layers.BatchNormalization(
                axis=-1,
                momentum=bn_momentum,
                name=name + "expand_bn",
            )(x)
            x = layers.Activation(activation, name=name + "expand_activation")(x)
        else:
            x = inputs

        # Depthwise conv
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="depthwise_conv",
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "dwconv",
        )(x)
        x = layers.BatchNormalization(axis=-1, momentum=bn_momentum, name=name + "bn")(x)
        x = layers.Activation(activation, name=name + "activation")(x)

        # Squeeze and excite
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(input_filters * se_ratio))
            se = get_pooling_layer(
                spatial_dims=spatial_dims,
                layer_type="avg",
                global_pool=True,
                name=name + "se_squeeze",
            )(x)
            se_shape = (1,) * spatial_dims + (filters,)
            se = layers.Reshape(se_shape, name=name + "se_reshape")(se)
            se = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=filters_se,
                kernel_size=1,
                padding="same",
                activation=activation,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_reduce",
            )(se)
            se = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=filters,
                kernel_size=1,
                padding="same",
                activation="sigmoid",
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_expand",
            )(se)
            x = layers.multiply([x, se], name=name + "se_excite")

        # Output phase
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=output_filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name=name + "project_conv",
        )(x)

        x = layers.BatchNormalization(axis=-1, momentum=bn_momentum, name=name + "project_bn")(x)

        if strides == 1 and input_filters == output_filters:
            if drop_rate:
                x = DropPath(rate=drop_rate, name=name + "drop")(x)
            x = layers.add([x, inputs], name=name + "add")

        return x

    return apply


def FusedMBConvBlock(
    input_filters,
    output_filters,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    bn_momentum=0.9,
    activation="swish",
    drop_rate=0.0,
    name=None,
):

    if name is None:
        name = keras.backend.get_uid("block0")

    def apply(inputs):
        # ndim
        spatial_dims = len(inputs.shape) - 2

        filters = input_filters * expand_ratio
        if expand_ratio != 1:
            x = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                padding="same",
                use_bias=False,
                name=name + "expand_conv",
            )(inputs)
            x = layers.BatchNormalization(axis=-1, momentum=bn_momentum, name=name + "expand_bn")(x)
            x = layers.Activation(activation=activation, name=name + "expand_activation")(x)
        else:
            x = inputs

        # Squeeze and excite
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(input_filters * se_ratio))
            se = get_pooling_layer(
                spatial_dims=spatial_dims,
                layer_type="avg",
                global_pool=True,
                name=name + "se_squeeze",
            )(x)

            se_shape = (1,) * spatial_dims + (filters,)
            se = layers.Reshape(se_shape, name=name + "se_reshape")(se)
            se = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=filters_se,
                kernel_size=1,
                padding="same",
                activation=activation,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_reduce",
            )(se)
            se = get_conv_layer(
                spatial_dims=spatial_dims,
                layer_type="conv",
                filters=filters,
                kernel_size=1,
                padding="same",
                activation="sigmoid",
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_expand",
            )(se)
            x = layers.multiply([x, se], name=name + "se_excite")

        # Output phase:
        x = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=output_filters,
            kernel_size=1 if expand_ratio != 1 else kernel_size,
            strides=1 if expand_ratio != 1 else strides,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name=name + "project_conv",
        )(x)
        x = layers.BatchNormalization(axis=-1, momentum=bn_momentum, name=name + "project_bn")(x)
        if expand_ratio == 1:
            x = layers.Activation(activation=activation, name=name + "project_activation")(x)

        # Residual:
        if strides == 1 and input_filters == output_filters:
            if drop_rate:
                x = DropPath(
                    drop_rate,
                    name=name + "drop",
                )(x)
            x = layers.add([x, inputs], name=name + "add")
        return x

    return apply


def correct_pad(inputs, kernel_size):
    # Number of spatial dimensions (exclude batch and channel)
    spatial_dims = len(inputs.shape) - 2

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * spatial_dims
    else:
        kernel_size = tuple(kernel_size)

    # Extract input spatial dimensions (may include None)
    input_size = inputs.shape[1 : 1 + spatial_dims]

    paddings = []
    for i, dim in enumerate(input_size):
        if dim is None:
            adjust = 1
        else:
            adjust = 1 - dim % 2
        correct = kernel_size[i] // 2
        paddings.append((correct - adjust, correct))

    return tuple(paddings)


DEFAULT_BLOCKS_ARGS_V1 = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
]

DEFAULT_BLOCKS_ARGS_V2 = {
    "efficientnetv2-s": [
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0.0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0.0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "conv_type": 1,
            "expand_ratio": 4,
            "input_filters": 48,
            "kernel_size": 3,
            "num_repeat": 4,
            "output_filters": 64,
            "se_ratio": 0,
            "strides": 2,
        },
        {
            "conv_type": 0,
            "expand_ratio": 4,
            "input_filters": 64,
            "kernel_size": 3,
            "num_repeat": 6,
            "output_filters": 128,
            "se_ratio": 0.25,
            "strides": 2,
        },
        {
            "conv_type": 0,
            "expand_ratio": 6,
            "input_filters": 128,
            "kernel_size": 3,
            "num_repeat": 9,
            "output_filters": 160,
            "se_ratio": 0.25,
            "strides": 1,
        },
        {
            "conv_type": 0,
            "expand_ratio": 6,
            "input_filters": 160,
            "kernel_size": 3,
            "num_repeat": 15,
            "output_filters": 256,
            "se_ratio": 0.25,
            "strides": 2,
        },
    ],
    "efficientnetv2-m": [
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 48,
            "output_filters": 80,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 80,
            "output_filters": 160,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 14,
            "input_filters": 160,
            "output_filters": 176,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 18,
            "input_filters": 176,
            "output_filters": 304,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 304,
            "output_filters": 512,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-l": [
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 32,
            "output_filters": 32,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 32,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 64,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 10,
            "input_filters": 96,
            "output_filters": 192,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 19,
            "input_filters": 192,
            "output_filters": 224,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 25,
            "input_filters": 224,
            "output_filters": 384,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 384,
            "output_filters": 640,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b0": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b1": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b2": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b3": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
}

from keras import layers

DEFAULT_BLOCKS_ARGS = [
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


def EfficientNetBlock(
    activation="swish",
    drop_rate=0.0,
    name="",
    filters_in=32,
    filters_out=16,
    kernel_size=3,
    strides=1,
    expand_ratio=1,
    se_ratio=0.0,
    id_skip=True,
):

    def apply(inputs):

        # Expansion phase
        filters = filters_in * expand_ratio
        if expand_ratio != 1:
            x = layers.Conv2D(
                filters,
                1,
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
            x = layers.ZeroPadding2D(
                padding=correct_pad(x, kernel_size),
                name=name + "dwconv_pad",
            )(x)
            conv_pad = "valid"
        else:
            conv_pad = "same"
        x = layers.DepthwiseConv2D(
            kernel_size,
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
            se = layers.GlobalAveragePooling2D(name=name + "se_squeeze")(x)
            se_shape = (1, 1, filters)
            se = layers.Reshape(se_shape, name=name + "se_reshape")(se)
            se = layers.Conv2D(
                filters_se,
                1,
                padding="same",
                activation=activation,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_reduce",
            )(se)
            se = layers.Conv2D(
                filters,
                1,
                padding="same",
                activation="sigmoid",
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_expand",
            )(se)
            x = layers.multiply([x, se], name=name + "se_excite")

        # Output phase
        x = layers.Conv2D(
            filters_out,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "project_conv",
        )(x)
        x = layers.BatchNormalization(axis=-1, name=name + "project_bn")(x)
        if id_skip and strides == 1 and filters_in == filters_out:
            if drop_rate > 0:
                x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=name + "drop")(x)
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

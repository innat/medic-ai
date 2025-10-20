from keras import layers

from medicai.utils import get_conv_layer, get_pooling_layer, get_reshaping_layer


def apply_resnet_basic_block(
    x,
    filters,
    kernel_size=3,
    stride=1,
    conv_shortcut=False,
    use_pre_activation=False,
    groups=1,  # for ResNeXt
    width_per_group=64,  # for ResNeXt
    name=None,
):
    """Applies a basic residual block.

    Args:
        x: Tensor. The input tensor to pass through the block.
        filters: int. The number of filters in the block.
        kernel_size: int. The kernel size of the bottleneck layer. Defaults to
            `3`.
        stride: int. The stride length of the first layer. Defaults to `1`.
        conv_shortcut: bool. If `True`, use a convolution shortcut. If `False`,
            use an identity or pooling shortcut based on the stride. Defaults to
            `False`.
        use_pre_activation: boolean. Whether to use pre-activation or not.
            `True` for ResNetV2, `False` for ResNet. Defaults to `False`.
        groups: int. Number of groups for grouped convolution. Defaults to 1.
        width_per_group: int. Bottleneck width for ResNeXt. Defaults to 64.
        name: str. A prefix for the layer names used in the block.

    Returns:
        The output tensor for the basic residual block.
    """
    spatial_dims = len(x.shape) - 2

    # Calculate bottleneck width for ResNeXt
    width = int(filters * (width_per_group / 64)) * groups

    x_preact = None
    if use_pre_activation:
        x_preact = layers.BatchNormalization(
            axis=-1,
            epsilon=1e-5,
            momentum=0.9,
            name=f"{name}_pre_activation_bn",
        )(x)
        x_preact = layers.Activation("relu", name=f"{name}_pre_activation_relu")(x_preact)

    if conv_shortcut:
        if x_preact is not None:
            shortcut = x_preact
        else:
            shortcut = x

        shortcut = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=filters,
            kernel_size=1,
            strides=stride,
            use_bias=False,
            name=f"{name}_0_conv",
        )(shortcut)

        if not use_pre_activation:
            shortcut = layers.BatchNormalization(
                axis=-1,
                epsilon=1e-5,
                momentum=0.9,
                name=f"{name}_0_bn",
            )(shortcut)
    else:
        shortcut = x

    x = x_preact if x_preact is not None else x
    if stride > 1:
        x = get_reshaping_layer(
            spatial_dims=spatial_dims,
            layer_type="padding",
            padding=(kernel_size - 1) // 2,
            name=f"{name}_1_pad",
        )(x)

    x = get_conv_layer(
        spatial_dims=spatial_dims,
        layer_type="conv",
        filters=width,
        kernel_size=kernel_size,
        padding="valid" if stride > 1 else "same",
        strides=stride,
        groups=groups,
        use_bias=False,
        name=f"{name}_1_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-5,
        momentum=0.9,
        name=f"{name}_1_bn",
    )(x)
    x = layers.Activation("relu", name=f"{name}_1_relu")(x)

    x = get_conv_layer(
        spatial_dims=spatial_dims,
        layer_type="conv",
        filters=width,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        groups=groups,
        use_bias=False,
        name=f"{name}_2_conv",
    )(x)

    if not use_pre_activation:
        x = layers.BatchNormalization(
            axis=-1,
            epsilon=1e-5,
            momentum=0.9,
            name=f"{name}_2_bn",
        )(x)
        x = layers.Add(name=f"{name}_add")([shortcut, x])
        x = layers.Activation("relu", name=f"{name}_out")(x)
    else:
        x = layers.Add(name=f"{name}_out")([shortcut, x])
    return x


def apply_resnet_bottleneck_block(
    x,
    filters,
    kernel_size=3,
    stride=1,
    conv_shortcut=False,
    use_pre_activation=False,
    groups=1,  # for ResNeXt
    width_per_group=64,  # for ResNeXt
    name=None,
):
    """Applies a bottleneck residual block.

    Args:
        x: Tensor. The input tensor to pass through the block.
        filters: int. The number of filters in the block.
        kernel_size: int. The kernel size of the bottleneck layer. Defaults to
            `3`.
        stride: int. The stride length of the first layer. Defaults to `1`.
        conv_shortcut: bool. If `True`, use a convolution shortcut. If `False`,
            use an identity or pooling shortcut based on the stride. Defaults to
            `False`.
        use_pre_activation: boolean. Whether to use pre-activation or not.
            `True` for ResNetV2, `False` for ResNet. Defaults to `False`.
        groups: int. Number of groups for grouped convolution. Defaults to 1.
        width_per_group: int. Bottleneck width for ResNeXt. Defaults to 64.
        name: str. A prefix for the layer names used in the block.

    Returns:
        The output tensor for the residual block.
    """

    spatial_dims = len(x.shape) - 2

    # Calculate bottleneck width for ResNeXt
    width = int(filters * (width_per_group / 64)) * groups

    x_preact = None
    if use_pre_activation:
        x_preact = layers.BatchNormalization(
            axis=-1,
            epsilon=1e-5,
            momentum=0.9,
            name=f"{name}_pre_activation_bn",
        )(x)
        x_preact = layers.Activation("relu", name=f"{name}_pre_activation_relu")(x_preact)

    if conv_shortcut:
        if x_preact is not None:
            shortcut = x_preact
        else:
            shortcut = x

        shortcut = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=4 * filters,
            kernel_size=1,
            strides=stride,
            use_bias=False,
            name=f"{name}_0_conv",
        )(shortcut)

        if not use_pre_activation:
            shortcut = layers.BatchNormalization(
                axis=-1,
                epsilon=1e-5,
                momentum=0.9,
                name=f"{name}_0_bn",
            )(shortcut)
    else:
        shortcut = x

    x = x_preact if x_preact is not None else x
    x = get_conv_layer(
        spatial_dims=spatial_dims,
        layer_type="conv",
        filters=width,
        kernel_size=1,
        strides=1,
        use_bias=False,
        name=f"{name}_1_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-5,
        momentum=0.9,
        name=f"{name}_1_bn",
    )(x)
    x = layers.Activation("relu", name=f"{name}_1_relu")(x)

    if stride > 1:
        x = get_reshaping_layer(
            spatial_dims=spatial_dims,
            layer_type="padding",
            padding=(kernel_size - 1) // 2,
            name=f"{name}_2_pad",
        )(x)

    x = get_conv_layer(
        spatial_dims=spatial_dims,
        layer_type="conv",
        filters=width,
        kernel_size=kernel_size,
        strides=stride,
        groups=groups,
        padding="valid" if stride > 1 else "same",
        use_bias=False,
        name=f"{name}_2_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-5,
        momentum=0.9,
        name=f"{name}_2_bn",
    )(x)
    x = layers.Activation("relu", name=f"{name}_2_relu")(x)

    x = get_conv_layer(
        spatial_dims=spatial_dims,
        layer_type="conv",
        filters=4 * filters,
        kernel_size=1,
        use_bias=False,
        name=f"{name}_3_conv",
    )(x)

    if not use_pre_activation:
        x = layers.BatchNormalization(
            axis=-1,
            epsilon=1e-5,
            momentum=0.9,
            name=f"{name}_3_bn",
        )(x)
        x = layers.Add(name=f"{name}_add")([shortcut, x])
        x = layers.Activation("relu", name=f"{name}_out")(x)
    else:
        x = layers.Add(name=f"{name}_out")([shortcut, x])
    return x


def apply_bottleneck_block_vd(
    x,
    filters,
    kernel_size=3,
    stride=1,
    conv_shortcut=False,
    use_pre_activation=False,
    groups=1,
    width_per_group=64,
    name=None,
):
    """Applies a bottleneck residual block.

    Args:
        x: Tensor. The input tensor to pass through the block.
        filters: int. The number of filters in the block.
        kernel_size: int. The kernel size of the bottleneck layer. Defaults to
            `3`.
        stride: int. The stride length of the first layer. Defaults to `1`.
        conv_shortcut: bool. If `True`, use a convolution shortcut. If `False`,
            use an identity or pooling shortcut based on the stride. Defaults to
            `False`.
        use_pre_activation: boolean. Whether to use pre-activation or not.
            `True` for ResNetV2, `False` for ResNet. Defaults to `False`.
        groups: int. Number of groups for grouped convolution. Defaults to 1.
        width_per_group: int. Bottleneck width for ResNeXt. Defaults to 64.
        name: str. A prefix for the layer names used in the block.

    Returns:
        The output tensor for the residual block.
    """
    x_preact = None
    spatial_dims = len(x.shape) - 2

    # Calculate bottleneck width for ResNeXt
    width = int(filters * (width_per_group / 64)) * groups

    if use_pre_activation:
        x_preact = layers.BatchNormalization(
            axis=-1,
            epsilon=1e-5,
            momentum=0.9,
            name=f"{name}_pre_activation_bn",
        )(x)
        x_preact = layers.Activation("relu", name=f"{name}_pre_activation_relu")(x_preact)

    if conv_shortcut:
        if x_preact is not None:
            shortcut = x_preact
        elif stride > 1:
            shortcut = get_pooling_layer(
                spatial_dims=spatial_dims,
                layer_type="avg",
                pool_size=2,
                strides=stride,
                padding="same",
            )(x)
        else:
            shortcut = x

        shortcut = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=4 * filters,
            kernel_size=1,
            strides=1,
            use_bias=False,
            name=f"{name}_0_conv",
        )(shortcut)

        if not use_pre_activation:
            shortcut = layers.BatchNormalization(
                axis=-1,
                epsilon=1e-5,
                momentum=0.9,
                name=f"{name}_0_bn",
            )(shortcut)
    else:
        shortcut = x

    x = x_preact if x_preact is not None else x

    x = get_conv_layer(
        spatial_dims=spatial_dims,
        layer_type="conv",
        filters=width,
        kernel_size=1,
        strides=1,
        use_bias=False,
        name=f"{name}_1_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-5,
        momentum=0.9,
        name=f"{name}_1_bn",
    )(x)
    x = layers.Activation("relu", name=f"{name}_1_relu")(x)

    if stride > 1:
        x = get_reshaping_layer(
            spatial_dims=spatial_dims,
            layer_type="padding",
            padding=(kernel_size - 1) // 2,
            name=f"{name}_2_pad",
        )(x)

    x = get_conv_layer(
        spatial_dims=spatial_dims,
        layer_type="conv",
        filters=width,
        kernel_size=kernel_size,
        strides=stride,
        groups=groups,
        padding="valid" if stride > 1 else "same",
        use_bias=False,
        name=f"{name}_2_conv",
    )(x)

    x = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-5,
        momentum=0.9,
        name=f"{name}_2_bn",
    )(x)
    x = layers.Activation("relu", name=f"{name}_2_relu")(x)

    x = get_conv_layer(
        spatial_dims=spatial_dims,
        layer_type="conv",
        filters=4 * filters,
        kernel_size=1,
        use_bias=False,
        name=f"{name}_3_conv",
    )(x)

    if not use_pre_activation:
        x = layers.BatchNormalization(
            axis=-1,
            epsilon=1e-5,
            momentum=0.9,
            name=f"{name}_3_bn",
        )(x)
        x = layers.Add(name=f"{name}_add")([shortcut, x])
        x = layers.Activation("relu", name=f"{name}_out")(x)
    else:
        x = layers.Add(name=f"{name}_out")([shortcut, x])
    return x


def apply_resnet_block(
    x,
    filters,
    blocks,
    stride,
    block_type,
    use_pre_activation,
    first_shortcut=True,
    groups=1,
    width_per_group=64,
    name=None,
):
    """Applies a set of stacked residual blocks.

    Args:
        x: Tensor. The input tensor to pass through the stack.
        filters: int. The number of filters in a block.
        blocks: int. The number of blocks in the stack.
        stride: int. The stride length of the first layer in the first block.
        block_type: str. The block type to stack. One of `"basic_block"`,
            `"bottleneck_block"`, or `"bottleneck_block_vd"`. Use `"basic_block"`
            for ResNet18 and ResNet34. Use `"bottleneck_block"` for ResNet50, ResNet101 and
            ResNet152. Use `"bottleneck_block_vd"` for the `_vd`
            variants.
        use_pre_activation: boolean. Whether to use pre-activation or not.
            `True` for ResNetV2, `False` for ResNet and ResNeXt.
        first_shortcut: bool. If `True`, use a convolution shortcut. If `False`,
            use an identity or pooling shortcut based on the stride. Defaults to
            `True`.
        groups: int. Number of groups for grouped convolution. Defaults to 1.
        width_per_group: int. Bottleneck width for ResNeXt. Defaults to 64.
        name: str. A prefix for the layer names used in the stack.

    Returns:
        Output tensor for the stacked blocks.
    """
    if name is None:
        name = "stack"

    block_type_map = {
        "basic_block": apply_resnet_basic_block,
        "bottleneck_block": apply_resnet_bottleneck_block,
        "bottleneck_block_vd": apply_bottleneck_block_vd,
    }

    block_fn = block_type_map.get(block_type, None)
    if block_fn is None:
        raise ValueError(
            f"`block_type` must be one of {list(block_type_map.keys())}, "
            f"but received block_type={block_type}."
        )

    for i in range(blocks):
        if i == 0:
            stride = stride
            conv_shortcut = first_shortcut
        else:
            stride = 1
            conv_shortcut = False

        x = block_fn(
            x,
            filters,
            stride=stride,
            conv_shortcut=conv_shortcut,
            use_pre_activation=use_pre_activation,
            groups=groups,
            width_per_group=width_per_group,
            name=f"{name}_block{str(i)}",
        )
    return x

from medicai.utils.general import hide_warnings

hide_warnings()

import keras
from keras import activations, layers

BACKBONE_ZOO = {}


def get_conv_layer(spatial_dims: int, layer_type: str, **kwargs):
    """
    Returns a convolutional layer (2D or 3D) based on the given type.

    Args:
        spatial_dims (int): Number of spatial dimensions. Must be 2 or 3.
        layer_type (str): One of {'conv', 'conv_transpose', 'separable_conv', 'depthwise_conv'}.
        **kwargs: Additional keyword arguments for the layer (filters, kernel_size, strides, etc.).

    Returns:
        keras.layers.Layer: A Keras convolutional layer.

    Raises:
        ValueError: If spatial_dims is not in {2, 3} or if an unsupported layer_type is requested.
    """
    SUPPORTED_TYPES = {"conv", "conv_transpose", "separable_conv", "depthwise_conv"}

    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

    layer_type = layer_type.lower()

    # Validation rules
    if layer_type in {"conv", "conv_transpose", "separable_conv"}:
        if "filters" not in kwargs:
            raise ValueError(f"'{layer_type}' requires a 'filters' argument.")
    if layer_type == "depthwise_conv":
        if "filters" in kwargs:
            raise ValueError(
                "'depthwise_conv' does not accept 'filters' (use 'kernel_size', 'depth_multiplier', etc.)."
            )

    # Dispatch
    if layer_type == "conv":
        ConvClass = {2: layers.Conv2D, 3: layers.Conv3D}[spatial_dims]

    elif layer_type == "conv_transpose":
        ConvClass = {2: layers.Conv2DTranspose, 3: layers.Conv3DTranspose}[spatial_dims]

    elif layer_type == "separable_conv":
        if spatial_dims == 2:
            ConvClass = layers.SeparableConv2D
        else:
            raise ValueError("SeparableConv is only available in 2D.")

    elif layer_type == "depthwise_conv":
        if spatial_dims == 2:
            ConvClass = layers.DepthwiseConv2D
        else:
            raise ValueError("DepthwiseConv is only available in 2D.")

    else:
        raise ValueError(
            f"Unsupported layer_type: '{layer_type}'. "
            f"Supported types are: {sorted(SUPPORTED_TYPES)}"
        )

    return ConvClass(**kwargs)


def get_reshaping_layer(spatial_dims, layer_type, **kwargs):
    """Returns a reshaping layer (UpSampling or ZeroPadding)
    for 2D or 3D inputs.

    Args:
        spatial_dims (int): 2 or 3, determines if 2D or 3D layer is used.
        layer_type (str): "upsampling" or "padding".
        **kwargs: Additional arguments passed to the selected layer.
    """
    assert spatial_dims in (2, 3), "spatial_dims must be 2 or 3"
    assert layer_type in (
        "upsampling",
        "padding",
        "cropping",
    ), "layer_type must be 'upsampling' or 'padding' or 'cropping'"

    layers_map = {
        "upsampling": {2: layers.UpSampling2D, 3: layers.UpSampling3D},
        "padding": {2: layers.ZeroPadding2D, 3: layers.ZeroPadding3D},
        "cropping": {2: layers.Cropping2D, 3: layers.Cropping3D},
    }

    LayerClass = layers_map[layer_type][spatial_dims]
    return LayerClass(**kwargs)


def get_pooling_layer(spatial_dims, layer_type, global_pool=False, **kwargs):
    """
    Returns a pooling layer (Max or Average) for 1D, 2D, or 3D inputs, including global pooling.

    Args:
        spatial_dims (int): Number of spatial dimensions. Must be 1, 2, or 3.
        layer_type (str): Type of pooling. Must be "max" or "avg".
        global_pool (bool): If True, returns a Global pooling layer (GlobalMaxPooling or GlobalAveragePooling).
                            If False, returns regular pooling (MaxPooling or AveragePooling).
        **kwargs: Additional keyword arguments passed to the layer constructor
                  (e.g., pool_size, strides, padding).

    Returns:
        keras.layers.Layer: The corresponding Keras pooling layer.

    Example:
        # 2D max pooling
        pool2d = get_pooling_layer(2, "max", pool_size=(2, 2), strides=(2, 2))

        # Global average pooling 3D
        gap3d = get_pooling_layer(3, "avg", global_pool=True)
    """
    assert spatial_dims in (2, 3), "spatial_dims must be 2, or 3"
    assert layer_type in ("max", "avg"), "pool_type must be 'max' or 'avg'"

    if global_pool:
        layers_map = {
            "max": {2: layers.GlobalMaxPooling2D, 3: layers.GlobalMaxPooling3D},
            "avg": {2: layers.GlobalAveragePooling2D, 3: layers.GlobalAveragePooling3D},
        }
    else:
        layers_map = {
            "max": {2: layers.MaxPooling2D, 3: layers.MaxPooling3D},
            "avg": {2: layers.AveragePooling2D, 3: layers.AveragePooling3D},
        }

    LayerClass = layers_map[layer_type][spatial_dims]
    return LayerClass(**kwargs)


def get_act_layer(name, **kwargs):
    """
    Returns a Keras activation layer based on the provided name and keyword arguments
    using the official keras.activations.get() function.

    Args:
        name (str): The name of the activation function (e.g., 'relu', 'sigmoid', 'leaky_relu').
                     Can also be a callable activation function.
        **kwargs: Keyword arguments to be passed to the activation function (if applicable).

    Returns:
        A Keras Activation layer.
    """
    name = name.lower()
    if name == "leaky_relu":
        return layers.LeakyReLU(**kwargs)
    elif name == "prelu":
        return layers.PReLU(**kwargs)
    elif name == "elu":
        return layers.ELU(**kwargs)
    elif name == "relu":
        return layers.ReLU(**kwargs)
    else:
        activation_fn = activations.get(name)
        return layers.Activation(activation_fn)


def get_norm_layer(norm_name, **kwargs):
    """
    Returns a Keras normalization layer based on the provided name and keyword arguments.

    Args:
        norm_name (str): The name of the normalization layer to create.
                           Supported names are: "instance", "batch", "layer", "unit", "group".
        **kwargs: Keyword arguments to be passed to the constructor of the
                  chosen normalization layer.

    Returns:
        A Keras normalization layer instance.

    Raises:
        ValueError: If an unsupported `norm_name` is provided.

    Examples:
        >>> batch_norm = get_norm_layer("batch", momentum=0.9)
        >>> isinstance(batch_norm, layers.BatchNormalization)
        True
        >>> instance_norm = get_norm_layer("instance")
        >>> isinstance(instance_norm, layers.GroupNormalization)
        True
        >>> try:
        ...     unknown_norm = get_norm_layer("unknown")
        ... except ValueError as e:
        ...     print(e)
        Unsupported normalization: unknown
    """
    norm_name = norm_name.lower()
    if norm_name == "instance":
        return layers.GroupNormalization(groups=-1, epsilon=1e-05, scale=False, center=False)

    elif norm_name == "batch":
        return layers.BatchNormalization(**kwargs)

    elif norm_name == "layer":
        return layers.LayerNormalization(**kwargs)

    elif norm_name == "unit":
        return layers.UnitNormalization(**kwargs)

    elif norm_name == "group":
        return layers.GroupNormalization(**kwargs)
    else:
        raise ValueError(f"Unsupported normalization: {norm_name}")


def parse_model_inputs(input_shape, input_tensor=None, **kwargs):
    if input_tensor is None:
        return keras.layers.Input(shape=input_shape, **kwargs)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            return keras.layers.Input(tensor=input_tensor, shape=input_shape, **kwargs)
        else:
            return input_tensor

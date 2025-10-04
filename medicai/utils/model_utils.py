from medicai.utils.general import hide_warnings

hide_warnings()

import keras
from keras import activations, layers

from .registry import registration

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
            from medicai.layers import DepthwiseConv3D

            ConvClass = DepthwiseConv3D

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


def get_act_layer(layer_type, **kwargs):
    """
    Returns a Keras activation layer based on the provided layer_type and keyword arguments
    using the official keras.activations.get() function.

    Args:
        layer_type (str): The name of the activation function (e.g., 'relu', 'sigmoid', 'leaky_relu').
            Can also be a callable activation function.
        **kwargs: Keyword arguments to be passed to the activation function (if applicable).

    Returns:
        A Keras Activation layer.
    """
    # Normalize name
    layer_type = layer_type.lower()

    # Map of special activations that need dedicated layer classes
    special_layers = {
        "leaky_relu": layers.LeakyReLU,
        "prelu": layers.PReLU,
        "elu": layers.ELU,
        "relu": layers.ReLU,
    }

    if layer_type in special_layers:
        return special_layers[layer_type](**kwargs)

    # Fallback to standard keras.activations.get
    activation_fn = activations.get(layer_type)
    return layers.Activation(activation_fn, **kwargs)


def get_norm_layer(layer_type, **kwargs):
    """
    Returns a Keras normalization layer based on the provided name and keyword arguments.

    Args:
        layer_type (str): The name of the normalization layer to create.
            Supported names are: "instance", "batch", "layer", "unit", "group".
        **kwargs: Keyword arguments to be passed to the constructor of the
            chosen normalization layer.

    Returns:
        A Keras normalization layer instance.

    Raises:
        ValueError: If an unsupported `layer_type` is provided.

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
    layer_type = layer_type.lower()

    if layer_type == "instance":
        # Instance normalization is implemented via GroupNormalization with specific settings.
        # Note: Custom kwargs are ignored for this layer type.
        return layers.GroupNormalization(groups=-1, epsilon=1e-05, scale=False, center=False)

    # Lookup table for other normalizations
    norm_layers = {
        "batch": layers.BatchNormalization,
        "layer": layers.LayerNormalization,
        "unit": layers.UnitNormalization,
        "group": layers.GroupNormalization,
    }

    if layer_type not in norm_layers:
        raise ValueError(f"Unsupported normalization: {layer_type}")

    layer_cls = norm_layers[layer_type]
    return layer_cls(**kwargs)


def parse_model_inputs(input_shape, input_tensor=None, **kwargs):
    if input_tensor is None:
        return keras.layers.Input(shape=input_shape, **kwargs)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            return keras.layers.Input(tensor=input_tensor, shape=input_shape, **kwargs)
        else:
            return input_tensor


def resolve_encoder(encoder, encoder_name, input_shape, allowed_families, **kwargs):
    """
    Initializes and validates the backbone encoder for a segmentation model.

    This function handles the logic for choosing between an instantiated
    encoder object or a registered encoder name, performs input validation,
    and ensures the selected encoder is compatible with the model's requirements.

    Args:
        encoder (keras.Model, optional): An already instantiated encoder model.
        encoder_name (str, optional): The name of a registered encoder to load.
        input_shape (tuple, optional): The shape of the input data. Required
            if `encoder_name` is provided.
        allowed_families (list): A list of strings representing the allowed
            backbone families for the current model.

    Returns:
        tuple: A tuple containing the instantiated encoder model and the
              derived or validated input shape.

    Raises:
        ValueError: If the arguments are invalid or the encoder is incompatible.
        AttributeError: If the encoder does not have a `pyramid_outputs` attribute.
    """
    if bool(encoder) == bool(encoder_name):
        raise ValueError("Exactly one of `encoder` or `encoder_name` must be provided.")

    if encoder is not None:
        input_shape = encoder.input_shape[1:]
    elif encoder_name is not None:
        if not input_shape:
            raise ValueError(
                "Argument `input_shape` must be provided. "
                "It should be a tuple of integers specifying the dimensions of the input "
                "data, not including the batch size. "
                "For 2D data, the format is `(height, width, channels)`. "
                "For 3D data, the format is `(depth, height, width, channels)`."
            )

        if encoder_name.lower() not in registration._registry:
            raise ValueError(
                f"Encoder '{encoder_name}' not found in the registry. "
                f"Available: {list(registration._registry.keys())}"
            )

        entry = registration.get_entry(encoder_name)
        invalid_families = [f for f in entry["family"] if f not in allowed_families]
        if invalid_families:
            raise ValueError(
                f"The provided encoder_name='{encoder_name}' uses unsupported families: "
                f"{invalid_families}. Allowed families: {allowed_families}"
            )

        encoder = entry["class"](input_shape=input_shape, include_top=False, **kwargs)

    if not hasattr(encoder, "pyramid_outputs"):
        raise AttributeError(
            f"The provided `encoder` must have a `pyramid_outputs` attribute, "
            f"but the provided encoder of type {type(encoder).__name__} does not."
        )

    return encoder, input_shape

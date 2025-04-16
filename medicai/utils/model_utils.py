from medicai.utils.general import hide_warnings

hide_warnings()

from keras import activations, layers


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

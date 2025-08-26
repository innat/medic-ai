from functools import partial

from medicai.utils.general import hide_warnings

hide_warnings()

import keras
from keras import activations, layers


def get_conv_layer(spatial_dims, transpose=False, **kwargs):
    """Returns a convolutional layer (2D or 3D) or its transposed version.

    Args:
        spatial_dims (int): Number of spatial dimensions. Must be 2 or 3.
        transpose (bool): If True, returns a ConvNDTranspose layer
                          (Conv2DTranspose or Conv3DTranspose).
                          If False, returns a standard ConvND layer.
        **kwargs: Additional keyword arguments passed to the layer constructor
                  (e.g., filters, kernel_size, strides, padding, activation).

    Returns:
        tf.keras.layers.Layer: The corresponding convolutional or transposed convolutional layer.

    Example:
        conv2d = get_conv_layer(2, filters=32, kernel_size=3, padding='same')
        conv3d_t = get_conv_layer(3, transpose=True, filters=16, kernel_size=3, strides=2)
    """
    ConvND = {
        2: layers.Conv2D,
        3: layers.Conv3D,
    }
    ConvNDTranspose = {
        2: layers.Conv2DTranspose,
        3: layers.Conv3DTranspose,
    }

    ConvClass = ConvNDTranspose[spatial_dims] if transpose else ConvND[spatial_dims]
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
    assert layer_type in ("upsampling", "padding"), "layer_type must be 'upsampling' or 'padding'"

    layers_map = {
        "upsampling": {2: layers.UpSampling2D, 3: layers.UpSampling3D},
        "padding": {2: layers.ZeroPadding2D, 3: layers.ZeroPadding3D},
    }

    LayerClass = layers_map[layer_type][spatial_dims]
    return LayerClass(**kwargs)


def get_pooling_layer(spatial_dims, pool_type, global_pool=False, **kwargs):
    """
    Returns a pooling layer (Max or Average) for 1D, 2D, or 3D inputs, including global pooling.

    Args:
        spatial_dims (int): Number of spatial dimensions. Must be 1, 2, or 3.
        pool_type (str): Type of pooling. Must be "max" or "avg".
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
        gap3d = get_pooling_layer(3, "average", global_pool=True)
    """
    assert spatial_dims in (2, 3), "spatial_dims must be 1, 2, or 3"
    assert pool_type in ("max", "avg"), "pool_type must be 'max' or 'average'"

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

    LayerClass = layers_map[pool_type][spatial_dims]
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


BACKBONE_ARGS = {
    "densenet121": [6, 12, 24, 16],
    "densenet169": [6, 12, 32, 32],
    "densenet201": [6, 12, 48, 32],
}

SKIP_CONNECTION_ARGS = {
    "densenet121": [309, 137, 49, 3],
    "densenet169": [365, 137, 49, 3],
    "densenet201": [477, 137, 49, 3],
}

BACKBONE_ZOO = {}

KERAS_APPLICATION = {
    "densenet121": partial(keras.applications.DenseNet121, weights=None),
    "densenet169": partial(keras.applications.DenseNet169, weights=None),
    "densenet201": partial(keras.applications.DenseNet201, weights=None),
}

from functools import partial

import keras
from keras import activations, layers

from .constant import keras_constants
from .registry import registration


def get_conv_layer(spatial_dims: int, layer_type: str, **kwargs):
    """
    Creates and returns a convolutional layer for ``2D`` or ``3D`` inputs.

    This utility provides a unified interface for constructing standard,
    transposed, separable, and depthwise convolution layers across both
    ``2D`` and ``3D`` data. The appropriate Keras layer implementation is selected
    automatically based on ``spatial_dims`` and ``layer_type``.

    Args:
        spatial_dims (int):
            Number of spatial dimensions. Supported values are:

            - ``2`` for images with shape
              ``(batch, height, width, channels)``
            - ``3`` for volumetric data with shape
              ``(batch, depth, height, width, channels)``

        layer_type (str):
            Type of convolution layer to create. Supported values are:

            - ``"conv"``
            - ``"conv_transpose"``
            - ``"separable_conv"``
            - ``"depthwise_conv"``

        **kwargs:
            Additional keyword arguments passed directly to the underlying
            Keras layer constructor.

            Common arguments include:

            - ``filters``
            - ``kernel_size``
            - ``strides``
            - ``padding``
            - ``activation``
            - ``use_bias``

            For ``depthwise_conv``, ``filters`` must **not** be provided.

    Returns:
        ``keras.layers.Layer``:
            Instantiated convolution layer corresponding to the requested
            configuration.

    Raises:
        ValueError:
            If ``spatial_dims`` is not ``2`` or ``3``.

        ValueError:
            If ``layer_type`` is not one of the supported layer types.

        ValueError:
            If ``filters`` is missing for layer types that require it
            (``conv``, ``conv_transpose``, ``separable_conv``).

        ValueError:
            If ``filters`` is provided for ``depthwise_conv``.

    Example:
        .. code-block:: python

            import keras
            from medicai.utils import get_conv_layer

            layer = get_conv_layer(
                spatial_dims=2,
                layer_type="conv",
                filters=64,
                kernel_size=3,
                padding="same"
            )
            isinstance(layer, keras.layers.Conv2D) # True

        .. code-block:: python

            import keras
            from medicai.utils import get_conv_layer

            layer = get_conv_layer(
                spatial_dims=3,
                layer_type="conv",
                filters=64,
                kernel_size=3,
                padding="same"
            )
            isinstance(layer, keras.layers.Conv3D) # True

        .. code-block:: python

            import keras
            from medicai.utils import get_conv_layer

            layer = get_conv_layer(
                spatial_dims=2,
                layer_type="conv_transpose",
                filters=64,
                kernel_size=3,
                padding="same"
            )
            isinstance(
                layer, keras.layers.Conv2DTranspose
            ) # True

        .. code-block:: python

            import keras
            from medicai.utils import get_conv_layer

            layer = get_conv_layer(
                spatial_dims=3,
                layer_type="conv_transpose",
                filters=64,
                kernel_size=3,
                padding="same"
            )
            isinstance(
                layer, keras.layers.Conv3DTranspose
            ) # True

        .. code-block:: python

            import keras
            from medicai.utils import get_conv_layer

            layer = get_conv_layer(
                spatial_dims=2,
                layer_type="separable_conv",
                filters=64,
                kernel_size=3,
                padding="same"
            )
            isinstance(
                layer, keras.layers.SeparableConv2D
            ) # True

        .. code-block:: python

            import medicai
            from medicai.utils import get_conv_layer

            layer = get_conv_layer(
                spatial_dims=3,
                layer_type="separable_conv",
                filters=64,
                kernel_size=3,
                padding="same"
            )
            isinstance(
                layer, medicai.layers.SeparableConv3D
            ) # True
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
            from medicai.layers import SeparableConv3D

            ConvClass = SeparableConv3D
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
    """
    Creates and returns a reshaping layer for ``2D`` or ``3D`` inputs.

    This utility provides a unified interface for constructing common
    spatial reshaping layers across both ``2D`` and ``3D`` data. Depending on
    the requested ``layer_type``, it returns an upsampling, padding,
    or cropping layer.

    Args:
        spatial_dims (int):
            Number of spatial dimensions. Supported values are:

            - ``2`` for images with shape
              ``(batch, height, width, channels)``
            - ``3`` for volumetric data with shape
              ``(batch, depth, height, width, channels)``

        layer_type (str):
            Type of reshaping layer to create. Supported values are:

            - ``"upsampling"``
            - ``"padding"``
            - ``"cropping"``

        **kwargs:
            Additional keyword arguments passed directly to the
            underlying Keras layer constructor.

            Common arguments include:

            - ``size`` for upsampling layers
            - ``padding`` for padding layers
            - ``cropping`` for cropping layers

    Returns:
        ``keras.layers.Layer``:
            Instantiated reshaping layer corresponding to the requested
            configuration.

    Raises:
        ValueError:
            If ``spatial_dims`` is not ``2`` or ``3``.

        ValueError:
            If ``layer_type`` is not one of:

            - ``"upsampling"``
            - ``"padding"``
            - ``"cropping"``

    Example:
        .. code-block:: python

            import keras
            from medicai.utils import get_reshaping_layer

            layer = get_reshaping_layer(
                spatial_dims=2,
                layer_type="upsampling",
                size=(2, 2)
            )
            isinstance(
                layer, keras.layers.UpSampling2D
            ) # True


        .. code-block:: python

            import keras
            from medicai.utils import get_reshaping_layer

            layer = get_reshaping_layer(
                spatial_dims=3,
                layer_type="padding",
                padding=((1, 1), (2, 2), (2, 2))
            )
            isinstance(
                layer, keras.layers.ZeroPadding3D
            ) # True


        .. code-block:: python

            import keras
            from medicai.utils import get_reshaping_layer

            layer = get_reshaping_layer(
                spatial_dims=3,
                layer_type="cropping",
                cropping=((2, 2), (10, 10), (10, 10))
            )
            isinstance(
                layer, keras.layers.Cropping3D
            ) # True
    """
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

    if layer_type not in (
        "upsampling",
        "padding",
        "cropping",
    ):
        raise ValueError(
            f"layer_type must be 'upsampling' or 'padding' or 'cropping'. Got: {layer_type}"
        )

    layers_map = {
        "upsampling": {2: layers.UpSampling2D, 3: layers.UpSampling3D},
        "padding": {2: layers.ZeroPadding2D, 3: layers.ZeroPadding3D},
        "cropping": {2: layers.Cropping2D, 3: layers.Cropping3D},
    }

    LayerClass = layers_map[layer_type][spatial_dims]
    return LayerClass(**kwargs)


def get_dropout_layer(spatial_dims, layer_type, **kwargs):
    """
    Creates and returns a dropout layer for ``2D`` or ``3D`` feature maps.

    This utility provides a unified interface for constructing spatial
    dropout layers across both ``2D`` and ``3D`` inputs. Unlike standard dropout,
    spatial dropout randomly drops entire feature maps (channels) rather
    than individual elements, which is often more effective for convolutional
    neural networks.

    
    Args:
        spatial_dims (int):
            Number of spatial dimensions. Supported values are:

            - ``2`` for images with shape
              ``(batch, height, width, channels)``
            - ``3`` for volumetric data with shape
              ``(batch, depth, height, width, channels)``

        layer_type (str):
            Type of dropout layer to create.

            Supported values:

            - ``"spatial_dropout"``

        **kwargs:
            Additional keyword arguments passed directly to the
            underlying Keras layer constructor.

            Common arguments include:

            - ``rate``: Fraction of channels to drop.
            - ``name``: Layer name.

    Returns:
        ``keras.layers.Layer``:
            Instantiated dropout layer corresponding to the requested
            configuration.

    Raises:
        ValueError:
            If ``spatial_dims`` is not ``2`` or ``3``.

        ValueError:
            If ``layer_type`` is not ``"spatial_dropout"``.

    Example:
        .. code-block:: python

            import keras
            from medicai.utils import get_dropout_layer

            layer = get_dropout_layer(
                spatial_dims=2,
                layer_type="spatial_dropout",
                rate=0.2
            )
            isinstance(
                layer, keras.layers.SpatialDropout2D
            ) # True

        .. code-block:: python

            import keras
            from medicai.utils import get_dropout_layer

            layer = get_dropout_layer(
                spatial_dims=3,
                layer_type="spatial_dropout",
                rate=0.2
            )
            isinstance(
                layer, keras.layers.SpatialDropout3D
            ) # True
    """
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

    if layer_type not in ("spatial_dropout",):
        raise ValueError(f"layer_type must be 'spatial_dropout'. Got: {layer_type}")

    layers_map = {
        "spatial_dropout": {2: layers.SpatialDropout2D, 3: layers.SpatialDropout3D},
    }
    LayerClass = layers_map[layer_type][spatial_dims]
    return LayerClass(**kwargs)


def get_pooling_layer(spatial_dims, layer_type, global_pool=False, **kwargs):
    """
    Creates and returns a pooling layer for ``2D`` or ``3D`` inputs.

    This utility provides a unified interface for constructing standard,
    global, and adaptive pooling layers across ``2D`` and ``3D`` feature maps.

    Args:
        spatial_dims (int):
            Number of spatial dimensions. Supported values are:

            - ``2`` for images: ``(batch, height, width, channels)``
            - ``3`` for volumes: ``(batch, depth, height, width, channels)``

        layer_type (str):
            Type of pooling operation. Supported values:

            - ``"max"``
            - ``"avg"``
            - ``"adaptive_max"``
            - ``"adaptive_avg"``

        global_pool (bool, optional):
            If ``True``, returns global pooling layers that reduce spatial
            dimensions entirely. Default is ``False``.

            Only applies to ``"max"`` and ``"avg"``.

        **kwargs:
            Additional keyword arguments passed to the underlying Keras
            pooling layer.

            Common arguments include:

            - ``pool_size``
            - ``strides``
            - ``padding``

            Adaptive pooling layers ignore most of these arguments.

    Returns:
        ``keras.layers.Layer``:
            Instantiated pooling layer corresponding to the requested configuration.

    Raises:
        ValueError:
            If ``spatial_dims`` is not ``2`` or ``3``.

        ValueError:
            If ``layer_type`` is not one of the supported pooling types.

    Example:
        .. code-block:: python

            import keras
            from medicai.utils import get_pooling_layer

            layer = get_pooling_layer(
                spatial_dims=2,
                layer_type="max",
                pool_size=(2, 2),
                strides=(2, 2)
            )
            isinstance(
                layer, keras.layers.MaxPooling2D
            ) # True

        .. code-block:: python

            import keras
            from medicai.utils import get_pooling_layer

            layer = get_pooling_layer(
                spatial_dims=3,
                layer_type="avg",
                pool_size=(2, 2, 2),
                strides=(2, 2, 2)
            )
            isinstance(
                layer, keras.layers.AveragePooling3D
            ) # True

        .. code-block:: python

            import keras
            from medicai.utils import get_pooling_layer

            layer = get_pooling_layer(
                spatial_dims=2,
                layer_type="avg",
                global_pool=True
            )
            isinstance(
                layer, keras.layers.GlobalAveragePooling2D
            ) # True

        .. code-block:: python

            import medicai
            from medicai.utils import get_pooling_layer

            layer = get_pooling_layer(
                spatial_dims=2,
                layer_type="adaptive_avg",
                output_size=(1, 1)
            )
            isinstance(
                layer, medicai.layers.AdaptiveAveragePooling2D
            ) # True
    """
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

    if layer_type not in (
        "max",
        "avg",
        "adaptive_max",
        "adaptive_avg",
    ):
        raise ValueError(
            "layer_type must be one of 'max', 'avg', 'adaptive_max', or 'adaptive_avg'. "
            f"Got: {layer_type}"
        )

    if layer_type.startswith("adaptive"):
        from medicai.layers import (
            AdaptiveAveragePooling2D,
            AdaptiveAveragePooling3D,
            AdaptiveMaxPooling2D,
            AdaptiveMaxPooling3D,
        )

    if layer_type == "adaptive_max":
        layers_map = {2: AdaptiveMaxPooling2D, 3: AdaptiveMaxPooling3D}
    elif layer_type == "adaptive_avg":
        layers_map = {2: AdaptiveAveragePooling2D, 3: AdaptiveAveragePooling3D}
    elif global_pool:
        layers_map = {
            "max": {2: layers.GlobalMaxPooling2D, 3: layers.GlobalMaxPooling3D},
            "avg": {2: layers.GlobalAveragePooling2D, 3: layers.GlobalAveragePooling3D},
        }[layer_type]
    else:
        layers_map = {
            "max": {2: layers.MaxPooling2D, 3: layers.MaxPooling3D},
            "avg": {2: layers.AveragePooling2D, 3: layers.AveragePooling3D},
        }[layer_type]

    LayerClass = layers_map[spatial_dims]
    return LayerClass(**kwargs)


def get_act_layer(layer_type, **kwargs):
    """
    Creates and returns a Keras activation layer.

    This utility provides a unified interface for constructing both
    standard activation functions (via ``keras.activations.get``) and
    parameterized activation layers (e.g., LeakyReLU, PReLU, ELU).

    Args:
        layer_type (str or callable):
            Name of the activation function or callable activation.

            - Standard activations (e.g., ``relu``, ``sigmoid``) are
              resolved using ``keras.activations.get``.
            - Special activations (``leaky_relu``, ``prelu``, ``elu``,
              ``relu``) use dedicated Keras layer implementations.

        **kwargs:
            Additional keyword arguments passed to the activation layer
            constructor when applicable.

            Examples:
            - ``negative_slope`` for ``leaky_relu``
            - ``alpha`` for ``elu``
            - ``alpha_initializer`` for ``prelu``

    Returns:
        ``keras.layers.Layer``:
            A Keras activation layer instance.

    Raises:
        ValueError:
            If the provided activation type is not supported by
            ``keras.activations.get`` and not in the special layer map.

    Example:
        .. code-block:: python

            import keras
            from medicai.utils import get_act_layer

            layer = get_act_layer("relu")
            isinstance(
                layer, keras.layers.ReLU
            ) # True
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
    Creates and returns a normalization layer for neural networks.

    This utility provides a unified interface for constructing different
    normalization strategies commonly used in deep learning architectures,
    including batch, layer, group, and instance normalization variants.

    Args:
        layer_type (str):
            Type of normalization layer to create. Supported values are:

            - ``"batch"``
            - ``"layer"``
            - ``"unit"``
            - ``"group"``
            - ``"instance"``
            - ``"sync_batch"``

        **kwargs:
            Additional keyword arguments passed directly to the
            normalization layer constructor.

            Common arguments include:

            - ``epsilon``: Small constant for numerical stability
            - ``momentum``: Momentum for batch statistics (BatchNorm)
            - ``axis``: Axis for normalization (LayerNorm, BatchNorm)
            - ``groups``: Number of groups (GroupNorm)

            Note:
                The ``instance`` normalization mode internally uses
                ``GroupNormalization`` with ``groups=-1`` and disables
                affine parameters (``scale=False, center=False``).

    Returns:
        ``keras.layers.Layer``:
            Instantiated normalization layer.

    Raises:
        ValueError:
            If ``layer_type`` is not one of the supported normalization types.

    Examples:
        .. code-block:: python

            import keras
            from medicai.utils import get_norm_layer

            layer = get_norm_layer("batch", momentum=0.9)
            isinstance(
                layer, keras.layers.BatchNormalization
            ) # True

        .. code-block:: python

            import keras
            from medicai.utils import get_norm_layer

            layer = get_norm_layer("layer", epsilon=1e-5)
            isinstance(
                layer, keras.layers.LayerNormalization
            ) # True

        .. code-block:: python

            norm = get_norm_layer("instance")

        .. code-block:: python

            norm = get_norm_layer("group", groups=8)

        .. code-block:: python

            norm = get_norm_layer("sync_batch")
    """
    layer_type = layer_type.lower()

    # Lookup table for other normalizations
    norm_layers = {
        "batch": layers.BatchNormalization,
        "layer": layers.LayerNormalization,
        "unit": layers.UnitNormalization,
        "group": layers.GroupNormalization,
        "instance": partial(
            layers.GroupNormalization, groups=-1, epsilon=1e-05, scale=False, center=False
        ),
        "sync_batch": partial(layers.BatchNormalization, synchronized=True),
    }

    if layer_type not in norm_layers:
        supported = ", ".join(norm_layers.keys())
        raise ValueError(
            f"Unsupported normalization type '{layer_type}'. " f"Supported types are: {supported}"
        )

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


def validate_activation(activation):
    """
    Validates an activation function string against the
    list of valid Keras activations.
    """
    if activation is None:
        return "linear"

    if not isinstance(activation, str):
        raise TypeError(
            f"Activation must be a string or None, but received type {type(activation).__name__!r} "
            f"with value {activation!r}."
        )

    normalized_activation = activation.lower()
    VALID_ACTIVATION_LIST = keras_constants.get_valid_activations()

    if normalized_activation not in VALID_ACTIVATION_LIST:
        raise ValueError(
            f"Invalid activation name: {activation!r}. "
            f"Supported string values are: {VALID_ACTIVATION_LIST}"
        )

    return normalized_activation


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
        if not isinstance(encoder, keras.Model):
            raise ValueError(
                "Argument `encoder` must be a `keras.Model` instance. "
                "Received instead "
                f"{encoder} (of type {type(encoder)})."
            )
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

        if encoder_name.lower() not in registration._backbone_registry:
            # TODO: Return supported family specific encoder name, not all!
            raise ValueError(
                f"Encoder '{encoder_name}' not found in the registry. "
                f"Available: {list(registration._backbone_registry.keys())}"
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

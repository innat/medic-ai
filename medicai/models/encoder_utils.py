from medicai.utils import registration


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

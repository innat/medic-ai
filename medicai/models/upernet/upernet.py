import keras
from keras import layers

from medicai.layers import ResizingND
from medicai.utils import (
    DescribeMixin,
    get_conv_layer,
    keras_constants,
    registration,
    resolve_encoder,
    validate_activation,
)

from .decoder import UPerNetDecoder


@keras.saving.register_keras_serializable(package="upernet")
@registration.register(name="upernet", type="segmentation")
class UPerNet(keras.Model, DescribeMixin):
    """
    UPerNet can be constructed either from a registered encoder name or from a
    pre-built encoder instance. The encoder must expose a
    ``pyramid_outputs`` dictionary with four or five feature levels so the
    decoder can build the PPM and FPN pathways.

    The model combines three components:

    1. An encoder that extracts a multi-scale feature pyramid.
    2. A Pyramid Pooling Module (PPM) that gathers global context from the
       deepest encoder feature.
    3. A Feature Pyramid Network (FPN) that progressively fuses deeper
       semantic features with higher-resolution shallower features.

    Args:
        encoder: Optional pre-built Keras model to use as the encoder. It must
            expose ``pyramid_outputs`` with four or five feature levels.
        encoder_name: Optional name of a registered backbone model to build and
            use as the encoder.
        input_shape: Optional input shape excluding the batch dimension.
            Required when ``encoder_name`` is used. This can describe either
            2D or 3D inputs.
        num_classes: Number of segmentation classes. Must be greater than
            zero.
        decoder_filters: Channel width used throughout the PPM and FPN decoder
            stages.
        decoder_normalization: Normalization behavior used in decoder blocks.
        decoder_activation: Activation function used in decoder blocks.
        head_upsample: Final upsampling factor applied before prediction. Can
            be an integer, float, tuple, or list.
        classifier_activation: Activation function used by the final
            segmentation head.
        name: Optional model name.
        **kwargs: Additional keyword arguments passed to ``keras.Model``.

    Examples:
        .. code-block:: python

            import tensorflow as tf
            from medicai.models import UPerNet

            model = UPerNet(
                encoder_name="efficientnet_v2_m",
                input_shape=(96, 96, 96, 1),
                num_classes=2,
            )

            x = tf.random.uniform(shape=[1, 96, 96, 96, 1])
            y = model(x)
            print(y.shape) # (1, 96, 96, 96, 2)

    .. rubric:: Encoder feature hierarchy
       :class: api-subheading

    The decoder uses the feature hierarchy as follows:

    - For five-stage encoders, the ``PPM`` operates on ``P5`` and the FPN fuses
      ``P4``, ``P3``, and ``P2``.
    - For four-stage encoders, the ``PPM`` operates on ``P4`` and the FPN fuses
      ``P3``, ``P2``, and ``P1``.

    Unlike other segmentation models in this codebase, UPerNet does not
    expose an ``encoder_depth`` argument. The number of encoder stages is
    inferred directly from ``encoder.pyramid_outputs``.

    Example:
        Build `UPerNet` with a four-stage encoder::

            import tensorflow as tf
            from medicai.models import UPerNet

            model = UPerNet(
                encoder_name="convnext_tiny",
                input_shape=(224, 224, 3),
                num_classes=5,
            )

            x = tf.random.uniform(shape=[1, 224, 224, 3])
            y = model(x)
            print(y.shape) # (1, 224, 224, 5)

    .. rubric:: Output resolution
       :class: api-subheading

    Some encoders, such as Swin Transformer variants, may produce feature
    hierarchies whose final decoder output is lower than the input resolution.
    In these cases, ``head_upsample`` can be used to restore the final
    segmentation output to the desired resolution.

    Example:
        Adjust the final upsampling factor for a Swin encoder::

            import tensorflow as tf
            from medicai.models import UPerNet

            model = UPerNet(
                encoder_name="swin_tiny",
                input_shape=(96, 96, 96, 4),
                num_classes=3,
                head_upsample=8,
            )

            x = tf.random.uniform(shape=[1, 96, 96, 96, 4])
            y = model(x)
            print(y.shape) # (1, 96, 96, 96, 3)

    .. rubric:: Custom encoder
       :class: api-subheading

    When providing a custom encoder through ``encoder``, ensure that:

    1. It defines a ``pyramid_outputs`` dictionary with either four or five
       ordered feature levels.
    2. The feature hierarchy is compatible with the PPM and FPN decoder
       design.
    3. The final ``head_upsample`` factor is adjusted if the encoder uses a
       non-standard downsampling pattern.

    Example:
        Build the model from a custom encoder::

            import tensorflow as tf
            from medicai.models import UPerNet, DenseNetBackbone

            encoder = DenseNetBackbone(
                blocks=[6, 12, 64, 48],
                input_shape=(96, 96, 96, 4),
            )

            model = UPerNet(
                encoder=encoder,
                num_classes=3,
            )

            x = tf.random.uniform(shape=[1, 96, 96, 96, 4])
            y = model(x)
            print(y.shape) # (1, 96, 96, 96, 3)

    Returns:
        A ``keras.Model`` whose forward pass returns a segmentation tensor of
        shape ``(batch_size, ..., num_classes)``.

    References:
        - Unified Perceptual Parsing for Scene Understanding.
          `arXiv:1807.10221 <https://arxiv.org/abs/1807.10221>`_
    """

    ALLOWED_BACKBONE_FAMILIES = [
        "resnet",
        "densenet",
        "efficientnet",
        "convnext",
        "senet",
        "xception",
        "swin",
        "mit",
    ]

    def __init__(
        self,
        *,
        input_shape=None,
        encoder_name=None,
        encoder=None,
        num_classes=1,
        decoder_filters=256,
        decoder_normalization="batch",
        decoder_activation="relu",
        head_upsample=4,
        classifier_activation="sigmoid",
        name=None,
        **kwargs,
    ):
        """
        Initializes the UPerNet model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size.
            encoder: (Optional) A Keras model to use as the encoder (backbone).
                This argument is intended for passing a custom or pre-trained
                model. If provided, the model must have a `pyramid_outputs` attribute,
                which should be a dictionary of intermediate feature vectors from shallow
                to deep layers (e.g., `'P1'`, `'P2'`, ...).
            encoder_name: (Optional) A string specifying the name of a
                pre-configured backbone from the `medicai.models.list_models()` to use as
                the encoder. This is a convenient option for using a backbone from
                the library without having to instantiate it manually.
            decoder_normalization (str | bool): Controls the use of normalization layers in
                UPerNet decoder blocks. It is a string specifying the normalization type.
                - If a string is provided, the specified normalization type will be used instead.
                    Supported arguments:
                        [False, "batch", "layer", "unit", "group", "instance", "sync_batch"]
                - If `False`, no normalization layer will be used.
                Supported options include:
                    - `"batch"`: `keras.layers.BatchNormalization`
                    - `"layer"`: `keras.layers.LayerNormalization`
                    - `"unit"`: `keras.layers.UnitNormalization`
                    - `"group"`: `keras.layers.GroupNormalization`
                    - `"instance"`: `partial(
                            keras.layers.GroupNormalization,
                            groups=-1, epsilon=1e-05, scale=False, center=False
                        )`
                    - `"sync_batch"`: `partial(keras.layers.BatchNormalization, synchronized=True)`
            decoder_activation (str): Controls the use of activation layers in UPerNet decoder blocks.
                It should the activation string identifier in available in keras.
                Default: 'relu'
            decoder_filters: An integer specifying the number of channels used for all feature
                maps in the PyramidPoolingModule (PPM) and FeaturePyramidNetwork (FPN) stages.
            num_classes: An integer specifying the number of classes for the
                final segmentation mask.
            head_upsample : int or tuple/list of ints, default=4
                Optional upsampling factor for the final segmentation head.
                - If an `int` > 1, all spatial dimensions are upsampled by this factor.
                - If a `tuple` or `list`, each element specifies the upsampling factor for the
                    corresponding spatial dimension (2D: (H, W), 3D: (D, H, W)).
                - If 1 or all elements are 1, no upsampling is applied.
                This is useful when the decoder output is smaller than the desired output resolution.
            classifier_activation: A string specifying the activation function
                for the final classification layer.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """

        encoder, input_shape = resolve_encoder(
            encoder=encoder,
            encoder_name=encoder_name,
            input_shape=input_shape,
            allowed_families=UPerNet.ALLOWED_BACKBONE_FAMILIES,
        )
        inputs = encoder.input
        spatial_dims = len(input_shape) - 1
        pyramid_outputs = encoder.pyramid_outputs

        # Determine required pyramid levels dynamically
        available_keys = set(pyramid_outputs.keys())

        # Find missing ones
        if len(available_keys) not in (4, 5):
            raise ValueError(
                f"UPerNet requires 4 or 5 pyramid levels, but got {len(available_keys)}. "
                f"Available keys: {available_keys}"
            )

        # number of classes must be positive.
        if num_classes <= 0:
            raise ValueError(
                f"Number of classes (`num_classes`) must be greater than 0, "
                f"but received {num_classes}."
            )

        if isinstance(decoder_normalization, str):
            decoder_normalization = decoder_normalization.lower()

        if decoder_normalization not in keras_constants.VALID_DECODER_NORMS:
            raise ValueError(
                f"Invalid value for `decoder_normalization`: {decoder_normalization!r}. "
                f"Supported values are: {keras_constants.VALID_DECODER_NORMS}"
            )

        # verify input activation.
        classifier_activation = validate_activation(classifier_activation)
        decoder_activation = validate_activation(decoder_activation)

        # Prepare head and skip layers
        # For UPerNet, we take deepeset feature (P5) for PPM block and others for FPN block.
        sorted_keys = sorted(available_keys, key=lambda x: int(x[1:]), reverse=True)
        bottleneck = pyramid_outputs[sorted_keys[0]]  # P5
        skip_layers = [pyramid_outputs[key] for key in sorted_keys[1:4]]  # [P4, P3, P2]

        # UPerNet Decoder
        decoder = UPerNetDecoder(
            spatial_dims=spatial_dims,
            skip_layers=skip_layers,
            decoder_filters=decoder_filters,
            decoder_normalization=decoder_normalization,
            decoder_activation=decoder_activation,
        )
        x = decoder(bottleneck)

        # Final segmentation head
        x = get_conv_layer(
            spatial_dims, layer_type="conv", filters=num_classes, kernel_size=1, padding="same"
        )(x)

        # Some encoder like, i.e. convnext need final upsampling.
        if isinstance(head_upsample, (int, float)):
            if head_upsample > 1:
                x = ResizingND(
                    scale_factor=head_upsample,
                    interpolation="bilinear" if spatial_dims == 2 else "trilinear",
                )(x)
        elif isinstance(head_upsample, (list, tuple)):
            if any(s > 1 for s in head_upsample):
                x = ResizingND(
                    scale_factor=head_upsample,
                    interpolation="bilinear" if spatial_dims == 2 else "trilinear",
                )(x)
        else:
            raise ValueError(
                f"`head_upsample` must be int, float, tuple, or list, got {type(head_upsample)}"
            )

        outputs = layers.Activation(classifier_activation, dtype="float32", name="predictions")(x)

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or f"UPerNet{spatial_dims}D", **kwargs
        )

        # Store config
        self._input_shape = input_shape
        self.encoder_name = encoder_name
        self.encoder = encoder
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
        self.decoder_filters = decoder_filters
        self.decoder_activation = decoder_activation
        self.decoder_normalization = decoder_normalization
        self.head_upsample = head_upsample

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self._input_shape,
                "encoder_name": self.encoder_name,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "decoder_filters": self.decoder_filters,
                "decoder_normalization": self.decoder_normalization,
                "decoder_activation": self.decoder_activation,
                "head_upsample": self.head_upsample,
            }
        )

        if self.encoder_name is None and self.encoder is not None:
            config.update({"encoder": keras.saving.serialize_keras_object(self.encoder)})
        return config

    @classmethod
    def from_config(cls, config):
        if "encoder" in config and isinstance(config["encoder"], dict):
            config["encoder"] = keras.layers.deserialize(config["encoder"])
        return super().from_config(config)

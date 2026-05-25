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

from .decoder import UNetDecoder


@keras.saving.register_keras_serializable(package="unet")
@registration.register(name="unet", type="segmentation")
class UNet(keras.Model, DescribeMixin):
    """
    UNet can be constructed either from a registered encoder name or from a
    pre-built encoder instance. The encoder must expose a
    ``pyramid_outputs`` dictionary so the decoder can retrieve the bottleneck
    feature and the skip features used during upsampling.

    The model combines three components:

    1. An encoder that extracts a multi-scale feature pyramid.
    2. A decoder that progressively upsamples the deepest selected feature.
    3. Skip connections that fuse shallower encoder features to recover
       spatial detail before the final segmentation head.

    Args:
        encoder: Optional pre-built Keras model to use as the encoder. It must
            expose ``pyramid_outputs`` with the required feature levels.
        encoder_name: Optional name of a registered backbone model to build and
            use as the encoder.
        input_shape: Optional input shape excluding the batch dimension.
            Required when ``encoder_name`` is used. This can describe either
            2D or 3D inputs.
        encoder_depth: Number of encoder pyramid levels to use. Valid values
            are ``3``, ``4``, and ``5``.
        num_classes: Number of segmentation classes. Must be greater than
            zero.
        decoder_block_type: Decoder upsampling strategy. Supported values are
            ``"upsampling"`` and ``"transpose"``.
        decoder_filters: Sequence of channel widths used by the decoder
            refinement path.
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
            from medicai.models import UNet

            model = UNet(
                encoder_name="efficientnet_b1",
                input_shape=(96, 96, 96, 1),
                num_classes=2,
            )

            x = tf.random.uniform(shape=[1, 96, 96, 96, 1])
            y = model(x)
            print(y.shape) # (1, 96, 96, 96, 2)


    .. rubric:: Encoder depth
       :class: api-subheading

    The ``encoder_depth`` argument controls which encoder feature is used as
    the bottleneck and which shallower features are used as skip connections:

    - ``encoder_depth=5`` uses ``P5`` as the bottleneck and ``P4`` through
      ``P1`` as skip connections.
    - ``encoder_depth=4`` uses ``P4`` as the bottleneck and ``P3`` through
      ``P1`` as skip connections.
    - ``encoder_depth=3`` uses ``P3`` as the bottleneck and ``P2`` through
      ``P1`` as skip connections.

    Example:
        Reduce encoder depth::

            import tensorflow as tf
            from medicai.models import UNet

            model = UNet(
                encoder_name="efficientnet_b1",
                encoder_depth=3,
                input_shape=(96, 96, 96, 1),
                num_classes=2,
            )

            x = tf.random.uniform(shape=[1, 96, 96, 96, 1])
            y = model(x)
            print(y.shape) # (1, 96, 96, 96, 2)

    .. rubric:: Decoder strategy
       :class: api-subheading

    The decoder can be built with one of two upsampling strategies:

    - ``decoder_block_type="upsampling"`` uses interpolation-based upsampling.
    - ``decoder_block_type="transpose"`` uses trainable transposed
      convolutions.

    Some encoders reduce the spatial resolution more aggressively than the
    default UNet decoder assumes. In these cases, ``head_upsample`` can be
    used to restore the final segmentation output to the desired resolution.

    Example:
        Use transposed convolution in the decoder::

            import tensorflow as tf
            from medicai.models import UNet

            model = UNet(
                encoder_name="efficientnet_b1",
                decoder_block_type="transpose",
                input_shape=(96, 96, 96, 1),
                num_classes=2,
            )

            x = tf.random.uniform(shape=[1, 96, 96, 96, 1])
            y = model(x)
            print(y.shape) # (1, 96, 96, 96, 2)

    .. rubric:: Custom encoder
       :class: api-subheading

    When providing a custom encoder through ``encoder``, ensure that:

    1. It defines a ``pyramid_outputs`` dictionary with the required feature
       levels.
    2. The selected ``encoder_depth`` matches the available pyramid levels.
    3. The feature hierarchy is compatible with the UNet decoder.
    4. The final ``head_upsample`` factor is adjusted if the encoder uses a
       non-standard downsampling pattern.

    Example:
        Build the model from a custom encoder::

            import tensorflow as tf
            from medicai.models import UNet, DenseNetBackbone

            encoder = DenseNetBackbone(
                blocks=[6, 12, 64, 48],
                input_shape=(96, 96, 96, 4),
            )
            model = UNet(
                encoder=encoder,
                encoder_depth=4,
                num_classes=3,
                classifier_activation="softmax",
                name='densenet_unet'
            )

            x = tf.random.uniform(shape=[1, 96, 96, 96, 4])
            y = model(x)
            print(y.shape) # (1, 96, 96, 96, 3)

    Returns:
        A ``keras.Model`` whose forward pass returns a segmentation tensor of
        shape ``(batch_size, ..., num_classes)``.
    """

    ALLOWED_BACKBONE_FAMILIES = [
        "resnet",
        "densenet",
        "efficientnet",
        "convnext",
        "senet",
        "xception",
    ]

    def __init__(
        self,
        *,
        input_shape=None,
        encoder_name=None,
        encoder=None,
        encoder_depth=5,
        num_classes=1,
        decoder_block_type="upsampling",
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_normalization="batch",
        decoder_activation="relu",
        head_upsample=1,
        classifier_activation="sigmoid",
        name=None,
        **kwargs,
    ):
        """
        Initializes the UNet model.

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
            encoder_depth: An integer specifying how many stages of the encoder
                backbone to use. A number of stages used in encoder in range [3, 5].
                Expected available intermediate or pyramid level, P1, P2, ... P5.
                If `encoder_depth=5`, bottleneck key would be P5, and P4...P1 will
                be used for skip connection. If `encoder_depth=4`, bottleneck key
                would be P4, and P3..P1 will be used for skip connection.
                The `encoder_depth` should be in [3, 4, 5]. This can be used to
                reduce the size of the model. Default: 5.
            decoder_block_type: A string specifying the type of decoder block
                to use. Can be "upsampling" or "transpose". "upsampling"
                uses a `UpSamplingND` layer followed by a convolution, while
                "transpose" uses a `ConvNDTranspose` layer.
                Note: When using 'transpose', only one Conv3x3BnReLU block is
                applied after upsampling to reduce trainable parameters, whereas
                'upsampling' uses two blocks.
            decoder_normalization (str | bool): Controls the use of normalization layers in UNet decoder blocks.
                It is a string specifying the normalization type.
                - If a string is provided, the specified normalization type will be used instead.
                    Supported arguments:
                        [False, "batch", "layer", "unit", "group", "instance", "sync_batch]
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
            decoder_activation (str): Controls the use of activation layers in UNet decoder blocks.
                It should the activation string identifier in available in keras.
                Default: 'relu'
            decoder_filters: A tuple of integers specifying the number of
                filters for each block in the decoder path. The number of
                filters should correspond to the `encoder_depth`.
            num_classes: An integer specifying the number of classes for the
                final segmentation mask.
            head_upsample : int or tuple/list of ints, default=1
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
            allowed_families=UNet.ALLOWED_BACKBONE_FAMILIES,
        )
        inputs = encoder.input
        spatial_dims = len(input_shape) - 1
        pyramid_outputs = encoder.pyramid_outputs

        # Determine required pyramid levels dynamically
        required_keys = {f"P{i}" for i in range(1, encoder_depth + 1)}
        available_keys = set(pyramid_outputs.keys())

        # Find missing ones
        missing_keys = required_keys - available_keys
        if missing_keys:
            raise ValueError(
                f"The encoder's `pyramid_outputs` is missing required pyramid levels. "
                f"Missing: {missing_keys}. "
                f"Expected keys (based on encoder_depth={encoder_depth}): {required_keys}, "
                f"but got: {available_keys}"
            )

        if not (3 <= encoder_depth <= 5):
            raise ValueError(f"encoder_depth must be in range [3, 5], but got {encoder_depth}")

        if len(decoder_filters) < encoder_depth:
            raise ValueError(
                f"Length of decoder_filters ({len(decoder_filters)}) must be >= encoder_depth ({encoder_depth})."
            )

        # number of classes must be positive.
        if num_classes <= 0:
            raise ValueError(
                f"Number of classes (`num_classes`) must be greater than 0, "
                f"but received {num_classes}."
            )

        if isinstance(decoder_block_type, str):
            decoder_block_type = decoder_block_type.lower()

        if decoder_block_type not in keras_constants.VALID_DECODER_BLOCK_TYPE:
            raise ValueError(
                f"Invalid decoder_block_type: '{decoder_block_type}'. "
                "Expected one of ('upsampling', 'transpose')."
            )

        if isinstance(decoder_normalization, str):
            decoder_normalization = decoder_normalization.lower()

        if decoder_normalization not in keras_constants.VALID_DECODER_NORMS:
            raise ValueError(
                f"Invalid value for `decoder_normalization`: {decoder_normalization!r}. "
                f"Supported values are: {keras_constants.VALID_DECODER_NORMS}"
            )

        # verify input activation.
        decoder_activation = validate_activation(decoder_activation)
        classifier_activation = validate_activation(classifier_activation)

        # prepare head and skip layers
        sorted_keys = sorted(required_keys, key=lambda x: int(x[1:]), reverse=True)
        bottleneck = pyramid_outputs[sorted_keys[0]]
        skip_layers = [pyramid_outputs[key] for key in sorted_keys[1:]]
        decoder_filters = decoder_filters[:encoder_depth]

        # unet decoder blocks
        decoder_attention = getattr(self, "decoder_attention_gate", False)
        decoder = UNetDecoder(
            spatial_dims,
            skip_layers,
            decoder_filters,
            decoder_block_type=decoder_block_type,
            decoder_attention=decoder_attention,
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
            inputs=inputs, outputs=outputs, name=name or f"UNet{spatial_dims}D", **kwargs
        )

        self._input_shape = input_shape
        self.encoder_name = encoder_name
        self.encoder = encoder
        self.encoder_depth = encoder_depth
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
        self.decoder_block_type = decoder_block_type
        self.decoder_filters = decoder_filters
        self.decoder_normalization = decoder_normalization
        self.decoder_activation = decoder_activation
        self.head_upsample = head_upsample

    def get_config(self):
        config = {
            "input_shape": self._input_shape,
            "encoder_name": self.encoder_name,
            "encoder_depth": self.encoder_depth,
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "decoder_block_type": self.decoder_block_type,
            "decoder_filters": self.decoder_filters,
            "decoder_normalization": self.decoder_normalization,
            "decoder_activation": self.decoder_activation,
            "head_upsample": self.head_upsample,
        }

        if self.encoder_name is None and self.encoder is not None:
            config.update({"encoder": keras.saving.serialize_keras_object(self.encoder)})
        return config

    @classmethod
    def from_config(cls, config):
        if "encoder" in config and isinstance(config["encoder"], dict):
            config["encoder"] = keras.layers.deserialize(config["encoder"])
        return super().from_config(config)

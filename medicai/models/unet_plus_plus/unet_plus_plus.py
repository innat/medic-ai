import keras
from keras import layers

from medicai.utils import (
    VALID_ACTIVATION_LIST,
    VALID_DECODER_BLOCK_TYPE,
    VALID_DECODER_NORMS,
    DescribeMixin,
    get_conv_layer,
    resolve_encoder,
)

from .decoder import UNetPlusPlusDecoder


@keras.saving.register_keras_serializable(package="unet")
class UNetPlusPlus(keras.Model, DescribeMixin):
    """
    UNet++ model with dense skip connections.

    Reference: https://arxiv.org/abs/1807.10165

    Example:
    >>> from medicai.models import UNetPlusPlus
    >>> model = UNetPlusPlus(input_shape=(96, 96, 1), encoder_name="densenet121")
    >>> model = UNetPlusPlus(input_shape=(96, 96, 96, 1), encoder_name="efficientnet_b0")
    """

    ALLOWED_BACKBONE_FAMILIES = ["resnet", "densenet", "efficientnet"]

    def __init__(
        self,
        *,
        input_shape=None,
        encoder_name=None,
        encoder=None,
        encoder_depth=5,
        num_classes=1,
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_normalization="batch",
        decoder_activation="relu",
        decoder_block_type="upsampling",
        classifier_activation="sigmoid",
        name=None,
        **kwargs,
    ):
        """
        Initializes the UNet++ model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size.
            encoder: (Optional) A Keras model to use as the encoder (backbone).
                This argument is intended for passing a custom or pre-trained
                model not available in the `BACKBONE_ZOO`. If provided, the
                model must have a `pyramid_outputs` attribute, which should be
                a dictionary of intermediate feature vectors from shallow to
                deep layers (e.g., `'P1'`, `'P2'`, ...).
            encoder_name: (Optional) A string specifying the name of a
                pre-configured backbone from the `BACKBONE_ZOO` to use as the
                encoder. This is a convenient option for using a backbone from
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
            classifier_activation: A string specifying the activation function
                for the final classification layer.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """
        encoder, input_shape = resolve_encoder(
            encoder=encoder,
            encoder_name=encoder_name,
            input_shape=input_shape,
            allowed_families=UNetPlusPlus.ALLOWED_BACKBONE_FAMILIES,
        )
        inputs = encoder.input
        spatial_dims = len(input_shape) - 1
        pyramid_outputs = encoder.pyramid_outputs

        required_keys = {"P1", "P2", "P3", "P4", "P5"}
        missing_keys = set(required_keys) - set(pyramid_outputs.keys())
        if missing_keys:
            raise ValueError(
                f"The encoder's `pyramid_outputs` is missing one or more required keys. "
                f"Missing keys: {missing_keys}. "
                f"Required: {set(required_keys)}, Available: {set(pyramid_outputs.keys())}"
            )

        if not (3 <= encoder_depth <= 5):
            raise ValueError(f"encoder_depth must be in range [3, 5], but got {encoder_depth}")

        if len(decoder_filters) < encoder_depth:
            raise ValueError(
                f"Length of decoder_filters ({len(decoder_filters)}) must be >= encoder_depth ({encoder_depth})."
            )

        if isinstance(decoder_normalization, str):
            decoder_normalization = decoder_normalization.lower()

        if decoder_normalization not in VALID_DECODER_NORMS:
            supported = ", ".join([str(v) for v in VALID_DECODER_NORMS])
            raise ValueError(
                f"Invalid value for `decoder_normalization`: {decoder_normalization!r}. "
                f"Supported values are: {supported}"
            )

        if isinstance(decoder_activation, str):
            decoder_activation = decoder_activation.lower()

        if decoder_activation not in VALID_ACTIVATION_LIST:
            raise ValueError(
                f"Invalid value for `decoder_activation`: {decoder_activation!r}. "
                f"Supported values are: {VALID_ACTIVATION_LIST}"
            )

        # Prepare head and skip layers (same as UNet)
        bottleneck_keys = sorted(required_keys, key=lambda x: int(x[1:]), reverse=True)
        bottleneck_index = 5 - encoder_depth
        bottleneck = pyramid_outputs[bottleneck_keys[bottleneck_index]]
        skip_layers = [pyramid_outputs[key] for key in bottleneck_keys[bottleneck_index + 1 :]]
        decoder_filters = decoder_filters[:encoder_depth]

        # UNet++ Decoder
        decoder = UNetPlusPlusDecoder(
            spatial_dims=spatial_dims,
            skip_layers=skip_layers,
            decoder_filters=decoder_filters,
            decoder_block_type=decoder_block_type,
            decoder_normalization=decoder_normalization,
            decoder_activation=decoder_activation,
        )
        x = decoder(bottleneck)

        # Final segmentation head
        x = get_conv_layer(
            spatial_dims, layer_type="conv", filters=num_classes, kernel_size=1, padding="same"
        )(x)
        outputs = layers.Activation(classifier_activation, dtype="float32")(x)

        super().__init__(
            inputs=inputs, outputs=outputs, name=name or f"UNetPlusPlus{spatial_dims}D", **kwargs
        )

        # Store config
        self._input_shape = input_shape
        self.encoder_name = encoder_name
        self.encoder = encoder
        self.encoder_depth = encoder_depth
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
        self.decoder_block_type = decoder_block_type
        self.decoder_filters = decoder_filters
        self.decoder_activation = decoder_activation
        self.decoder_normalization = decoder_normalization

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self._input_shape,
                "encoder_name": self.encoder_name,
                "encoder_depth": self.encoder_depth,
                "num_classes": self.num_classes,
                "classifier_activation": self.classifier_activation,
                "decoder_filters": self.decoder_filters,
                "decoder_block_type": self.decoder_block_type,
                "decoder_normalization": self.decoder_normalization,
                "decoder_activation": self.decoder_activation,
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

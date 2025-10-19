import keras
from keras import layers

from medicai.layers import ResizingND
from medicai.utils import (
    VALID_ACTIVATION_LIST,
    VALID_DECODER_BLOCK_TYPE,
    VALID_DECODER_NORMS,
    DescribeMixin,
    get_conv_layer,
    registration,
    resolve_encoder,
)

from .decoder import UNetDecoder


@keras.saving.register_keras_serializable(package="unet")
@registration.register(name="unet", type="segmentation")
class UNet(keras.Model, DescribeMixin):
    """
    The UNet model for semantic segmentation.

    UNet is a convolutional neural network architecture developed for biomedical
    image segmentation. It consists of a symmetric encoder-decoder structure
    where the encoder (downsampling path) captures context and the decoder
    (upsampling path) enables precise localization. The key feature of UNet is
    the use of "skip connections," which concatenate feature maps from the
    encoder directly to the corresponding layers in the decoder. This allows
    the decoder to leverage high-resolution features lost during downsampling,
    leading to more accurate segmentations.

    Example:
    >>> from medicai.models import UNet
    >>> model = UNet(input_shape=(96, 96, 1), encoder_name="densenet121")
    >>> model = UNet(input_shape=(96, 96, 96, 1), encoder_name="densenet121")
    >>> model = UNet(input_shape=(96, 96, 1), encoder_name="densenet121", encoder_depth=3)

    """

    ALLOWED_BACKBONE_FAMILIES = ["resnet", "densenet", "efficientnet", "convnext", "resnext"]

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

        if decoder_block_type not in VALID_DECODER_BLOCK_TYPE:
            raise ValueError(
                f"Invalid decoder_block_type: '{decoder_block_type}'. "
                "Expected one of ('upsampling', 'transpose')."
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

        outputs = layers.Activation(classifier_activation, dtype="float32")(x)

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

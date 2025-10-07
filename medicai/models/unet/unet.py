import keras
from keras import layers

from medicai.layers import SpatialResize
from medicai.utils import DescribeMixin, get_conv_layer, resolve_encoder

from .unet_decoder import UNetDecoder


@keras.saving.register_keras_serializable(package="unet")
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

    ALLOWED_BACKBONE_FAMILIES = ["resnet", "densenet", "efficientnet"]

    def __init__(
        self,
        *,
        input_shape=None,
        encoder_name=None,
        encoder=None,
        encoder_depth=5,
        decoder_block_type="upsampling",
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_interpolation="nearest",
        decoder_use_batchnorm=True,
        decoder_use_attention=False,
        num_classes=1,
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
                This can be used to reduce the size of the model. Default: 5.
            decoder_block_type: A string specifying the type of decoder block
                to use. Can be "upsampling" or "transpose". "upsampling"
                uses a `UpSamplingND` layer followed by a convolution, while
                "transpose" uses a `ConvNDTranspose` layer.
            decoder_filters: A tuple of integers specifying the number of
                filters for each block in the decoder path. The number of
                filters should correspond to the `encoder_depth`.
            num_classes: An integer specifying the number of classes for the
                final segmentation mask.
            classifier_activation: A string specifying the activation function
                for the final classification layer.
            use_attention: A boolean indicating whether to use attention blocks
                in the decoder. If it is enabled, the UNet will be built as
                Attention-UNet. Default: False.
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

        required_keys = {"P1", "P2", "P3", "P4", "P5"}
        missing_keys = set(required_keys) - set(pyramid_outputs.keys())
        if missing_keys:
            raise ValueError(
                f"The encoder's `pyramid_outputs` is missing one or more required keys. "
                f"Missing keys: {missing_keys}. "
                f"Required: {set(required_keys)}, Available: {set(pyramid_outputs.keys())}"
            )

        pyramid_outputs = list(pyramid_outputs.values())

        if not (3 <= encoder_depth <= 5):
            raise ValueError(f"encoder_depth must be in range [3, 5], but got {encoder_depth}")

        if len(decoder_filters) < encoder_depth:
            raise ValueError(
                f"Length of decoder_filters ({len(decoder_filters)}) must be >= encoder_depth ({encoder_depth})."
            )

        # Ensure we only use up to `encoder_depth` stages
        pyramid_outputs = pyramid_outputs[::-1]  # reverse for decoder (deep (P5) → shallow (P1))
        pyramid_outputs = pyramid_outputs[:encoder_depth]

        # decoder filters
        decoder_filters = decoder_filters[:encoder_depth]

        # decoder block
        bottleneck = encoder.output
        decoder = UNetDecoder(
            pyramid_outputs,
            decoder_filters,
            spatial_dims,
            block_type=decoder_block_type,
            use_attention=decoder_use_attention,
            use_batchnorm=decoder_use_batchnorm,
            interpolation=decoder_interpolation,
        )
        x = decoder(bottleneck)

        # Final segmentation head
        x = get_conv_layer(
            spatial_dims, layer_type="conv", filters=num_classes, kernel_size=1, padding="same"
        )(x)
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
        self.decoder_use_attention = decoder_use_attention
        self.decoder_interpolation = decoder_interpolation
        self.decoder_use_batchnorm = decoder_use_batchnorm

    def get_config(self):
        config = {
            "input_shape": self._input_shape,
            "encoder_name": self.encoder_name,
            "encoder_depth": self.encoder_depth,
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "decoder_block_type": self.decoder_block_type,
            "decoder_filters": self.decoder_filters,
            "decoder_interpolation": self.decoder_interpolation,
            "decoder_use_batchnorm": self.decoder_use_batchnorm,
            "decoder_use_attention": self.decoder_use_attention,
        }

        if self.encoder_name is None and self.encoder is not None:
            config.update({"encoder": keras.saving.serialize_keras_object(self.encoder)})
        return config

    @classmethod
    def from_config(cls, config):
        if "encoder" in config and isinstance(config["encoder"], dict):
            config["encoder"] = keras.layers.deserialize(config["encoder"])
        return super().from_config(config)

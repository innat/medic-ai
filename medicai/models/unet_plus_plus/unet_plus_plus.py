import keras
from keras import layers

from medicai.utils import get_conv_layer, resolve_encoder

from .decoder import UNetPlusPlusDecoder


@keras.saving.register_keras_serializable(package="unet")
class UNetPlusPlus(keras.Model):
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
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        num_classes=1,
        decoder_block_type="upsampling",
        classifier_activation="sigmoid",
        name=None,
        **kwargs,
    ):
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
            block_type=decoder_block_type,
            use_batchnorm=decoder_use_batchnorm,
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
        self.decoder_use_batchnorm = decoder_use_batchnorm

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
                "decoder_use_batchnorm": self.decoder_use_batchnorm,
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

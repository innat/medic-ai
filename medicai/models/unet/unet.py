import keras
from keras import layers

from medicai.utils.model_utils import (
    BACKBONE_ZOO,
    get_conv_layer,
)

from .unet_imagenet_decoder import UNetDecoder


class UNet(keras.Model):
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

    """

    def __init__(
        self,
        *,
        input_shape,
        encoder=None,
        encoder_name=None,
        encoder_depth=4,
        decoder_block_type="upsampling",
        decoder_filters=(256, 128, 64, 32, 16),
        num_classes=1,
        classifier_activation="sigmoid",
        use_attention=False,
        name=None,
        **kwargs,
    ):
        """
        Initializes the UNet model.

        Args:
            input_shape: A tuple specifying the input shape of the model,
                not including the batch size.
            encoder: (Optional) A Keras model to use as the encoder (backbone).
                If `None`, an encoder will be built using `encoder_name`.
            encoder_name: (Optional) A string specifying the name of a
                pre-configured backbone from the `BACKBONE_ZOO` to use as the
                encoder.
            encoder_depth: An integer specifying how many stages of the encoder
                backbone to use.
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
                in the decoder.
            name: (Optional) The name of the model.
            **kwargs: Additional keyword arguments.
        """
        # If encoder provided, use it
        if encoder is not None:
            backbone = encoder
        elif encoder_name is not None:
            BackboneClass = BACKBONE_ZOO[encoder_name]
            backbone = BackboneClass(input_shape=input_shape)
        else:
            raise ValueError("Either `encoder` or `encoder_name` must be provided.")

        input = backbone.input
        spatial_dims = len(input[0].shape) - 2
        pyramid_outputs = list(backbone.pyramid_outputs.values())

        # Ensure we only use up to `encoder_depth` stages
        pyramid_outputs = pyramid_outputs[:encoder_depth][
            ::-1
        ]  # reverse for decoder (deep → shallow)
        bottleneck = pyramid_outputs[0]
        skip_layers = pyramid_outputs[1:]
        decoder_filters = decoder_filters[:encoder_depth]

        decoder = UNetDecoder(
            skip_layers,
            decoder_filters,
            spatial_dims,
            block_type=decoder_block_type,
            use_attention=use_attention,
        )
        x = decoder(bottleneck)

        # Final segmentation head
        x = get_conv_layer(
            spatial_dims, layer_type="conv", filters=num_classes, kernel_size=1, padding="same"
        )(x)
        outputs = layers.Activation(classifier_activation, dtype="float32")(x)

        super().__init__(
            inputs=input, outputs=outputs, name=name or f"UNet{spatial_dims}D", **kwargs
        )

        self.encoder_name = encoder_name
        self.encoder = encoder
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation
        self.decoder_block_type = decoder_block_type
        self.decoder_filters = decoder_filters
        self.use_attention = use_attention

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "encoder_name": self.encoder_name,
            "num_classes": self.num_classes,
            "encoder": self.encoder,
            "classifier_activation": self.classifier_activation,
            "decoder_block_type": self.decoder_block_type,
            "decoder_filters": self.decoder_filters,
            "use_attention": self.use_attention,
        }

        if self.encoder is not None:
            config.update({"encoder": keras.saving.serialize_keras_object(self.encoder)})
        return config

    @classmethod
    def from_config(cls, config):
        if "encoder" in config and isinstance(config["encoder"], dict):
            config["encoder"] = keras.layers.deserialize(config["encoder"])
        return super().from_config(config)

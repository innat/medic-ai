import keras

from medicai.blocks import UNetOutBlock, UNETRPlusPlusUpBlock
from medicai.utils import (
    DescribeMixin,
    keras_constants,
    registration,
    resolve_encoder,
    validate_activation,
)

from .encoder_layers import UNetResBlock


@keras.saving.register_keras_serializable(package="unetr_plusplus")
@registration.register(name="unetr_plusplus", type="segmentation")
class UNETRPlusPlus(keras.Model, DescribeMixin):
    ALLOWED_BACKBONE_FAMILIES = ["unetr_plusplus_encoder"]

    def __init__(
        self,
        input_shape=None,
        encoder_name=None,
        encoder=None,
        num_classes=1,
        feature_size=16,
        target_sequence_length=[8 * 8 * 8, 16 * 16 * 16, 32 * 32 * 32, 128 * 128 * 128],
        norm_name="instance",
        classifier_activation="sigmoid",
        **kwargs,
    ):
        encoder, input_shape = resolve_encoder(
            encoder=encoder,
            encoder_name=encoder_name,
            input_shape=input_shape,
            allowed_families=UNETRPlusPlus.ALLOWED_BACKBONE_FAMILIES,
        )
        inputs = encoder.input
        spatial_dims = len(input_shape) - 1
        pyramid_outputs = encoder.pyramid_outputs

        required_keys = ["P1", "P2", "P3", "P4"]
        missing_keys = set(required_keys) - set(pyramid_outputs.keys())
        if missing_keys:
            raise ValueError(
                f"The encoder's `pyramid_outputs` is missing one or more required keys. "
                f"Missing keys: {missing_keys}. "
                f"Required: {set(required_keys)}, Available: {set(pyramid_outputs.keys())}"
            )
        if num_classes <= 0:
            raise ValueError(
                f"Number of classes (`num_classes`) must be greater than 0, "
                f"but received {num_classes}."
            )

        if isinstance(norm_name, str):
            norm_name = norm_name.lower()
        if norm_name not in keras_constants.VALID_DECODER_NORMS:
            raise ValueError(
                f"Invalid value for `decoder_normalization`: {norm_name!r}. "
                f"Supported values are: {keras_constants.VALID_DECODER_NORMS}"
            )

        skips = [pyramid_outputs[key] for key in required_keys]
        classifier_activation = validate_activation(classifier_activation)

        # Build UNETR++ decoder.
        unetr_plusplus_head = self.build_decoder(
            spatial_dims=spatial_dims,
            num_classes=num_classes,
            feature_size=feature_size,
            norm_name=norm_name,
            classifier_activation=classifier_activation,
            target_sequence_length=target_sequence_length,
        )
        outputs = unetr_plusplus_head([inputs] + skips)

        super().__init__(inputs=inputs, outputs=outputs, name="model", **kwargs)

        self.num_classes = num_classes
        self.feature_size = feature_size
        self.norm_name = norm_name
        self.encoder_name = encoder_name
        self.encoder = encoder
        self.target_sequence_length = target_sequence_length
        self.classifier_activation = classifier_activation

    def build_decoder(
        self,
        spatial_dims,
        num_classes,
        feature_size,
        norm_name,
        classifier_activation,
        target_sequence_length,
    ):

        def apply(inputs):
            enc_input = inputs[0]
            enc1, enc2, enc3, enc4 = inputs[1:]
            dec4 = enc4

            convBlock = UNetResBlock(
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                name="stem_unet_residual_block",
            )(enc_input)

            dec3 = UNETRPlusPlusUpBlock(
                spatial_dims=spatial_dims,
                out_channels=feature_size * 8,
                kernel_size=3,
                upsample_kernel_size=2,
                sequence_length=target_sequence_length[0],
                norm_name=norm_name,
            )(dec4, enc3)

            dec2 = UNETRPlusPlusUpBlock(
                spatial_dims=spatial_dims,
                out_channels=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                sequence_length=target_sequence_length[1],
                norm_name=norm_name,
            )(dec3, enc2)

            dec1 = UNETRPlusPlusUpBlock(
                spatial_dims=spatial_dims,
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                sequence_length=target_sequence_length[2],
                norm_name=norm_name,
            )(dec2, enc1)

            out = UNETRPlusPlusUpBlock(
                spatial_dims=spatial_dims,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=4,
                sequence_length=target_sequence_length[3],
                norm_name=norm_name,
                conv_decoder=True,
            )(dec1, convBlock)

            output = UNetOutBlock(
                num_classes=num_classes,
                activation=classifier_activation,
            )(out)
            return output

        return apply

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "encoder_name": self.encoder_name,
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "feature_size": self.feature_size,
            "norm_name": self.norm_name,
            "target_sequence_length": self.target_sequence_length,
        }
        if self.encoder_name is None and self.encoder is not None:
            config.update({"encoder": keras.saving.serialize_keras_object(self.encoder)})
        return config

    @classmethod
    def from_config(cls, config):
        if "encoder" in config and isinstance(config["encoder"], dict):
            config["encoder"] = keras.layers.deserialize(config["encoder"])
        return super().from_config(config)

import keras
import numpy as np

from medicai.blocks import UNetOutBlock, UNetResBlock
from medicai.utils import (
    DescribeMixin,
    keras_constants,
    registration,
    resolve_encoder,
    validate_activation,
)

from .encoder_layers import UNETRPlusPlusUpsamplingBlock


@keras.saving.register_keras_serializable(package="unetr_plusplus")
@registration.register(name="unetr_plusplus", type="segmentation")
class UNETRPlusPlus(keras.Model, DescribeMixin):
    """
    Implements the UNETR++ model, a robust 3D segmentation architecture
    that enhances the original UNETR by integrating a hierarchical,
    convolutional-based decoder (similar to U-Net) to process the output
    from a Vision Transformer (ViT) or standard CNN encoder.

    The model expects an encoder that provides four pyramid output stages (P1, P2, P3, P4).
    The last feature map (P4/ViT output) is projected and then fed into the decoder.

    Args:
        input_shape (tuple, optional): The shape of the input tensor (e.g., (128, 128, 128, 1)).
            Must be specified if `encoder` or `encoder_name` is None.
        encoder_name (str, optional): Name of the backbone encoder model to use
            (e.g., 'unetr_plusplus_small').
        encoder (keras.Model, optional): A pre-instantiated encoder model.
            Must provide `pyramid_outputs` with keys P1, P2, P3, P4.
        num_classes (int): The number of output classes for segmentation. (default: 1).
        feature_size (int): Base number of filters in the decoder stages (e.g., 16, 32).
            Decoder channels are scaled by multiples of this size. (default: 16).
        norm_name (str): The normalization layer name used in the decoder blocks
            ('instance', 'batch'). (default: "instance").
        classifier_activation (str): The activation function for the final output layer
            ('sigmoid' for binary, 'softmax' for multi-class). (default: "sigmoid").
        name (str): The model name. (default: "UNETRPlusPlus").

    Example:
        ```python
        from medicai.models import UNETRPlusPlus, UNETRPlusPlusEncoder

        # Case 1
        input_shape = (128, 128, 128, 4)
        model = UNETRPlusPlus(
            input_shape=input_shape,
            encoder_name="unetr_plusplus_encoder",
            num_classes=3,
            classifier_activation=None
        )

        # Case 2
        input_shape = (16, 160, 160, 1)
        encoder = UNETRPlusPlusEncoder(
            input_shape=(16, 160, 160, 1),
            patch_size=(1, 4, 4)
        )
        model = UNETRPlusPlus(
            encoder=encoder,
            num_classes=3
        )
        ```
    """

    ALLOWED_BACKBONE_FAMILIES = ["unetr_plusplus"]

    def __init__(
        self,
        input_shape=None,
        encoder_name=None,
        encoder=None,
        num_classes=1,
        feature_size=16,
        norm_name="instance",
        classifier_activation="sigmoid",
        name="UNETRPlusPlus",
        **kwargs,
    ):
        encoder, input_shape = resolve_encoder(
            encoder=encoder,
            encoder_name=encoder_name,
            input_shape=input_shape,
            allowed_families=UNETRPlusPlus.ALLOWED_BACKBONE_FAMILIES,
        )
        inputs = encoder.input
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
        target_sequence_length = self.get_target_sequence_length(encoder)
        final_upsampling_kernel = self.get_final_upsample_kernel(encoder)

        # Build UNETR++ decoder.
        unetr_plusplus_head = self.build_decoder(
            num_classes=num_classes,
            feature_size=feature_size,
            norm_name=norm_name,
            classifier_activation=classifier_activation,
            target_sequence_length=target_sequence_length,
            final_upsampling_kernel=final_upsampling_kernel,
        )
        outputs = unetr_plusplus_head([inputs] + skips)

        super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)

        self.num_classes = num_classes
        self.feature_size = feature_size
        self.norm_name = norm_name
        self.encoder_name = encoder_name
        self.encoder = encoder
        self.classifier_activation = classifier_activation

    def build_decoder(
        self,
        num_classes,
        feature_size,
        norm_name,
        classifier_activation,
        target_sequence_length,
        final_upsampling_kernel,
    ):

        def apply(inputs):
            enc_input = inputs[0]
            enc1, enc2, enc3, enc4 = inputs[1:]
            dec4 = enc4

            convBlock = UNetResBlock(
                filters=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                name="decoder_input_projection",
            )(enc_input)

            dec3 = UNETRPlusPlusUpsamplingBlock(
                filters=feature_size * 8,
                kernel_size=3,
                upsample_kernel_size=2,
                sequence_length=target_sequence_length[0],
                norm_name=norm_name,
                name="unetrpp_decoder_up_stage3",
            )([dec4, enc3])

            dec2 = UNETRPlusPlusUpsamplingBlock(
                filters=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                sequence_length=target_sequence_length[1],
                norm_name=norm_name,
                name="unetrpp_decoder_up_stage2",
            )([dec3, enc2])

            dec1 = UNETRPlusPlusUpsamplingBlock(
                filters=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                sequence_length=target_sequence_length[2],
                norm_name=norm_name,
                name="unetrpp_decoder_up_stage1",
            )([dec2, enc1])

            out = UNETRPlusPlusUpsamplingBlock(
                filters=feature_size,
                kernel_size=3,
                upsample_kernel_size=final_upsampling_kernel,
                norm_name=norm_name,
                conv_decoder=True,
                name="unetrpp_decoder_output_stage",
            )([dec1, convBlock])

            output = UNetOutBlock(
                num_classes=num_classes,
                activation=classifier_activation,
                name="unetrpp_segmentation_head",
            )(out)
            return output

        return apply

    @staticmethod
    def get_target_sequence_length(encoder):
        """
        Determines token sequence lengths for each encoder stage.
        Works for:
        - UNETR++ encoder (has encoder.sequence_lengths)
        - Arbitrary CNN encoder
        - Arbitrary resolution inputs (2D or 3D)
        """
        # compute from pyramid_outputs spatial shapes
        if hasattr(encoder, "pyramid_outputs"):
            seq = []
            keys = ["P3", "P2", "P1"]
            for k in keys:
                feat = encoder.pyramid_outputs[k]
                spatial = feat.shape[1:-1]
                seq.append(int(np.prod(spatial)))
            return seq

        raise ValueError(
            "Cannot infer `sequence_length` for decoder. "
            "Encoder must provide pyramid_outputs` with spatial dimensions."
        )

    @staticmethod
    def get_final_upsample_kernel(encoder):
        """
        Computes the upsampling factor needed to recover input resolution
        from P1 resolution.
        """
        if hasattr(encoder, "pyramid_outputs"):
            input_spatial = encoder.input.shape[1:-1]
            p1_spatial = encoder.pyramid_outputs["P1"].shape[1:-1]

            if any(d is None for d in input_spatial) or any(d is None for d in p1_spatial):
                raise ValueError(
                    "Cannot infer final upsampling kernel with dynamic spatial dims. "
                    f"input={input_spatial}, P1={p1_spatial}. "
                    "Please pass a fully-specified input_shape."
                )

            upsample = tuple(
                int(in_dim // p1_dim)
                for in_dim, p1_dim in zip(input_spatial, p1_spatial, strict=True)
            )

            if any(
                in_dim != p1_dim * u
                for in_dim, p1_dim, u in zip(input_spatial, p1_spatial, upsample, strict=True)
            ):
                raise ValueError(
                    f"Input spatial dims must be divisible by P1 dims. input={input_spatial}, P1={p1_spatial}."
                )
            return upsample

        raise ValueError(
            "Cannot infer final upsampling kernel for decoder. "
            "Encoder must provide `pyramid_outputs` with spatial dimensions."
        )

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "encoder_name": self.encoder_name,
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "feature_size": self.feature_size,
            "norm_name": self.norm_name,
        }
        if self.encoder_name is None and self.encoder is not None:
            config.update({"encoder": keras.saving.serialize_keras_object(self.encoder)})
        return config

    @classmethod
    def from_config(cls, config):
        if "encoder" in config and isinstance(config["encoder"], dict):
            config["encoder"] = keras.layers.deserialize(config["encoder"])
        return super().from_config(config)

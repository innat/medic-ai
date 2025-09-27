from medicai.utils import hide_warnings

hide_warnings()


import keras

from medicai.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from medicai.utils import resolve_encoder


@keras.saving.register_keras_serializable(package="swin.unetr")
class SwinUNETR(keras.Model):
    """Swin-UNETR: A hybrid transformer-CNN for 3D or 2D medical image segmentation.

    This model combines the strengths of the Swin Transformer for feature extraction
    and a U-Net-like architecture for segmentation. It uses a Swin Transformer
    backbone to encode the input and a decoder with upsampling and skip connections
    to generate segmentation maps.
    """

    ALLOWED_BACKBONE_FAMILIES = ["swin"]

    def __init__(
        self,
        *,
        input_shape=None,
        encoder_name=None,
        encoder=None,
        num_classes=1,
        patch_size=2,
        window_size=7,
        classifier_activation=None,
        feature_size=48,
        res_block=True,
        norm_name="instance",
        **kwargs,
    ):
        """Initializes the SwinUNETR model.

        Args:
            input_shape (tuple): The shape of the input tensor (depth, height, width, channels).
            num_classes (int): The number of segmentation classes. Default is 4.
            classifier_activation (str, optional): The activation function for the final
                classification layer (e.g., 'softmax'). If None, no activation is applied.
                Default is None.
            patch_size (list): Size of the video patches (PD, PH, PW). Default is [2, 2, 2].
            window_size (list): Size of the attention windows (WD, WH, WW). Default is [7, 7, 7].
            feature_size (int): The base feature map size in the decoder. Default is 48.
            res_block (bool): Whether to use residual connections in the decoder blocks.
                Default is True.
            norm_name (str): The type of normalization to use in the decoder blocks
                (e.g., 'instance', 'batch'). Default is "instance".
            **kwargs: Additional keyword arguments passed to the base Model class.
        """

        # Compute spatial dimention and resolve encoder arguments
        encoder, input_shape = resolve_encoder(
            encoder=encoder,
            encoder_name=encoder_name,
            input_shape=input_shape,
            patch_size=patch_size,
            window_size=window_size,
            allowed_families=SwinUNETR.ALLOWED_BACKBONE_FAMILIES,
            pooling=None,
        )
        spatial_dims = len(input_shape) - 1

        # Get intermediate vectores
        pyramid_outputs = encoder.pyramid_outputs
        required_keys = ["P1", "P2", "P3", "P4", "P5"]
        missing_keys = set(required_keys) - set(pyramid_outputs.keys())
        if missing_keys:
            raise ValueError(
                f"The encoder's `pyramid_outputs` is missing one or more required keys. "
                f"Missing keys: {missing_keys}. "
                f"Required: {set(required_keys)}, Available: {set(pyramid_outputs.keys())}"
            )

        inputs = encoder.input
        skips = [pyramid_outputs[key] for key in required_keys]
        unetr_head = self.build_decoder(
            spatial_dims=spatial_dims,
            num_classes=num_classes,
            feature_size=feature_size,
            res_block=True,
            norm_name=norm_name,
            classifier_activation=classifier_activation,
        )

        # Combine encoder and decoder
        outputs = unetr_head([inputs] + skips)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.window_size = window_size
        self.feature_size = feature_size
        self.res_block = res_block
        self.norm_name = norm_name
        self.encoder_name = encoder_name
        self.encoder = encoder
        self.classifier_activation = classifier_activation

    def build_decoder(
        self,
        spatial_dims,
        num_classes=4,
        feature_size=16,
        res_block=True,
        norm_name="instance",
        classifier_activation=None,
    ):
        """Builds the UNETR-like decoder part of the model.

        Args:
            num_classes (int): The number of segmentation classes. Default is 4.
            feature_size (int): The base feature map size in the decoder. Default is 16.
            res_block (bool): Whether to use residual connections in the decoder blocks.
                Default is True.
            norm_name (str): The type of normalization to use in the decoder blocks
                (e.g., 'instance', 'batch'). Default is "instance".
            classifier_activation (str, optional): The activation function for the final
                classification layer. Default is None.

        Returns:
            callable: A function that takes a list of encoder outputs (including input)
                and skip connections and returns the final segmentation logits.
        """

        def apply(inputs):
            enc_input = inputs[0]
            hidden_states_out = inputs[1:]

            # Encoder 1 (process raw input)
            enc0 = UnetrBasicBlock(
                spatial_dims,
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )(enc_input)

            # Encoder 2 (process hidden_states_out[0])
            enc1 = UnetrBasicBlock(
                spatial_dims,
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )(hidden_states_out[0])

            # Encoder 3 (process hidden_states_out[1])
            enc2 = UnetrBasicBlock(
                spatial_dims,
                out_channels=feature_size * 2,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )(hidden_states_out[1])

            # Encoder 4 (process hidden_states_out[2])
            enc3 = UnetrBasicBlock(
                spatial_dims,
                out_channels=feature_size * 4,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )(hidden_states_out[2])

            # Encoder 5 (process hidden_states_out[4] as bottleneck)
            dec4 = UnetrBasicBlock(
                spatial_dims,
                out_channels=feature_size * 16,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )(hidden_states_out[4])

            # Decoder 5 (upsample dec4 and concatenate with hidden_states_out[3])
            dec3 = UnetrUpBlock(
                spatial_dims,
                out_channels=feature_size * 8,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )(dec4, hidden_states_out[3])

            # Decoder 4 (upsample dec3 and concatenate with enc3)
            dec2 = UnetrUpBlock(
                spatial_dims,
                out_channels=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )(dec3, enc3)

            # Decoder 3 (upsample dec2 and concatenate with enc2)
            dec1 = UnetrUpBlock(
                spatial_dims,
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )(dec2, enc2)

            # Decoder 2 (upsample dec1 and concatenate with enc1)
            dec0 = UnetrUpBlock(
                spatial_dims,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )(dec1, enc1)

            out = UnetrUpBlock(
                spatial_dims,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )(dec0, enc0)

            # Final output (process dec0 and produce logits)
            logits = UnetOutBlock(spatial_dims, num_classes, activation=classifier_activation)(out)
            return logits

        return apply

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "encoder_name": self.encoder_name,
            "num_classes": self.num_classes,
            "patch_size": self.patch_size,
            "window_size": self.window_size,
            "classifier_activation": self.classifier_activation,
            "feature_size": self.feature_size,
            "res_block": self.res_block,
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

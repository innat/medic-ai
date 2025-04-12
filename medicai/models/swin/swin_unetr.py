import os
from functools import partial

from medicai.utils import hide_warnings

hide_warnings()


import keras

from medicai.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock

from .swin_backbone import SwinBackbone


@keras.saving.register_keras_serializable(package="swin.unetr")
class SwinUNETR(keras.Model):
    """Swin-UNETR: A hybrid transformer-CNN for 3D medical image segmentation.

    This model combines the strengths of the Swin Transformer for feature extraction
    and a U-Net-like architecture for segmentation. It uses a Swin Transformer
    backbone to encode the input and a decoder with upsampling and skip connections
    to generate segmentation maps.
    """

    def __init__(
        self,
        *,
        input_shape=(96, 96, 96, 1),
        num_classes=4,
        classifier_activation=None,
        feature_size=48,
        res_block=True,
        norm_name="instance",
        **kwargs
    ):
        """Initializes the SwinUNETR model.

        Args:
            input_shape (tuple): The shape of the input tensor (depth, height, width, channels).
                Default is (96, 96, 96, 1).
            num_classes (int): The number of segmentation classes. Default is 4.
            classifier_activation (str, optional): The activation function for the final
                classification layer (e.g., 'softmax'). If None, no activation is applied.
                Default is None.
            feature_size (int): The base feature map size in the decoder. Default is 48.
            res_block (bool): Whether to use residual connections in the decoder blocks.
                Default is True.
            norm_name (str): The type of normalization to use in the decoder blocks
                (e.g., 'instance', 'batch'). Default is "instance".
            **kwargs: Additional keyword arguments passed to the base Model class.
        """
        encoder = SwinBackbone(
            input_shape=input_shape,
            patch_size=[2, 2, 2],
            depths=[2, 2, 2, 2],
            window_size=[7, 7, 7],
            num_heads=[3, 6, 12, 24],
            embed_dim=48,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            patch_norm=False,
        )
        inputs = encoder.input
        skips = [
            encoder.get_layer("patching_and_embedding").output,
            encoder.get_layer("swin_feature1").output,
            encoder.get_layer("swin_feature2").output,
            encoder.get_layer("swin_feature3").output,
            encoder.get_layer("swin_feature4").output,
        ]
        unetr_head = self.build_decoder(
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
        self.feature_size = feature_size
        self.res_block = res_block
        self.norm_name = norm_name

    def build_decoder(
        self,
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
                feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block
            )(enc_input)

            # Encoder 2 (process hidden_states_out[0])
            enc1 = UnetrBasicBlock(
                feature_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block
            )(hidden_states_out[0])

            # Encoder 3 (process hidden_states_out[1])
            enc2 = UnetrBasicBlock(
                feature_size * 2, kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block
            )(hidden_states_out[1])

            # Encoder 4 (process hidden_states_out[2])
            enc3 = UnetrBasicBlock(
                feature_size * 4, kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block
            )(hidden_states_out[2])

            # Encoder 5 (process hidden_states_out[4] as bottleneck)
            dec4 = UnetrBasicBlock(
                feature_size * 16, kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block
            )(hidden_states_out[4])

            # Decoder 5 (upsample dec4 and concatenate with hidden_states_out[3])
            dec3 = UnetrUpBlock(
                feature_size * 8,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )(dec4, hidden_states_out[3])

            # Decoder 4 (upsample dec3 and concatenate with enc3)
            dec2 = UnetrUpBlock(
                feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )(dec3, enc3)

            # Decoder 3 (upsample dec2 and concatenate with enc2)
            dec1 = UnetrUpBlock(
                feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )(dec2, enc2)

            # Decoder 2 (upsample dec1 and concatenate with enc1)
            dec0 = UnetrUpBlock(
                feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )(dec1, enc1)

            out = UnetrUpBlock(
                feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )(dec0, enc0)

            # Final output (process dec0 and produce logits)
            logits = UnetOutBlock(num_classes, activation=classifier_activation)(out)
            return logits

        return apply

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "feature_size": self.feature_size,
            "res_block": self.res_block,
            "norm_name": self.norm_name,
        }
        return config

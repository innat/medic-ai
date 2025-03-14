import keras
import numpy as np
from keras import ops
from functools import partial

import keras
import numpy as np
from keras import layers
from medicai.models.swin_unetr.swin_unetr_layers import *
from medicai.blocks import UnetBasicBlock
from medicai.blocks import UnetOutBlock
from medicai.blocks import UnetResBlock
from medicai.blocks import UnetrBasicBlock
from medicai.blocks import UnetrUpBlock


@keras.saving.register_keras_serializable(package="swin.unetr")
class SwinUNETR(keras.Model):
    def __init__(
        self,
        *,
        input_shape=(96,96,96,1),
        num_classes=4, 
        feature_size=48, 
        res_block=True, 
        norm_name="instance",
        **kwargs
    ):
        encoder = SwinBackbone(
            input_shape=input_shape,
            patch_size=[2, 2, 2],
            depths=[2, 2, 2, 2],
            window_size=[7, 7, 7],
            num_heads=[3, 6, 12, 24],
            embed_dim=48,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            patch_norm=False
        )
        inputs = encoder.input
        skips = [
            encoder.get_layer("patching_and_embedding").output,
            encoder.get_layer("swin_feature1").output,
            encoder.get_layer("swin_feature2").output,
            encoder.get_layer("swin_feature3").output,
            encoder.get_layer("swin_feature4").output,
        ]
        unetr_head = self.build_unet(
            num_classes=num_classes,
            feature_size=feature_size, 
            res_block=True, 
            norm_name=norm_name, 
        )

        # Combine encoder and decoder
        outputs = unetr_head([inputs] + skips)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.num_classes = num_classes
        self.feature_size = feature_size
        self.res_block = res_block
        self.norm_name = norm_name

    def build_unet(
        self, num_classes=4, feature_size=16, res_block=True, norm_name="instance"
    ):
        def apply(inputs):
            enc_input = inputs[0]
            hidden_states_out = inputs[1:]

            # Encoder 1 (process raw input)
            enc0 = UnetrBasicBlock(
                feature_size, 
                kernel_size=3, 
                stride=1, 
                norm_name=norm_name, 
                res_block=res_block
            )(enc_input)

            # Encoder 2 (process hidden_states_out[0])
            enc1 = UnetrBasicBlock(
                feature_size, 
                kernel_size=3, 
                stride=1, 
                norm_name=norm_name, 
                res_block=res_block
            )(hidden_states_out[0])

            # Encoder 3 (process hidden_states_out[1])
            enc2 = UnetrBasicBlock(
                feature_size * 2, 
                kernel_size=3, 
                stride=1, 
                norm_name=norm_name, 
                res_block=res_block
            )(hidden_states_out[1])

            # Encoder 4 (process hidden_states_out[2])
            enc3 = UnetrBasicBlock(
                feature_size * 4, 
                kernel_size=3, 
                stride=1, 
                norm_name=norm_name, 
                res_block=res_block
            )(hidden_states_out[2])

            # Encoder 5 (process hidden_states_out[4] as bottleneck)
            dec4 = UnetrBasicBlock(
                feature_size * 16, 
                kernel_size=3, 
                stride=1, 
                norm_name=norm_name, 
                res_block=res_block
            )(hidden_states_out[4])

            # Decoder 5 (upsample dec4 and concatenate with hidden_states_out[3])
            dec3 = UnetrUpBlock(
                feature_size * 8, 
                kernel_size=3, 
                upsample_kernel_size=2,
                norm_name=norm_name, 
                res_block=res_block
            )(dec4, hidden_states_out[3])

            # Decoder 4 (upsample dec3 and concatenate with enc3)
            dec2 = UnetrUpBlock(
                feature_size * 4, 
                kernel_size=3, 
                upsample_kernel_size=2,
                norm_name=norm_name, 
                res_block=res_block
            )(dec3, enc3)

            # Decoder 3 (upsample dec2 and concatenate with enc2)
            dec1 = UnetrUpBlock(
                feature_size * 2, 
                kernel_size=3, 
                upsample_kernel_size=2,
                norm_name=norm_name, 
                res_block=res_block
            )(dec2, enc2)

            # Decoder 2 (upsample dec1 and concatenate with enc1)
            dec0 = UnetrUpBlock(
                feature_size, 
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name, 
                res_block=res_block
            )(dec1, enc1)

            out = UnetrUpBlock(
                feature_size, 
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name, 
                res_block=res_block
            )(dec0, enc0)

            # Final output (process dec0 and produce logits)
            logits = UnetOutBlock(num_classes)(out)
            return logits
        return apply

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "num_classes": self.num_classes,
            "feature_size": self.feature_size,
            "res_block": self.res_block,
            "norm_name": self.norm_name
        }
        return config

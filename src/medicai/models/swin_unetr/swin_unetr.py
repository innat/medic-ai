import keras
import numpy as np
from keras import ops
from functools import partial

import keras
import numpy as np
from keras import layers
from .swin_unetr_layers import *


@keras.saving.register_keras_serializable(package="swin.unetr")
class SwinUNETR(keras.Model):
    def __init__(
        self,
        *,
        input_shape=(96,96,96,1),
        out_channels=4, 
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
        unetr_head = UnetrHead(
            out_channels=out_channels,
            feature_size=feature_size, 
            res_block=True, 
            norm_name=norm_name, 
        )
        
        # Combine encoder and decoder
        outputs = unetr_head([inputs] + skips)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.out_channels = out_channels
        self.feature_size = feature_size
        self.res_block = res_block
        self.norm_name = norm_name

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "out_channels": self.out_channels,
            "feature_size": self.feature_size,
            "res_block": self.res_block,
            "norm_name": self.norm_name
        }
        return config

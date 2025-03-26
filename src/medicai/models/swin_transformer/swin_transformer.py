
from medicai.utils import hide_warnings
hide_warnings()

import keras
from medicai.models.swin_unetr.swin_unetr_layers import *

@keras.saving.register_keras_serializable(package="swin.transformer")
class SwinTransformer(keras.Model):
    def __init__(
        self,
        *,
        input_shape=(96, 96, 96, 1),
        num_classes=4,
        classifier_activation=None,
        **kwargs
    ):
        inputs = keras.Input(shape=input_shape)
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
        )(inputs)

        x = keras.layers.GlobalAveragePooling3D()(encoder)
        outputs = keras.layers.Dense(
            num_classes,
            activation=classifier_activation
        )(x)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
        }
        return config

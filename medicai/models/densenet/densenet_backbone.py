import keras
from keras import layers

from ...layers.densenet import DenseBlock3D, TransitionLayer3D


def parse_model_inputs(input_shape, input_tensor, **kwargs):
    if input_tensor is None:
        return keras.layers.Input(shape=input_shape, **kwargs)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            return keras.layers.Input(tensor=input_tensor, shape=input_shape, **kwargs)
        else:
            return input_tensor


class DenseNet3DBackbone(keras.Model):
    def __init__(
        self,
        *,
        blocks,
        input_shape=(None, None, None, 1),
        input_tensor=None,
        growth_rate=32,
        bn_size=4,
        compression=0.5,
        dropout_rate=0.0,
        include_rescaling=False,
        name="densenet3d_backbone",
        **kwargs
    ):
        """Builds a functional DenseNet3D backbone"""
        inputs = parse_model_inputs(input_shape, input_tensor)

        x = inputs
        if include_rescaling:
            x = layers.Rescaling(1.0 / 255)(x)

        # Initial convolution stem
        x = layers.Conv3D(64, kernel_size=7, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling3D(pool_size=3, strides=2, padding="same")(x)

        # Dense blocks and transitions
        num_channels = 64

        for i, num_layers in enumerate(blocks):
            x = DenseBlock3D(x, num_layers, growth_rate, bn_size, dropout_rate)
            num_channels += num_layers * growth_rate

            if i != len(blocks) - 1:
                out_channels = int(num_channels * compression)
                x = TransitionLayer3D(x, out_channels)
                num_channels = out_channels

        # Final batch norm
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        self.blocks = blocks
        self.input_tensor = input_tensor
        self.growth_rate = (growth_rate,)
        self.bn_size = (bn_size,)
        self.compression = (compression,)
        self.dropout_rate = (dropout_rate,)
        self.include_rescaling = (include_rescaling,)
        self.name = name

    def get_config(self):
        config = {
            "input_shape": self.input_shape[1:],
            "input_tensor": self.input_tensor,
            "blocks": self.blocks,
            "growth_rate": self.growth_rate,
            "bn_size": self.bn_size,
            "compression": self.compression,
            "dropout_rate": self.dropout_rate,
            "include_rescaling": self.include_rescaling,
            "name": self.name,
        }
        return config

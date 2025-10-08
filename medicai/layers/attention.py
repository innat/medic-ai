from medicai.utils.general import hide_warnings

hide_warnings()

import keras
from keras import layers, ops

from medicai.utils.model_utils import get_conv_layer


class AttentionGate(keras.layers.Layer):
    """https://arxiv.org/abs/1804.03999"""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        # variable
        self.filters = filters
        # ops
        self.relu = layers.Activation("relu")
        self.sigmoid = layers.Activation("sigmoid")
        self.mul = layers.Multiply()
        self.add = layers.Add()

    def build(self, input_shape):
        self.spatial_dims = len(input_shape[0]) - 2
        if self.spatial_dims not in (2, 3):
            raise ValueError(
                f"AttentionGate only supports 2D or 3D inputs. Got spatial_dims={self.spatial_dims}"
            )

        # input_shape = [x_shape, g_shape]
        self.theta_x = get_conv_layer(
            self.spatial_dims,
            layer_type="conv",
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding="same",
        )
        self.phi_g = get_conv_layer(
            self.spatial_dims,
            layer_type="conv",
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding="same",
        )
        self.psi = get_conv_layer(
            self.spatial_dims,
            layer_type="conv",
            filters=1,
            kernel_size=1,
            strides=1,
            padding="same",
        )
        self.built = True

    def call(self, inputs):
        # expect list [x, g]
        x, g = inputs

        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)
        add = self.add([theta_x, phi_g])

        relu = self.relu(add)
        psi = self.psi(relu)
        alpha = self.sigmoid(psi)

        return self.mul([x, alpha])  # gated skip

    def compute_output_shape(self, input_shape):
        # output has the same shape as skip connection (x)
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
            }
        )
        return config

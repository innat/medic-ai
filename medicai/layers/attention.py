from medicai.utils.general import hide_warnings

hide_warnings()

import keras
from keras import layers, ops

from medicai.utils.model_utils import get_conv_layer


class AttentionGate(keras.layers.Layer):
    """https://arxiv.org/abs/1804.03999"""

    def __init__(self, filters, spatial_dims=2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.spatial_dims = spatial_dims
        # ops
        self.relu = layers.Activation("relu")
        self.sigmoid = layers.Activation("sigmoid")
        self.mul = layers.Multiply()
        self.add = layers.Add()

    def build(self, input_shape):
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
                "spatial_dims": self.spatial_dims,
            }
        )
        return config


class ChannelWiseAttention(layers.Layer):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        dims = input_shape.shape[-1]
        # squeeze
        self.gap = layers.GlobalAveragePooling2D()
        # excitation
        self.fc0 = layers.Dense(int(self.alpha * dims), use_bias=False, activation="relu")
        self.fc1 = layers.Dense(dims, use_bias=False, activation="sigmoid")
        self.rs = layers.Reshape((1, 1, dims))

    def call(self, inputs):
        # calculate channel-wise attention vector
        z = self.gap(inputs)
        u = self.fc0(z)
        u = self.fc1(u)
        u = self.rs(u)
        return u * inputs


class ElementWiseAttention(layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes

        self.conv0 = layers.Conv2D(
            512,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            activation=ops.relu,
        )
        self.conv1 = layers.Conv2D(
            512,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            activation=ops.relu,
        )
        self.conv2 = layers.Conv2D(
            self.num_classes,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            activation=ops.softmax,
        )

        # linear classifier
        self.linear = layers.Conv2D(
            self.num_classes,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            activation=None,
        )

    def call(self, inputs):
        # f(att)
        a = self.conv0(inputs)
        a = self.conv1(a)
        a = self.conv2(a)
        # confidence score
        s = self.linear(inputs)
        # element-wise multiply to prevent unnecessary attention
        m = s * a
        return m

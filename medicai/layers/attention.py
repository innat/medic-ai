import keras
from keras import layers

from medicai.utils.model_utils import (
    get_act_layer,
    get_conv_layer,
    get_pooling_layer,
    get_reshaping_layer,
)


class AttentionGate(keras.layers.Layer):
    """https://arxiv.org/abs/1804.03999"""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        # variable
        self.filters = filters

        # ops
        self.relu = get_act_layer("relu")
        self.sigmoid = get_act_layer("sigmoid")
        self.mul = layers.Multiply()
        self.add = layers.Add()

    def build(self, input_shape):
        self.spatial_dims = len(input_shape[0]) - 2
        if self.spatial_dims not in (2, 3):
            raise ValueError(
                f"{self.__class__.__name__} only supports 2D or 3D inputs. Got spatial_dims={self.spatial_dims}"
            )

        # input_shape = [x_shape, g_shape]
        x_shape, g_shape = input_shape
        self.theta_x = get_conv_layer(
            self.spatial_dims,
            layer_type="conv",
            filters=self.filters,
            kernel_size=1,
            strides=2,
            padding="same",
        )
        self.theta_x.build(x_shape)

        self.upsampling = get_reshaping_layer(
            self.spatial_dims,
            layer_type="upsampling",
            size=2,
        )

        self.phi_g = get_conv_layer(
            self.spatial_dims,
            layer_type="conv",
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding="same",
        )
        self.phi_g.build(g_shape)

        self.psi = get_conv_layer(
            self.spatial_dims,
            layer_type="conv",
            filters=1,
            kernel_size=1,
            strides=1,
            padding="same",
        )
        psi_shape = self.add.compute_output_shape(
            [self.theta_x.compute_output_shape(x_shape), self.phi_g.compute_output_shape(g_shape)]
        )
        self.psi.build(psi_shape)

        self.built = True

    def call(self, inputs):
        # expect list [skip, input]
        x, g = inputs

        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)
        add = self.add([theta_x, phi_g])

        relu = self.relu(add)
        psi = self.psi(relu)
        alpha = self.sigmoid(psi)
        alpha = self.upsampling(alpha)
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


class SqueezeExcitation(keras.layers.Layer):
    def __init__(self, ratio=16, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.activation = activation

    def build(self, input_shape):
        spatial_dims = len(input_shape) - 2
        self.channels = input_shape[-1]
        self.reduced_channels = max(1, self.channels // self.ratio)

        # Global average pooling
        self.global_pool = get_pooling_layer(
            spatial_dims=spatial_dims, layer_type="avg", global_pool=True, keepdims=True
        )
        self.global_pool.build(input_shape)

        # Build excitation layers using 1x1 (or 1x1x1) convolutions
        self.conv1 = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=self.reduced_channels,
            kernel_size=1,
            activation=self.activation,
        )
        pooled_shape_conv1 = (input_shape[0],) + (1,) * spatial_dims + (self.channels,)
        self.conv1.build(pooled_shape_conv1)

        self.conv2 = get_conv_layer(
            spatial_dims=spatial_dims,
            layer_type="conv",
            filters=self.channels,
            kernel_size=1,
            activation="sigmoid",
        )
        pooled_shape_conv2 = self.conv1.compute_output_shape(pooled_shape_conv1)
        self.conv2.build(pooled_shape_conv2)

        super().build(input_shape)

    def call(self, inputs):
        # Squeeze: Global average pooling
        se = self.global_pool(inputs)

        # Excitation: Two 1x1 convolutional layers
        se = self.conv1(se)
        se = self.conv2(se)

        # Scale the input features (broadcasting automatically handles dimensions)
        return inputs * se

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "ratio": self.ratio,
                "activation": self.activation,
            }
        )
        return config

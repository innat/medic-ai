import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ChannelWiseAttention(layers.Layer):
    def __init__(self, config):
        super().__init__()

        D = 480
        ALPHA = 1 / 16

        # squeeze
        self.gap = layers.GlobalAveragePooling2D()
        # excitation
        self.fc0 = layers.Dense(int(ALPHA * D), use_bias=False, activation=tf.nn.relu)
        self.fc1 = layers.Dense(D, use_bias=False, activation=tf.nn.sigmoid)
        # reshape so we can do channel-wise multiplication
        self.rs = layers.Reshape((1, 1, D))

    def call(self, inputs):
        # calculate channel-wise attention vector
        z = self.gap(inputs)
        u = self.fc0(z)
        u = self.fc1(u)
        u = self.rs(u)
        return u * inputs


class ElementWiseAttention(layers.Layer):
    def __init__(self, config):
        super().__init__()

        C = config.dataset.num_classes

        self.conv0 = layers.Conv2D(
            512,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            activation=tf.nn.relu,
        )
        self.conv1 = layers.Conv2D(
            512,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            activation=tf.nn.relu,
        )
        self.conv2 = layers.Conv2D(
            C,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            activation=tf.nn.softmax,
        )

        # linear classifier
        self.linear = layers.Conv2D(
            C,
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

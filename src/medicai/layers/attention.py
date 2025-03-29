from medicai.utils.general import hide_warnings

hide_warnings()

from keras import layers, ops


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

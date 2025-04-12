from medicai.utils.general import hide_warnings

hide_warnings()
from keras import layers


class SwinMLP(layers.Layer):
    """A multi-layer perceptron (MLP) block used in Swin-MLP architecture.

    This layer consists of two fully connected (Dense) layers with a GELU
    activation in between, followed by dropout layers. It's a fundamental
    building block in the Swin-MLP network, replacing the traditional
    multi-head self-attention mechanism found in standard Transformer
    architectures.
    """

    def __init__(self, hidden_dim, output_dim, drop_rate=0.0, activation="gelu", **kwargs):
        """Initializes the SwinMLP layer.

        Args:
            hidden_dim (int): The number of units in the hidden layer.
            output_dim (int): The number of output units.
            drop_rate (float): The dropout rate (between 0 and 1). Default is 0.0.
            activation (str): The activation function to use. Default is "gelu".
            **kwargs: Additional keyword arguments passed to the base Layer class.
        """
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self._activation_identifier = activation
        self.drop_rate = drop_rate
        self.activation = layers.Activation(self._activation_identifier)
        self.fc1 = layers.Dense(self.hidden_dim)
        self.fc2 = layers.Dense(self.output_dim)
        self.dropout = layers.Dropout(self.drop_rate)

    def build(self, input_shape):
        self.fc1.build(input_shape)
        self.fc2.build((*input_shape[:-1], self.hidden_dim))
        self.built = True

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "hidden_dim": self.hidden_dim,
                "drop_rate": self.drop_rate,
                "activation": self._activation_identifier,
            }
        )
        return config

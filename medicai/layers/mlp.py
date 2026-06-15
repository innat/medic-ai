from keras import layers


class MLPBlock(layers.Layer):
    """
    Multi-Layer Perceptron (MLP) block that implements a simple feed-forward network commonly used in
    Transformer-style architectures and modern deep learning models. It consists of:

    1. A fully connected (Dense) layer that expands or transforms features
       into a higher-dimensional hidden space.
    2. A non-linear activation function (e.g., ``GELU``).
    3. Dropout for regularization.
    4. A second Dense layer that projects features back to the output
       dimension.
    5. A final dropout layer.

    Args:
        hidden_dim (int): Number of hidden units in the intermediate Dense layer.
        output_dim (int): Dimensionality of the final output features.
        drop_rate (float, optional): Dropout probability applied after activation and after the
            second Dense layer. Defaults to ``0.0``.
        activation (str, optional): Activation function applied after the first Dense layer.
            Common choices include ``"gelu"``, ``"relu"``, etc. Defaults to ``"gelu"``.
        **kwargs:
            Additional keyword arguments passed to ``keras.layers.Layer``.

    Example:
        .. code-block:: python

            import numpy as np
            from medicai.layers import MLPBlock

            x = np.random.randn(1, 128, 256).astype(np.float32)
            mlp = MLPBlock(
                hidden_dim=512,
                output_dim=256,
                drop_rate=0.1,
                activation="gelu",
            )
            y = mlp(x)
            print(y.shape) # (1, 128, 256)

    Returns:
        keras.KerasTensor: Output tensor of the same shape as the input
        ``(..., output_dim)``, where all leading dimensions are preserved
        and only the last dimension is projected to ``output_dim``.

    Raises:
        ValueError: If the input tensor has fewer than ``2`` dimensions,
            since the Dense layers require at least a batch and feature axis.
    """

    def __init__(self, hidden_dim, output_dim, drop_rate=0.0, activation="gelu", **kwargs):
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


class ViTMLP(MLPBlock): ...


class SwinMLP(MLPBlock): ...


class TransUNetMLP(MLPBlock): ...

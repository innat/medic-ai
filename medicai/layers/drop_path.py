from medicai.utils.general import hide_warnings

hide_warnings()

import keras
from keras import layers, ops


class DropPath(layers.Layer):
    """Stochastically drops entire paths in a neural network during training.

    This layer implements the DropPath technique, which randomly sets the output
    of a layer to zero with a given probability (drop rate) during training.
    This can help to improve the generalization ability of the network by
    reducing overfitting. The scaling of the remaining paths ensures that the
    expected output remains the same.
    """

    def __init__(self, rate=0.5, seed=None, **kwargs):
        """Initializes the DropPath layer.

        Args:
            rate (float): The probability of dropping a path (a value between 0 and 1).
                Default is 0.5.
            seed (int, optional): A random seed to ensure deterministic behavior.
                If None, a random seed is used. Default is None.
            **kwargs: Additional keyword arguments passed to the base Layer class.
        """
        super().__init__(**kwargs)
        self.rate = rate
        self._seed_val = seed
        self.seed = keras.random.SeedGenerator(seed)

    def call(self, x, training=None):
        """Applies the DropPath operation to the input tensor.

        Args:
            x (tf.Tensor): The input tensor.
            training (bool, optional): A boolean tensor indicating whether the layer
                is being called during training or inference. If None, it defaults
                to `keras.backend.learning_phase()`.

        Returns:
            tf.Tensor: The output tensor after applying DropPath. During training,
            some paths are dropped and the remaining are scaled. During inference,
            the input tensor is returned unchanged if the drop rate is greater than 0.
        """
        if self.rate == 0.0 or not training:
            return x
        else:
            batch_size = x.shape[0] or ops.shape(x)[0]
            drop_map_shape = (batch_size,) + (1,) * (len(x.shape) - 1)
            drop_map = ops.cast(
                keras.random.uniform(drop_map_shape, seed=self.seed) > self.rate,
                x.dtype,
            )
            x = x / (1.0 - self.rate)
            x = x * drop_map
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate, "seed": self._seed_val})
        return config

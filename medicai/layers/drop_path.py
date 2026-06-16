import keras
from keras import layers, ops


class DropPath(layers.Layer):
    """
    Stochastically drops entire paths in a neural network during training. This layer implements the ``DropPath`` technique, which randomly sets the output
    of a layer to zero with a given probability (drop rate) during training.
    This can help to improve the generalization ability of the network by
    reducing overfitting. The scaling of the remaining paths ensures that the
    expected output remains the same.

    Args:
        rate (float): The probability of dropping a path (a value between 0 and 1).
            Default is ``0.5``.
        seed (int, optional): A random seed to ensure deterministic behavior.
            If ``None``, a random seed is used. Default is ``None``.
        **kwargs: Additional keyword arguments passed to the base Layer class.
    
    Example:
        .. code-block:: python    
        
            import keras
            import numpy as np

            x = keras.ops.ones((4, 8))
            layer = DropPath(rate=0.5, seed=42)
            y_train = layer(x, training=True)
            y_test = layer(x, training=False)

            print(y_train)
            # tf.Tensor(
            #     [[0. 0. 0. 0. 0. 0. 0. 0.]
            #     [0. 0. 0. 0. 0. 0. 0. 0.]
            #     [2. 2. 2. 2. 2. 2. 2. 2.]
            #     [0. 0. 0. 0. 0. 0. 0. 0.]], shape=(4, 8), dtype=float32
            # )

            print(y_test)
            # tf.Tensor(
            #     [[1. 1. 1. 1. 1. 1. 1. 1.]
            #     [1. 1. 1. 1. 1. 1. 1. 1.]
            #     [1. 1. 1. 1. 1. 1. 1. 1.]
            #     [1. 1. 1. 1. 1. 1. 1. 1.]], shape=(4, 8), dtype=float32
            # )

    Returns:
        ``keras.KerasTensor``: Output tensor of the same shape as the input.
        During training, entire sample paths are randomly zeroed with
        probability ``rate`` and the surviving paths are scaled by
        ``1 / (1 - rate)`` to preserve expected output magnitude.
        During inference, the input is returned unchanged regardless
        of ``rate``.

    Raises:
        ValueError: If ``rate`` is not in the range ``[0, 1)``, since a
            drop rate of ``1.0`` would zero all paths unconditionally.
    """

    def __init__(self, rate=0.5, seed=None, **kwargs):
        super().__init__(**kwargs)

        if not 0.0 <= rate < 1.0:
            raise ValueError(f"`rate` must be in [0, 1), got {rate}.")

        self.rate = rate
        self._seed_val = seed
        self.seed = keras.random.SeedGenerator(seed)

    def call(self, x, training=None):
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

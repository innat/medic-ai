from typing import Sequence

import tensorflow as tf

from ..base import RandomTransform
from ..tensor_bundle import TensorBundle
from ..spatial.rotate90 import Rotate90


class RandomRotate90(RandomTransform):
    """Randomly rotate tensors by multiples of 90 degrees."""

    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.1,
        max_k: int = 3,
        spatial_axes: Sequence[int] | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob)
        self.keys = tuple(keys)
        self.max_k = max_k
        self.spatial_axes = spatial_axes
        self.allow_missing_keys = allow_missing_keys

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        should_rotate = self.sample_should_apply()
        k = tf.random.uniform([], minval=1, maxval=self.max_k + 1, dtype=tf.int32)
        rotate = Rotate90(
            keys=self.keys,
            k=1,
            spatial_axes=self.spatial_axes,
            allow_missing_keys=self.allow_missing_keys,
        )

        present_keys = rotate.apply_to_present_keys(
            bundle,
            lambda tensor, _: tf.cond(
                should_rotate,
                lambda tensor=tensor: rotate.rotate_tensor(tensor, k=k),
                lambda tensor=tensor: tensor,
            ),
        )
        self.record_random_transform(
            bundle,
            params={
                "keys": list(present_keys),
                "k": k,
                "spatial_axes": self.spatial_axes,
            },
            applied=should_rotate,
            kernel="Rotate90",
        )
        return bundle

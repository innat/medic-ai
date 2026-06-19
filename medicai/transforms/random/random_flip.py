from typing import Sequence, Union

import tensorflow as tf

from ..base import RandomTransform
from ..tensor_bundle import TensorBundle
from ..spatial.flip import Flip


class RandomFlip(RandomTransform):
    """Randomly flip selected tensors along specified axes."""

    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.1,
        spatial_axis: Union[int, Sequence[int], None] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob)
        self.flip = Flip(
            keys=keys,
            spatial_axis=spatial_axis,
            allow_missing_keys=allow_missing_keys,
        )

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        should_flip = self.sample_should_apply()

        if self.flip.spatial_axis is None:
            return bundle

        present_keys = self.flip.apply_to_present_keys(
            bundle,
            lambda tensor, _: tf.cond(
                should_flip,
                lambda tensor=tensor: self.flip.flip_tensor(tensor),
                lambda tensor=tensor: tensor,
            ),
        )
        self.record_random_transform(
            bundle,
            params={
                "keys": list(present_keys),
                "spatial_axis": self.flip.spatial_axis,
            },
            applied=should_flip,
            kernel="Flip",
        )
        return bundle

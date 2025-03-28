from typing import Sequence, Union

from medicai.utils.general import hide_warnings

hide_warnings()
import tensorflow as tf

from src.medicai.transforms.tensor_bundle import MetaTensor


class RandFlip:
    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.1,
        spatial_axis: Union[int, Sequence[int], None] = None,
    ):
        self.keys = keys
        self.prob = prob
        self.spatial_axis = spatial_axis

    def __call__(self, inputs: MetaTensor) -> MetaTensor:
        should_flip = tf.random.uniform([]) < self.prob

        for key in self.keys:
            if key in inputs.data:
                if should_flip:
                    inputs.data[key] = tf.reverse(inputs.data[key], axis=self.spatial_axis)
        return inputs

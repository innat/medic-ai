
import tensorflow as tf
from typing import Tuple, Union, Sequence
from medicai.transforms import MetaTensor

class RandShiftIntensity:
    def __init__(self, keys: Sequence[str], offsets: Union[float, Tuple[float, float]], 
                 prob: float = 0.1, channel_wise: bool = False):
        """
        Args:
            keys: list of keys to apply the transform to.
            offsets: Tuple (min_offset, max_offset) or a single float.
            prob: Probability of applying the intensity shift.
            channel_wise: If True, applies different shifts per channel.
        """
        self.keys = keys
        if isinstance(offsets, (int, float)):
            self.offsets = (-abs(offsets), abs(offsets))
        else:
            self.offsets = (min(offsets), max(offsets))

        self.prob = prob
        self.channel_wise = channel_wise

    def __call__(self, inputs: MetaTensor) -> MetaTensor:
        rand_val = tf.random.uniform(())

        def apply_shift():
            shifted_data = inputs.data.copy()
            for key in self.keys:
                if key in shifted_data:
                    img = shifted_data[key]
                    if self.channel_wise and len(img.shape) == 4:
                        offsets = tf.random.uniform((1, 1, 1, img.shape[-1]), self.offsets[0], self.offsets[1])
                    else:
                        offsets = tf.random.uniform((), self.offsets[0], self.offsets[1])
                    shifted_data[key] = img + offsets
            return shifted_data

        def no_shift():
            return inputs.data.copy()

        shifted_data = tf.cond(rand_val <= self.prob, apply_shift, no_shift)
        return MetaTensor(shifted_data, inputs.meta)
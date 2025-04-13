from medicai.utils.general import hide_warnings

hide_warnings()

from typing import Dict, Sequence, Tuple, Union

import tensorflow as tf

from .tensor_bundle import TensorBundle


class RandShiftIntensity:
    """Randomly shifts the intensity values of specified tensors.

    This transform applies a random offset to the intensity of the tensors
    specified by `keys`. The offset is sampled from a uniform distribution
    defined by the `offsets` parameter. The shift is applied with a given
    probability. It can be applied globally to the entire tensor or independently
    for each channel.
    """

    def __init__(
        self,
        keys: Sequence[str],
        offsets: Union[float, Tuple[float, float]],
        prob: float = 0.1,
        channel_wise: bool = False,
    ):
        """
        Initializes the RandShiftIntensity transform.

        Args:
            keys (Sequence[str]): List of keys in the input TensorBundle to apply the transform to.
            offsets (Union[float, Tuple[float, float]]): The range from which the intensity
                shift will be sampled.
                - If a single float is provided, the offset will be uniformly sampled
                  from `(-abs(offsets), abs(offsets))`.
                - If a tuple `(min_offset, max_offset)` is provided, the offset will be
                  uniformly sampled from this range.
            prob (float): The probability (between 0 and 1) of applying the intensity shift.
                Default is 0.1.
            channel_wise (bool): If True and the input tensor has a channel dimension (e.g., shape [D, H, W, C]),
                a different random offset will be applied to each channel. If False, a single
                random offset is applied to all channels (or to the entire tensor if it has no
                channel dimension). Default is False.
        """
        self.keys = keys
        if isinstance(offsets, (int, float)):
            self.offsets = (-abs(offsets), abs(offsets))
        else:
            self.offsets = (min(offsets), max(offsets))

        self.prob = prob
        self.channel_wise = channel_wise

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:
        """Apply the random intensity shift to the specified tensors in the input TensorBundle.

        Args:
            inputs (TensorBundle): A dictionary containing tensors and metadata. The tensors
                specified by `self.keys` will have a random intensity shift applied with
                probability `self.prob`.

        Returns:
            TensorBundle: A dictionary with the intensity-shifted tensors (if the random
            condition is met) and the original metadata.
        """

        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        rand_val = tf.random.uniform(())

        def apply_shift():
            shifted_data = inputs.data.copy()
            for key in self.keys:
                if key in shifted_data:
                    img = shifted_data[key]
                    if self.channel_wise and len(img.shape) == 4:
                        offsets = tf.random.uniform(
                            (1, 1, 1, img.shape[-1]), self.offsets[0], self.offsets[1]
                        )
                    else:
                        offsets = tf.random.uniform((), self.offsets[0], self.offsets[1])
                    shifted_data[key] = img + offsets
            return shifted_data

        def no_shift():
            return inputs.data.copy()

        shifted_data = tf.cond(rand_val <= self.prob, apply_shift, no_shift)
        return TensorBundle(shifted_data, inputs.meta)

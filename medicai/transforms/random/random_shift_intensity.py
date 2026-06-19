from typing import Sequence, Tuple, Union

import tensorflow as tf

from ..base import RandomTransform
from ..tensor_bundle import TensorBundle
from ..intensity.shift_intensity import ShiftIntensity


class RandomShiftIntensity(RandomTransform):
    """Randomly shift the intensity values of specified tensors."""

    def __init__(
        self,
        keys: Sequence[str],
        offsets: Union[float, Tuple[float, float]],
        prob: float = 0.1,
        channel_wise: bool = False,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob)
        self.keys = tuple(keys)
        if isinstance(offsets, (int, float)):
            self.offsets = (-abs(offsets), abs(offsets))
        else:
            self.offsets = (min(offsets), max(offsets))

        self.channel_wise = channel_wise
        self.allow_missing_keys = allow_missing_keys

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        should_shift = self.sample_should_apply()
        shift = ShiftIntensity(keys=self.keys, offsets=0.0, allow_missing_keys=self.allow_missing_keys)
        sampled_offsets = {}
        present_keys = shift.iter_present_keys(bundle)

        def apply_shift(tensor: tf.Tensor, key: str) -> tf.Tensor:
            if self.channel_wise:
                offset_shape = [1] * (tensor.shape.rank - 1) + [tensor.shape[-1]]
                offsets = tf.random.uniform(
                    shape=offset_shape,
                    minval=self.offsets[0],
                    maxval=self.offsets[1],
                    dtype=tensor.dtype,
                )
            else:
                offsets = tf.random.uniform(
                    shape=(),
                    minval=self.offsets[0],
                    maxval=self.offsets[1],
                    dtype=tensor.dtype,
                )
            sampled_offsets[key] = offsets
            return tf.cond(
                should_shift,
                lambda tensor=tensor, offsets=offsets: shift.shift_tensor(tensor, offsets=offsets),
                lambda tensor=tensor: tensor,
            )

        shift.apply_to_present_keys(bundle, apply_shift, keys=present_keys)
        self.record_random_transform(
            bundle,
            params={
                "keys": list(sampled_offsets.keys()),
                "channel_wise": self.channel_wise,
                "offsets": sampled_offsets,
            },
            applied=should_shift,
            kernel="ShiftIntensity",
        )
        return bundle

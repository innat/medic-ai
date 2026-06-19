from typing import Sequence, Union

import tensorflow as tf

from ..base import KeyedTransform
from ..tensor_bundle import TensorBundle


class ShiftIntensity(KeyedTransform):
    """Deterministically shift tensor intensities by an additive offset.

    Args:
        keys: Keys of the tensors to shift.
        offsets: Scalar offset or per-channel offset tensor broadcastable to
            the selected tensors.
        allow_missing_keys: If ``True``, missing keys are skipped.
    """

    def __init__(
        self,
        keys: Sequence[str],
        offsets: Union[float, tf.Tensor],
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.offsets = offsets

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        self.apply_to_present_keys(bundle, lambda tensor, _: self.shift_tensor(tensor))
        return bundle

    def shift_tensor(self, tensor: tf.Tensor, offsets: Union[float, tf.Tensor, None] = None) -> tf.Tensor:
        """Shift one tensor by a scalar or broadcastable offset."""
        offset = tf.cast(self.offsets if offsets is None else offsets, dtype=tensor.dtype)
        return tensor + offset

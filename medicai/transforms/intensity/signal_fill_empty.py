from typing import Sequence

import tensorflow as tf

from ..base import KeyedTransform
from ..tensor_bundle import TensorBundle


class SignalFillEmpty(KeyedTransform):
    """Fill NaN, positive infinity, and negative infinity values in selected tensors."""

    def __init__(
        self,
        keys: Sequence[str],
        replacement: float = 0.0,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.replacement = replacement

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present_keys = self.apply_to_present_keys(bundle, lambda tensor, _: self.nan_to_num(tensor))
        bundle.push_transform(
            self.build_trace_entry(
                params={"keys": list(present_keys), "replacement": self.replacement},
                applied=True,
                random=False,
                invertible=False,
            )
        )
        return bundle

    def nan_to_num(
        self,
        tensor: tf.Tensor,
        nan: float | None = None,
        posinf: float | None = None,
        neginf: float | None = None,
    ) -> tf.Tensor:
        """Replace NaN, positive infinity, and negative infinity values in a tensor."""
        original_dtype = tensor.dtype
        tensor = tf.cast(tensor, tf.float32)

        nan = self.replacement if nan is None else nan
        posinf = tf.float32.max if posinf is None else posinf
        neginf = -tf.float32.max if neginf is None else neginf

        tensor = tf.where(tf.math.is_nan(tensor), tf.fill(tf.shape(tensor), tf.cast(nan, tf.float32)), tensor)
        tensor = tf.where(
            tf.math.is_inf(tensor) & (tensor > 0),
            tf.fill(tf.shape(tensor), tf.cast(posinf, tf.float32)),
            tensor,
        )
        tensor = tf.where(
            tf.math.is_inf(tensor) & (tensor < 0),
            tf.fill(tf.shape(tensor), tf.cast(neginf, tf.float32)),
            tensor,
        )
        return tf.cast(tensor, original_dtype)

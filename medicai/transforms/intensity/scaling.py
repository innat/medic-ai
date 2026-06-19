from typing import Optional, Sequence

import tensorflow as tf

from ..base import KeyedTransform
from ..tensor_bundle import TensorBundle


class ScaleIntensityRange(KeyedTransform):
    """Linearly scale selected tensor intensities to a target range."""

    def __init__(
        self,
        keys: Sequence[str],
        a_min: float,
        a_max: float,
        b_min: Optional[float] = None,
        b_max: Optional[float] = None,
        clip: bool = False,
        dtype: tf.DType = tf.float32,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.dtype = dtype

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present_keys = self.apply_to_present_keys(bundle, lambda tensor, _: self.scale_tensor(tensor))
        bundle.push_transform(
            self.build_trace_entry(
                params={
                    "keys": list(present_keys),
                    "a_min": self.a_min,
                    "a_max": self.a_max,
                    "b_min": self.b_min,
                    "b_max": self.b_max,
                    "clip": self.clip,
                },
                applied=True,
                random=False,
                invertible=False,
            )
        )
        return bundle

    def scale_tensor(self, tensor: tf.Tensor) -> tf.Tensor:
        """Scale one tensor from source range to target range."""
        tensor = tf.convert_to_tensor(tensor, dtype=self.dtype)
        if self.a_max == self.a_min:
            result = tensor - self.a_min if self.b_min is None else tensor - self.a_min + self.b_min
            return tf.cast(result, dtype=self.dtype)

        tensor = (tensor - self.a_min) / (self.a_max - self.a_min)
        if self.b_min is not None and self.b_max is not None:
            tensor = tensor * (self.b_max - self.b_min) + self.b_min
        if self.clip and self.b_min is not None and self.b_max is not None:
            tensor = tf.clip_by_value(tensor, self.b_min, self.b_max)
        return tf.cast(tensor, dtype=self.dtype)

from medicai.utils.general import hide_warnings
hide_warnings()

from typing import *

import tensorflow as tf

from medicai.transforms.meta_tensor import MetaTensor


class ScaleIntensityRange:
    def __init__(
        self,
        keys: Sequence[str],
        a_min: float,
        a_max: float,
        b_min: Optional[float] = None,
        b_max: Optional[float] = None,
        clip: bool = False,
        dtype: tf.DType = tf.float32,
    ):
        self.keys = keys
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.dtype = dtype

    def __call__(self, inputs: MetaTensor) -> MetaTensor:
        for key in self.keys:
            if key in inputs.data:
                inputs.data[key] = self.scale_intensity_range(inputs.data[key])
        return inputs

    def scale_intensity_range(self, image: tf.Tensor) -> tf.Tensor:
        image = tf.convert_to_tensor(image, dtype=self.dtype)

        if self.a_max == self.a_min:
            return image - self.a_min if self.b_min is None else image - self.a_min + self.b_min

        # Normalize to [0, 1]
        image = (image - self.a_min) / (self.a_max - self.a_min)

        # Scale to [b_min, b_max] if provided
        if self.b_min is not None and self.b_max is not None:
            image = image * (self.b_max - self.b_min) + self.b_min

        # Clip the values if required
        if self.clip and self.b_min is not None and self.b_max is not None:
            image = tf.clip_by_value(image, self.b_min, self.b_max)

        return tf.cast(image, dtype=self.dtype)

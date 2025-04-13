from medicai.utils.general import hide_warnings

hide_warnings()

from typing import Dict, Optional, Sequence, Union

import tensorflow as tf

from .tensor_bundle import TensorBundle


class ScaleIntensityRange:
    """Scales the intensity values of tensors to a specified range.

    This transform linearly scales the intensity values of the tensors specified
    by `keys` from an original range `[a_min, a_max]` to a target range
    `[b_min, b_max]`. Optionally, the output values can be clipped to the
    target range.
    """

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
        """Initializes the ScaleIntensityRange transform.

        Args:
            keys (Sequence[str]): Keys of the tensors to scale.
            a_min (float): The lower bound of the original intensity range.
            a_max (float): The upper bound of the original intensity range.
            b_min (Optional[float]): The lower bound of the target intensity range.
                If None, the output is normalized to [0, 1] (if `b_max` is also None)
                or shifted by `-a_min` (if `b_max` is None). Default is None.
            b_max (Optional[float]): The upper bound of the target intensity range.
                If None, the output is normalized to [0, 1] (if `b_min` is also None)
                or shifted by `-a_min` (if `b_min` is None). Default is None.
            clip (bool): If True, the output values will be clipped to the range `[b_min, b_max]`.
                This is only effective if both `b_min` and `b_max` are not None.
                Default is False.
            dtype (tf.DType): The data type to cast the scaled tensor to. Default is tf.float32.
        """
        self.keys = keys
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.dtype = dtype

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:
        """Apply the intensity scaling to the specified tensors in the input TensorBundle.

        Args:
            inputs (TensorBundle): A dictionary containing tensors and metadata. The tensors
                specified by `self.keys` will have their intensity range scaled.

        Returns:
            TensorBundle: A dictionary with the intensity-scaled tensors and the original metadata.
        """

        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        for key in self.keys:
            if key in inputs.data:
                inputs.data[key] = self.scale_intensity_range(inputs.data[key])
        return inputs

    def scale_intensity_range(self, image: tf.Tensor) -> tf.Tensor:
        """Scales the intensity range of a single tensor.

        Args:
            image (tf.Tensor): The input tensor to scale.

        Returns:
            tf.Tensor: The intensity-scaled tensor.
        """
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

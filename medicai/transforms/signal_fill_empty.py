from typing import Dict, Union

import tensorflow as tf

from .tensor_bundle import TensorBundle


class SignalFillEmpty:
    """Fill NaN, positive infinity, and negative infinity values in selected tensors.

    This transform iterates through the tensors associated with the provided keys
    and replaces invalid numeric values with a user-defined replacement. Keys
    that are not present in the input are skipped. Internally, the replacement
    helper converts tensors through ``float32`` before applying the fill.

    Args:
        keys (Sequence[str]): Keys of the tensors to process.
        replacement (float): Value used to replace NaN, positive infinity, and
            negative infinity entries. Default is ``0.0``.

    Example:
        Replace invalid numeric values in an image tensor::

            import tensorflow as tf
            from medicai.transforms import SignalFillEmpty

            filler = SignalFillEmpty(
                keys=["image"],
                replacement=0.0,
            )

            image = tf.constant([[[[float("nan")], [float("inf")]]]], dtype=tf.float32)
            result = filler({"image": image})
            filled_image = result["image"]

    Returns:
        ``TensorBundle``: The transformed output. We can retrieve the filled tensors
        using the same keys as the input.
    """

    def __init__(self, keys, replacement: float = 0.0):
        self.keys = keys
        self.replacement = replacement

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:
        """Apply the signal filling operation to the specified tensors.

        Args:
            inputs (TensorBundle): A sample dictionary or ``TensorBundle`` containing
                the tensors to process.

        Returns:
            TensorBundle: The transformed output. We can retrieve the filled tensors
            using the same keys as the input.
        """

        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        for key in self.keys:
            if key in inputs.data:
                replacement = tf.cast(self.replacement, dtype=inputs.data[key].dtype)
                inputs.data[key] = self.nan_to_num(inputs.data[key], nan=replacement)
        return inputs

    def nan_to_num(self, tensor, nan=0.0, posinf=None, neginf=None):
        """Replaces NaN, positive infinity, and negative infinity values in a tensor.

        Args:
            tensor (tf.Tensor): The input tensor.
            nan (float): The value to replace NaN with. Default is 0.0.
            posinf (float, optional): The value to replace positive infinity with.
                If None, it defaults to the maximum float32 value.
            neginf (float, optional): The value to replace negative infinity with.
                If None, it defaults to the minimum float32 value.

        Returns:
            tf.Tensor: A tensor with NaN, positive infinity, and negative infinity values
            replaced.
        """
        # Convert input to a TensorFlow tensor with float32 dtype
        tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)

        # Use default values if posinf or neginf are not provided
        posinf = tf.float32.max if posinf is None else posinf
        neginf = -tf.float32.max if neginf is None else neginf

        # Replace NaN values with the specified 'nan' value (default 0.0)
        tensor = tf.where(tf.math.is_nan(tensor), tf.fill(tf.shape(tensor), nan), tensor)

        # Replace positive infinity with 'posinf' (default: max float32)
        tensor = tf.where(
            tf.math.is_inf(tensor) & (tensor > 0), tf.fill(tf.shape(tensor), posinf), tensor
        )

        # Replace negative infinity with 'neginf' (default: min float32)
        tensor = tf.where(
            tf.math.is_inf(tensor) & (tensor < 0), tf.fill(tf.shape(tensor), neginf), tensor
        )

        return tensor

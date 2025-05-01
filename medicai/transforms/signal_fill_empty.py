from typing import Dict, Union

import tensorflow as tf

from .tensor_bundle import TensorBundle


class SignalFillEmpty:
    """Fills NaN, positive infinity, and negative infinity values in specified tensors with a
    given replacement.

    This transform iterates through the tensors associated with the provided keys in the
    input TensorBundle. For each such tensor, it replaces any occurrences of NaN (Not a Number),
    positive infinity, and negative infinity with a user-defined replacement value.
    """

    def __init__(self, keys, replacement: float = 0.0):
        """Initializes the SignalFillEmpty transform.

        Args:
            keys (Sequence[str]): A sequence of keys identifying the tensors within the
                TensorBundle to process.
            replacement (float): The value to use for replacing NaN, positive infinity,
                and negative infinity values. Default is 0.0.
        """
        self.keys = keys
        self.replacement = replacement

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:
        """Applies the signal filling operation to the specified tensors in the input
        TensorBundle.

        Args:
            inputs (TensorBundle): A dictionary-like object containing tensors and metadata.
                The tensors associated with the keys specified during initialization will
                be processed.

        Returns:
            TensorBundle: A new TensorBundle with the NaN, positive infinity, and negative
                infinity values in the specified tensors replaced by the `self.replacement` value.
                The metadata of the input TensorBundle is preserved.
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

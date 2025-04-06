import tensorflow as tf

from .tensor_bundle import TensorBundle


class SignalFillEmpty:
    def __init__(self, keys, replacement: float = 0.0):
        self.keys = keys
        self.replacement = replacement

    def __call__(self, inputs: TensorBundle) -> TensorBundle:
        for key in self.keys:
            if key in inputs.data:
                replacement = tf.cast(self.replacement, dtype=inputs.data[key].dtype)
                inputs.data[key] = self.nan_to_num(inputs.data[key], nan=replacement)
        return inputs

    def nan_to_num(self, tensor, nan=0.0, posinf=None, neginf=None):
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

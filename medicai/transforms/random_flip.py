from typing import Dict, Sequence, Union

from medicai.utils.general import hide_warnings

hide_warnings()
import tensorflow as tf

from .tensor_bundle import TensorBundle


class RandFlip:
    """Randomly flips tensors along specified spatial axes with a given probability.

    This transformation randomly decides whether to flip the input tensors based
    on the provided probability. If the decision is to flip, it reverses the
    tensor elements along the specified spatial axes.
    """

    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.1,
        spatial_axis: Union[int, Sequence[int], None] = None,
    ):
        """
        Initializes the RandFlip transform.

        Args:
            keys (Sequence[str]): Keys of the tensors to potentially flip.
            prob (float): Probability of applying the flip operation (between 0 and 1).
                Default is 0.1.
            spatial_axis (Optional[Union[int, Sequence[int]]]): The spatial axis or axes
                along which to flip the tensor. If None, no flipping is performed
                by this transform (even if the random probability condition is met).
                For a 4D tensor (e.g., depth, height, width, channels), the spatial
                axes are typically 0, 1, and 2. Default is None.
        """
        self.keys = keys
        self.prob = prob
        self.spatial_axis = spatial_axis

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:
        """
        Apply the random flipping transformation to the input TensorBundle.

        Args:
            inputs (TensorBundle): A dictionary containing tensors and metadata.

        Returns:
            TensorBundle: A dictionary with potentially flipped tensors and the original metadata.
        """

        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        should_flip = tf.random.uniform([]) < self.prob

        for key in self.keys:
            if key in inputs.data:
                if should_flip:
                    inputs.data[key] = tf.reverse(inputs.data[key], axis=self.spatial_axis)
        return inputs

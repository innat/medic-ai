from typing import Dict, Sequence, Union

import tensorflow as tf

from .tensor_bundle import TensorBundle


class RandFlip:
    """Randomly flip selected tensors along specified axes.

    This transform samples one random decision per call. If the sample is below
    ``prob``, it reverses each selected tensor along ``spatial_axis``. Keys that
    are not present in the input are skipped.

    Args:
        keys (Sequence[str]): Keys of the tensors to potentially flip.
        prob (float): Probability of applying the flip operation. Default is
            ``0.1``.
        spatial_axis (Optional[Union[int, Sequence[int]]]): Spatial axis or
            axes passed to ``tf.reverse``. If ``None``, no flipping is applied
            even when the random condition is met.

    Example:
        Randomly flip an image-label pair along the height axis::

            import tensorflow as tf
            from medicai.transforms import RandFlip

            flipper = RandFlip(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=1,
            )

            image = tf.random.normal((64, 64, 64, 1))
            label = tf.random.uniform((64, 64, 64, 1), maxval=2, dtype=tf.int32)

            result = flipper({"image": image, "label": label})
            flipped_image = result["image"]
            flipped_label = result["label"]

    Returns:
        ``TensorBundle``: The transformed output. We can retrieve the flipped
        tensors using the same keys as the input.
    """

    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.1,
        spatial_axis: Union[int, Sequence[int], None] = None,
    ):
        self.keys = keys
        self.prob = prob
        self.spatial_axis = spatial_axis

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:
        """
        Apply the random flipping transformation to the input TensorBundle.

        Args:
            inputs (TensorBundle): A sample dictionary or ``TensorBundle`` containing
                the tensors to flip.

        Returns:
            TensorBundle: The transformed output. We can retrieve the flipped
            tensors using the same keys as the input.
        """

        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        should_flip = tf.random.uniform([]) < self.prob

        for key in self.keys:
            if key in inputs.data:
                if should_flip:
                    inputs.data[key] = tf.reverse(inputs.data[key], axis=self.spatial_axis)
        return inputs

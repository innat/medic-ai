import tensorflow as tf
from typing import Union, Sequence, Dict

class RandFlip:
    def __init__(self, prob: float = 0.1, spatial_axis: Union[int, Sequence[int], None] = None):
        """
        TensorFlow implementation of RandFlipd for 3D inputs (d, h, w, 1).

        Args:
            prob (float): Probability of flipping. Defaults to 0.1.
            spatial_axis (Union[int, Sequence[int], None]): Spatial axes along which to flip.
                Defaults to None (flip along all axes).
        """
        self.prob = prob
        self.spatial_axis = spatial_axis

    def __call__(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Apply random flipping to the input dictionary.

        Args:
            inputs (Dict[str, tf.Tensor]): Input dictionary containing 'image' and 'label'.

        Returns:
            Dict[str, tf.Tensor]: Transformed dictionary.
        """
        image = inputs['image']
        label = inputs['label']

        if tf.random.uniform([]) < self.prob:
            image = tf.reverse(image, axis=spatial_axis)
            label = tf.reverse(label, axis=spatial_axis)

        return {'image': image, 'label': label}
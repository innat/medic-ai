import tensorflow as tf
from typing import Union, Sequence, Dict

class RandFlipd:
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
            image = self._rand_flip(image)
            label = self._rand_flip(label)

        return {'image': image, 'label': label}

    def _rand_flip(self, image: tf.Tensor) -> tf.Tensor:
        """
        Randomly flip a 3D image tensor along specified axes.

        Args:
            image (tf.Tensor): Input 3D image tensor with shape (d, h, w, 1).

        Returns:
            tf.Tensor: Flipped image tensor.
        """
        axes = self._get_flip_axes()
        if axes:
            image = tf.reverse(image, axis=axes)
        return image

    def _get_flip_axes(self) -> list[int]:
        """
        Get the axes along which to flip the image.

        Returns:
            list[int]: List of axes to flip.
        """
        if self.spatial_axis is None:
            return [0, 1, 2]  # Flip along all spatial axes (depth, height, width)
        elif isinstance(self.spatial_axis, int):
            return [self.spatial_axis]  # Flip along a single axis
        else:
            return list(self.spatial_axis)  # Flip along multiple axes
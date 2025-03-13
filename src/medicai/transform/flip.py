
import tensorflow as tf

def RandFlipd(
    prob: float = 0.1,
    spatial_axis: Union[int, Sequence[int], None] = None,
):
    """
    TensorFlow implementation of RandFlipd for 3D inputs (d, h, w, 1).

    Args:
        prob (float): Probability of flipping. Defaults to 0.1.
        spatial_axis (Union[int, Sequence[int], None]): Spatial axes along which to flip.
            Defaults to None (flip along all axes).
    """

    def wrapper(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Apply random flipping to the input dictionary.

        Args:
            inputs (Dict[str, tf.Tensor]): Input dictionary containing 'image' and 'label'.

        Returns:
            Dict[str, tf.Tensor]: Transformed dictionary.
        """
        image = inputs['image']
        label = inputs['label']

        if tf.random.uniform([]) < prob:
            image = _rand_flip(image)
            label = _rand_flip(label)

        return {'image': image, 'label': label}

    def _rand_flip(image: tf.Tensor) -> tf.Tensor:
        """
        Randomly flip a 3D image tensor along specified axes.

        Args:
            image (tf.Tensor): Input 3D image tensor with shape (d, h, w, 1).

        Returns:
            tf.Tensor: Flipped image tensor.
        """
        axes = _get_flip_axes(image)
        if axes:
            image = tf.reverse(image, axis=axes)
        return image

    def _get_flip_axes(image: tf.Tensor) -> list[int]:
        """
        Get the axes along which to flip the image.

        Args:
            image (tf.Tensor): Input 3D image tensor with shape (d, h, w, 1).

        Returns:
            list[int]: List of axes to flip.
        """
        if spatial_axis is None:
            # Flip along all spatial axes (depth, height, width)
            return [0, 1, 2]
        elif isinstance(spatial_axis, int):
            # Flip along a single axis
            return [spatial_axis]
        else:
            # Flip along multiple axes
            return list(spatial_axis)

    return wrapper
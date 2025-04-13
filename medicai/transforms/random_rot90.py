from medicai.utils.general import hide_warnings

hide_warnings()
from typing import Dict, Union

import tensorflow as tf

from .tensor_bundle import TensorBundle


class RandRotate90:
    def __init__(self, keys, prob=0.1, max_k=3, spatial_axes=(1, 2)):
        """
        Args:
            prob: Probability of applying the 90-degree rotation.
            max_k: Maximum number of 90-degree rotations (1 to max_k).
            spatial_axes: Axes along which to rotate (default: (1,2) for height-width).
        """
        self.keys = keys
        self.prob = prob
        self.max_k = max_k
        self.spatial_axes = spatial_axes

    def rot90(self, array, k=1, axes=(0, 1)):
        """Rotate an array by 90 degrees in the specified plane.

        Args:
            array: Input tensor
            k: Number of 90-degree rotations (default=1)
            axes: Tuple of two axes that define the plane of rotation.
            Defaults to (0, 1).

        Returns:
            Rotated tensor with correct shape transformation
        """
        array = tf.convert_to_tensor(array, dtype=array.dtype)

        if array.shape.rank < 2:
            raise ValueError(
                f"Input array must have at least 2 dimensions. "
                f"Received: array.ndim={array.shape.rank}"
            )

        if len(axes) != 2 or axes[0] == axes[1]:
            raise ValueError(
                f"Invalid axes: {axes}. Axes must be a tuple of " "two different dimensions."
            )

        k = k % 4
        if k == 0:
            return array

        axes = tuple(axis if axis >= 0 else array.shape.rank + axis for axis in axes)

        perm = [i for i in range(array.shape.rank) if i not in axes]
        perm.extend(axes)
        array = tf.transpose(array, perm)

        shape = tf.shape(array)
        non_rot_shape = shape[:-2]
        h, w = shape[-2], shape[-1]

        array = tf.reshape(array, tf.concat([[-1], [h, w]], axis=0))
        array = tf.reverse(array, axis=[2])
        array = tf.transpose(array, [0, 2, 1])

        if k % 2 == 1:
            final_h, final_w = w, h
        else:
            final_h, final_w = h, w

        if k > 1:
            array = tf.reshape(array, tf.concat([[-1], [final_h, final_w]], axis=0))
            for _ in range(k - 1):
                array = tf.reverse(array, axis=[2])
                array = tf.transpose(array, [0, 2, 1])

        final_shape = tf.concat([non_rot_shape, [final_h, final_w]], axis=0)
        array = tf.reshape(array, final_shape)

        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        array = tf.transpose(array, inv_perm)

        return array

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:

        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        rand_val = tf.random.uniform(())

        def apply_rotation():
            rotated_data = inputs.data.copy()
            k = tf.random.uniform((), minval=1, maxval=self.max_k + 1, dtype=tf.int32)
            for key in self.keys:
                if key in rotated_data:
                    rotated_data[key] = self.rot90(rotated_data[key], k, axes=self.spatial_axes)
            return rotated_data

        def no_rotation():
            return inputs.data.copy()

        rotated_data = tf.cond(rand_val <= self.prob, apply_rotation, no_rotation)
        return TensorBundle(rotated_data, inputs.meta)

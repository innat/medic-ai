from typing import Dict, Sequence, Union

import tensorflow as tf

from .tensor_bundle import TensorBundle


class RandCutOut:
    """
    Random 3D CutOut augmentation.

    Randomly masks one or more cuboid regions in a 3D volume and applies
    the same mask to image–label pairs to preserve spatial alignment.
    Supports multi-channel inputs and anisotropic mask sizes.
    """

    def __init__(
        self,
        keys: Sequence[str],
        mask_size: Sequence[int],
        num_cuts: int,
        prob: float = 0.5,
        fill_value: float = 0.0,
        label_fill_value: int = 0,
    ):
        if len(keys) != 2:
            raise ValueError("keys must be a sequence of two strings: [image_key, label_key]")
        if len(mask_size) != 3:
            raise ValueError("mask_size must be a sequence of three integers for 3D.")

        self.keys = keys
        self.mask_size = mask_size
        self.num_cuts = num_cuts
        self.prob = prob
        self.fill_value = fill_value
        self.label_fill_value = label_fill_value

    def __call__(self, inputs: Union["TensorBundle", Dict[str, tf.Tensor]]):
        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        if tf.random.uniform([]) >= self.prob:
            return inputs

        image_key, label_key = self.keys
        image = inputs.data[image_key]
        label = inputs.data[label_key]

        mask = self._generate_mask(image)
        mask_bool = tf.cast(mask, tf.bool)  # convert mask to boolean

        # apply mask safely with tf.where
        inputs.data[image_key] = tf.where(mask_bool, image, self.fill_value)
        inputs.data[label_key] = tf.where(mask_bool, label, self.label_fill_value)

        return inputs

    def _generate_mask(self, volume):
        """
        Generates a shared (D, H, W, 1) binary mask for 3D CutOut.
        """
        shape = tf.shape(volume)
        D, H, W = shape[0], shape[1], shape[2]
        md, mh, mw = self.mask_size

        mask = tf.ones((D, H, W), dtype=volume.dtype)

        for _ in range(self.num_cuts):
            cz = tf.random.uniform([], 0, D, dtype=tf.int32)
            cy = tf.random.uniform([], 0, H, dtype=tf.int32)
            cx = tf.random.uniform([], 0, W, dtype=tf.int32)

            z1 = tf.clip_by_value(cz - md // 2, 0, D)
            z2 = tf.clip_by_value(cz + md // 2, 0, D)
            y1 = tf.clip_by_value(cy - mh // 2, 0, H)
            y2 = tf.clip_by_value(cy + mh // 2, 0, H)
            x1 = tf.clip_by_value(cx - mw // 2, 0, W)
            x2 = tf.clip_by_value(cx + mw // 2, 0, W)

            indices = tf.reshape(
                tf.stack(
                    tf.meshgrid(
                        tf.range(z1, z2),
                        tf.range(y1, y2),
                        tf.range(x1, x2),
                        indexing="ij",
                    ),
                    axis=-1,
                ),
                [-1, 3],
            )

            updates = tf.zeros((tf.shape(indices)[0],), dtype=volume.dtype)

            mask = tf.tensor_scatter_nd_update(mask, indices, updates)

        return mask[..., None]  # (D, H, W, 1)

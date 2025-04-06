from medicai.utils.general import hide_warnings

hide_warnings()

from typing import Sequence

import tensorflow as tf

from .tensor_bundle import TensorBundle


class NormalizeIntensity:
    def __init__(
        self,
        keys: Sequence[str],
        subtrahend=None,
        divisor=None,
        nonzero=False,
        channel_wise=False,
        dtype=tf.float32,
        allow_missing_keys=False,
    ):
        self.keys = keys
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.dtype = dtype
        self.allow_missing_keys = allow_missing_keys

    def _normalize_channel_wise(self, image: tf.Tensor) -> tf.Tensor:
        mask = tf.not_equal(image, 0.0) if self.nonzero else tf.ones_like(image, dtype=tf.bool)
        num_dims = tf.rank(image)
        channel_axis = num_dims - 1

        def normalize_single_channel(channel_and_mask):
            channel, channel_mask_for_mean_std = channel_and_mask
            channel_masked = tf.boolean_mask(channel, channel_mask_for_mean_std)
            mean = tf.reduce_mean(channel_masked)
            std = tf.math.reduce_std(channel_masked)
            sub = self.subtrahend if self.subtrahend is not None else mean
            div = self.divisor if self.divisor is not None else std
            div = tf.where(tf.equal(div, 0.0), tf.ones_like(div), div)
            return (channel - sub) / div

        # Move the channel axis to the first dimension
        channel_axis_tensor = tf.expand_dims(channel_axis, axis=0)
        other_axes = tf.range(channel_axis)
        permutation = tf.concat([channel_axis_tensor, other_axes], axis=0)
        transposed_image = tf.transpose(image, perm=permutation)
        transposed_mask = tf.transpose(mask, perm=permutation)

        normalized_transposed = tf.map_fn(
            normalize_single_channel, (transposed_image, transposed_mask), dtype=image.dtype
        )

        # Move the channel axis back to the original position (last)
        # Create the inverse permutation using tf.range and tf.concat
        first_part_inverse = tf.range(1, num_dims)
        last_part_inverse = tf.constant([0], dtype=tf.int32)
        inverse_permutation = tf.concat([first_part_inverse, last_part_inverse], axis=0)
        normalized_image = tf.transpose(normalized_transposed, perm=inverse_permutation)
        return normalized_image

    def _normalize_global(self, image: tf.Tensor) -> tf.Tensor:
        mask = tf.not_equal(image, 0.0) if self.nonzero else tf.ones_like(image, dtype=tf.bool)
        image_masked = tf.boolean_mask(image, mask)
        mean = tf.reduce_mean(image_masked)
        std = tf.math.reduce_std(image_masked)
        sub = self.subtrahend if self.subtrahend is not None else mean
        div = self.divisor if self.divisor is not None else std
        div = tf.where(tf.equal(div, 0.0), tf.ones_like(div), div)
        return (image - sub) / div

    def __call__(self, inputs: TensorBundle) -> TensorBundle:
        for key in self.keys:
            if key not in inputs.data:
                if self.allow_missing_keys:
                    continue
                else:
                    raise KeyError(f"Key '{key}' not found in input data.")

            image = tf.cast(inputs.data[key], dtype=self.dtype or inputs.data[key].dtype)

            if self.channel_wise:
                norm = self._normalize_channel_wise(image)
            else:
                norm = self._normalize_global(image)

            inputs.data[key] = tf.cast(norm, self.dtype or image.dtype)

        return inputs

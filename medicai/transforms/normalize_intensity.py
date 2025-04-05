from medicai.utils.general import hide_warnings

hide_warnings()

import tensorflow as tf
from .tensor_bundle import TensorBundle

class NormalizeIntensity:
    def __init__(self, keys, subtrahend=None, divisor=None,
                 nonzero=False, channel_wise=False, dtype=tf.float32, allow_missing_keys=False):
        self.keys = keys
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.dtype = dtype
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, inputs: TensorBundle) -> TensorBundle:
        for key in self.keys:
            if key not in inputs.data:
                if self.allow_missing_keys:
                    continue
                else:
                    raise KeyError(f"Key '{key}' not found in input data.")

            image = tf.cast(inputs.data[key], dtype=self.dtype or inputs.data[key].dtype)
            mask = tf.not_equal(image, 0.0) if self.nonzero else tf.ones_like(image, dtype=tf.bool)

            if self.channel_wise:
                # Compute per-channel mean and std
                num_dims = tf.rank(image)
                channel_axis = num_dims - 1  # channel-last

                def compute_stats(x):
                    x_masked = tf.boolean_mask(x, mask[..., i])
                    mean = tf.reduce_mean(x_masked)
                    std = tf.math.reduce_std(x_masked)
                    return mean, std

                means = []
                stds = []
                for i in range(tf.shape(image)[channel_axis]):
                    mean, std = compute_stats(image[..., i])
                    means.append(mean)
                    stds.append(std)

                means = tf.stack(means)
                stds = tf.stack(stds)

                # Use provided subtrahend/divisor or computed ones
                sub = self.subtrahend if self.subtrahend is not None else means
                div = self.divisor if self.divisor is not None else stds
                div = tf.where(tf.equal(div, 0.0), tf.ones_like(div), div)

                norm = (image - sub) / div
            else:
                image_masked = tf.boolean_mask(image, mask)
                mean = tf.reduce_mean(image_masked)
                std = tf.math.reduce_std(image_masked)

                sub = self.subtrahend if self.subtrahend is not None else mean
                div = self.divisor if self.divisor is not None else std
                div = div if div != 0 else 1.0

                norm = (image - sub) / div

            inputs.data[key] = tf.cast(norm, self.dtype or image.dtype)

        return inputs

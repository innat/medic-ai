from typing import Sequence

import tensorflow as tf

from ..base import KeyedTransform
from ..tensor_bundle import TensorBundle


class NormalizeIntensity(KeyedTransform):
    """Normalize tensor intensities using global or channel-wise statistics."""

    def __init__(
        self,
        keys: Sequence[str],
        subtrahend=None,
        divisor=None,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype=tf.float32,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.dtype = dtype

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present_keys = self.apply_to_present_keys(bundle, lambda tensor, _: self.normalize_tensor(tensor))
        bundle.push_transform(
            self.build_trace_entry(
                params={
                    "keys": list(present_keys),
                    "nonzero": self.nonzero,
                    "channel_wise": self.channel_wise,
                },
                applied=True,
                random=False,
                invertible=False,
            )
        )
        return bundle

    def normalize_tensor(self, tensor: tf.Tensor) -> tf.Tensor:
        """Normalize one tensor with the configured statistics policy."""
        tensor = tf.cast(tensor, dtype=self.dtype or tensor.dtype)
        if self.channel_wise:
            normalized = self._normalize_channel_wise(tensor)
        else:
            normalized = self._normalize_global(tensor)
        return tf.cast(normalized, self.dtype or tensor.dtype)

    def _normalize_channel_wise(self, tensor: tf.Tensor) -> tf.Tensor:
        mask = tf.not_equal(tensor, 0.0) if self.nonzero else tf.ones_like(tensor, dtype=tf.bool)
        num_dims = tf.rank(tensor)
        channel_axis = num_dims - 1

        def normalize_single_channel(channel_and_mask):
            channel, channel_mask = channel_and_mask
            channel_masked = tf.boolean_mask(channel, channel_mask)
            mean = tf.reduce_mean(channel_masked)
            std = tf.math.reduce_std(channel_masked)
            sub = self.subtrahend if self.subtrahend is not None else mean
            div = self.divisor if self.divisor is not None else std
            div = tf.where(tf.equal(div, 0.0), tf.ones_like(div), div)
            return (channel - sub) / div

        permutation = tf.concat([tf.expand_dims(channel_axis, axis=0), tf.range(channel_axis)], axis=0)
        transposed_tensor = tf.transpose(tensor, perm=permutation)
        transposed_mask = tf.transpose(mask, perm=permutation)
        normalized_transposed = tf.map_fn(
            normalize_single_channel,
            (transposed_tensor, transposed_mask),
            dtype=tensor.dtype,
        )
        inverse_permutation = tf.concat([tf.range(1, num_dims), tf.constant([0], dtype=tf.int32)], axis=0)
        return tf.transpose(normalized_transposed, perm=inverse_permutation)

    def _normalize_global(self, tensor: tf.Tensor) -> tf.Tensor:
        mask = tf.not_equal(tensor, 0.0) if self.nonzero else tf.ones_like(tensor, dtype=tf.bool)
        num_valid = tf.reduce_sum(tf.cast(mask, tf.int32))

        def normalize():
            vals = tf.boolean_mask(tensor, mask)
            mean = tf.reduce_mean(vals)
            std = tf.math.reduce_std(vals)
            std = tf.where(std == 0.0, 1.0, std)
            sub = mean if self.subtrahend is None else tf.cast(self.subtrahend, tensor.dtype)
            div = std if self.divisor is None else tf.cast(self.divisor, tensor.dtype)
            div = tf.where(div == 0.0, 1.0, div)
            if self.nonzero:
                return tf.where(mask, (tensor - sub) / div, tensor)
            return (tensor - sub) / div

        return tf.cond(num_valid > 0, normalize, lambda: tensor)

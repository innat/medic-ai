from typing import Sequence

import tensorflow as tf

from ..base import KeyedTransform
from ..tensor_bundle import TensorBundle


class NormalizeIntensity(KeyedTransform):
    """Normalize selected tensors using global or channel-wise intensity statistics.

    ``NormalizeIntensity`` standardizes tensor values by subtracting a mean-like
    term and dividing by a standard-deviation-like term. The statistics can be
    computed over the whole tensor or independently per channel, and can be
    restricted to nonzero voxels or pixels when background values should be
    excluded from normalization.

    This transform operates on channel-last tensors and is commonly used for
    medical images shaped like ``(H, W, C)`` or ``(D, H, W, C)``. Label tensors
    are usually not appropriate inputs unless a workflow explicitly requires
    intensity-style normalization on them.

    Args:
        keys: Keys of the tensors to normalize.
        offset: Optional fixed value to subtract. If ``None``, the mean of
            the selected values is used.
        scale: Optional fixed value to divide by. If ``None``, the standard
            deviation of the selected values is used.
        nonzero: If ``True``, statistics are computed only over nonzero values.
            For global normalization, zero-valued locations are left unchanged
            after normalization. For channel-wise normalization, nonzero values
            determine the statistics used for each channel.
        channel_wise: If ``True``, normalize each channel independently using
            channel-specific statistics. If ``False``, normalize using one set
            of statistics over the full tensor.
        dtype: Output dtype used for computation and returned tensors.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Normalize a 2D image using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import NormalizeIntensity

            transform = NormalizeIntensity(
                keys=["image"],
                nonzero=True,
                channel_wise=False,
            )

            image = tf.random.normal((64, 64, 1))
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Normalize a 3D image volume using a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import NormalizeIntensity, TensorBundle

            transform = NormalizeIntensity(
                keys=["image"],
                nonzero=True,
                channel_wise=False,
            )

            image = tf.random.normal((32, 64, 64, 1))
            bundle = TensorBundle({"image": image})
            result = transform(bundle)
            output = result["image"]
            print(output.shape)

    Returns:
        ``TensorBundle``: The input bundle with selected tensors normalized in
        place and a non-invertible trace entry appended.

    Raises:
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
    """

    def __init__(
        self,
        keys: Sequence[str],
        offset=None,
        scale=None,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype=tf.float32,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.offset = offset
        self.scale = scale
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.dtype = dtype

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present_keys = self.apply_to_present_keys(
            bundle, lambda tensor, _: self.normalize_tensor(tensor)
        )
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
            has_valid = tf.size(channel_masked) > 0

            def normalize_nonempty():
                mean = tf.reduce_mean(channel_masked)
                std = tf.math.reduce_std(channel_masked)
                sub = self.offset if self.offset is not None else mean
                div = self.scale if self.scale is not None else std
                sub = tf.cast(sub, channel.dtype)
                div = tf.cast(div, channel.dtype)
                div = tf.where(tf.equal(div, 0.0), tf.ones_like(div), div)
                normalized = (channel - sub) / div
                if self.nonzero:
                    return tf.where(channel_mask, normalized, channel)
                return normalized

            return tf.cond(has_valid, normalize_nonempty, lambda: channel)

        permutation = tf.concat(
            [tf.expand_dims(channel_axis, axis=0), tf.range(channel_axis)], axis=0
        )
        transposed_tensor = tf.transpose(tensor, perm=permutation)
        transposed_mask = tf.transpose(mask, perm=permutation)
        normalized_transposed = tf.map_fn(
            normalize_single_channel,
            (transposed_tensor, transposed_mask),
            dtype=tensor.dtype,
        )
        inverse_permutation = tf.concat(
            [tf.range(1, num_dims), tf.constant([0], dtype=tf.int32)], axis=0
        )
        return tf.transpose(normalized_transposed, perm=inverse_permutation)

    def _normalize_global(self, tensor: tf.Tensor) -> tf.Tensor:
        mask = tf.not_equal(tensor, 0.0) if self.nonzero else tf.ones_like(tensor, dtype=tf.bool)
        num_valid = tf.reduce_sum(tf.cast(mask, tf.int32))

        def normalize():
            vals = tf.boolean_mask(tensor, mask)
            mean = tf.reduce_mean(vals)
            std = tf.math.reduce_std(vals)
            std = tf.where(std == 0.0, 1.0, std)
            sub = mean if self.offset is None else tf.cast(self.offset, tensor.dtype)
            div = std if self.scale is None else tf.cast(self.scale, tensor.dtype)
            div = tf.where(div == 0.0, 1.0, div)
            if self.nonzero:
                return tf.where(mask, (tensor - sub) / div, tensor)
            return (tensor - sub) / div

        return tf.cond(num_valid > 0, normalize, lambda: tensor)

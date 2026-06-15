from typing import Dict, Sequence, Union

import tensorflow as tf

from .tensor_bundle import TensorBundle


class NormalizeIntensity:
    """
    Normalize tensor intensities using global or channel-wise statistics.

    This transform subtracts ``subtrahend`` and divides by ``divisor`` for the
    selected tensors. When either value is ``None``, the transform computes it
    from the input tensor, either globally or independently for each channel.
    We can also choose to compute the statistics using only non-zero values.

    Args:
        keys (Sequence[str]): Keys of the tensors to normalize.
        subtrahend (Optional[Union[float, tf.Tensor]]): Value to subtract from the tensor(s).
            If None and `channel_wise` is False, the global mean of the non-zero
            (if `nonzero` is True) or all values is used. If None and `channel_wise`
            is True, the mean of each channel is used. Default is None.
        divisor (Optional[Union[float, tf.Tensor]]): Value to divide the tensor(s) by.
            If None and `channel_wise` is False, the global standard deviation of the
            non-zero (if `nonzero` is True) or all values is used. If None and
            `channel_wise` is True, the standard deviation of each channel is used.
            If the standard deviation is zero, it is replaced with one to avoid division by zero.
            Default is None.
        nonzero (bool): If True, normalization statistics (mean and std) are computed
            only over non-zero values. Default is False.
        channel_wise (bool): If True, normalization is performed independently for each channel.
            The subtrahend and divisor will be computed per channel if they are None.
            Default is False.
        dtype (Optional[tf.DType]): The data type to cast the normalized tensor(s) to.
            If None, the original data type is preserved. Default is tf.float32.
        allow_missing_keys (bool): If True, processing will continue even if some of the specified
            keys are not found in the input data. The missing keys will be ignored. Default is False.

    Example:
        Normalize an image tensor using automatically computed statistics::

            import tensorflow as tf
            from medicai.transforms import NormalizeIntensity

            normalizer = NormalizeIntensity(
                keys=["image"],
                nonzero=True,
                channel_wise=False,
            )

            image = tf.random.normal((64, 64, 64, 1))
            result = normalizer({"image": image})
            normalized_image = result["image"]

    Returns:
        TensorBundle: The transformed output. We can retrieve the normalized
        tensors using the same keys as the input.

    Raises:
        KeyError: If a requested key is missing and ``allow_missing_keys=False``.
    """

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
        """
        Normalize the input tensor channel-wise.

        Args:
            image (tf.Tensor): The input tensor to normalize (shape [..., channels]).

        Returns:
            tf.Tensor: The channel-wise normalized tensor.
        """
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
        num_valid = tf.reduce_sum(tf.cast(mask, tf.int32))

        def normalize():
            vals = tf.boolean_mask(image, mask)
            mean = tf.reduce_mean(vals)
            std = tf.math.reduce_std(vals)
            std = tf.where(std == 0.0, 1.0, std)

            sub = mean if self.subtrahend is None else tf.cast(self.subtrahend, image.dtype)
            div = std if self.divisor is None else tf.cast(self.divisor, image.dtype)
            div = tf.where(div == 0.0, 1.0, div)

            if self.nonzero:
                return tf.where(mask, (image - sub) / div, image)
            return (image - sub) / div

        def identity():
            return image

        return tf.cond(num_valid > 0, normalize, identity)

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:
        """
        Apply the NormalizeIntensity transform to the input TensorBundle.

        Args:
            inputs (TensorBundle): A sample dictionary or ``TensorBundle`` containing
                the tensors to normalize.

        Returns:
            TensorBundle: The transformed output. We can retrieve the normalized
            tensors using the same keys as the input.

        Raises:
            KeyError: If a requested key is missing and ``allow_missing_keys=False``.
        """

        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

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

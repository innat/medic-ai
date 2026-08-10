from typing import Sequence

import tensorflow as tf

from ..base import KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import custom_tf_boolean_mask, validate_input_mode, validate_layout, validate_spatial_dims


class NormalizeIntensity(KeyedTransform):
    """Normalize selected tensors using global or channel-wise intensity statistics.

    ``NormalizeIntensity`` standardizes tensor values by subtracting a mean-like
    term and dividing by a standard-deviation-like term. The statistics can be
    computed over the whole tensor or independently per channel, and can be
    restricted to nonzero voxels or pixels when background values should be
    excluded from normalization.

    Depending on ``input_mode``, this transform supports:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

    Label tensors are usually not appropriate inputs unless a workflow
    explicitly requires intensity-style normalization on them.

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
        input_mode: Either ``"sample"`` for ``(H, W, C)`` / ``(D, H, W, C)``
            tensors, or ``"batch"`` for ``(B, H, W, C)`` / ``(B, D, H, W, C)``
            tensors.
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
        *,
        spatial_dims: int,
        input_mode: str = "sample",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.offset = offset
        self.scale = scale
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.dtype = dtype
        self.input_mode = validate_input_mode(input_mode, transform_name=type(self).__name__)
        self.spatial_dims = validate_spatial_dims(spatial_dims, transform_name=type(self).__name__)

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
                    "input_mode": self.input_mode,
                    "spatial_dims": self.spatial_dims,
                },
                applied=True,
                random=False,
                invertible=False,
            )
        )
        return bundle

    def normalize_tensor(self, tensor: tf.Tensor) -> tf.Tensor:
        """Normalize one tensor with the configured statistics policy."""
        self._validate_tensor_layout(tensor)
        tensor = tf.cast(tensor, dtype=self.dtype or tensor.dtype)
        if self.channel_wise:
            normalized = self._normalize_channel_wise(tensor)
        else:
            normalized = self._normalize_global(tensor)
        return tf.cast(normalized, self.dtype or tensor.dtype)

    def _normalize_channel_wise(self, tensor: tf.Tensor) -> tf.Tensor:
        mask = tf.not_equal(tensor, 0.0) if self.nonzero else tf.ones_like(tensor, dtype=tf.bool)
        reduce_axes = tuple(range(tensor.shape.rank - 1))
        mask_f = tf.cast(mask, tensor.dtype)
        valid_counts = tf.reduce_sum(mask_f, axis=reduce_axes)

        broadcast_shape = tf.concat(
            [tf.ones([tensor.shape.rank - 1], dtype=tf.int32), [tf.shape(tensor)[-1]]],
            axis=0,
        )
        valid_counts_b = tf.reshape(valid_counts, broadcast_shape)

        masked_tensor = tf.where(mask, tensor, tf.zeros_like(tensor))
        mean = tf.reduce_sum(masked_tensor, axis=reduce_axes) / tf.where(
            valid_counts > 0,
            valid_counts,
            tf.ones_like(valid_counts),
        )
        mean_b = tf.reshape(mean, broadcast_shape)

        sq_diff = tf.where(mask, tf.square(tensor - mean_b), tf.zeros_like(tensor))
        std = tf.sqrt(tf.reduce_sum(sq_diff, axis=reduce_axes) / tf.where(
            valid_counts > 0,
            valid_counts,
            tf.ones_like(valid_counts),
        ))
        std_b = tf.reshape(std, broadcast_shape)

        sub = mean_b if self.offset is None else tf.cast(self.offset, tensor.dtype)
        div = std_b if self.scale is None else tf.cast(self.scale, tensor.dtype)
        div = tf.where(tf.equal(div, 0.0), tf.ones_like(div), div)

        normalized = (tensor - sub) / div
        if self.nonzero:
            normalized = tf.where(mask, normalized, tensor)

        has_valid = valid_counts_b > 0
        return tf.where(has_valid, normalized, tensor)

    def _normalize_global(self, tensor: tf.Tensor) -> tf.Tensor:
        mask = tf.not_equal(tensor, 0.0) if self.nonzero else tf.ones_like(tensor, dtype=tf.bool)
        num_valid = tf.reduce_sum(tf.cast(mask, tf.int32))

        def normalize():
            vals = custom_tf_boolean_mask(tensor, mask, mode="extract")
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

    def _validate_tensor_layout(self, tensor: tf.Tensor) -> None:
        """Validate sample or batch channel-last layout for normalization."""
        validate_layout(
            tensor,
            input_mode=self.input_mode,
            allowed_spatial_ranks=(2, 3),
            spatial_dims=self.spatial_dims,
            transform_name=type(self).__name__,
        )

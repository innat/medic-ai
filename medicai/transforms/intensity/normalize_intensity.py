from typing import Any, Sequence

from keras import ops

from ..base import KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import (
    resolve_input_layout,
    validate_tensor_matches_layout,
)


class NormalizeIntensity(KeyedTransform):
    """Normalize selected tensors using global or channel-wise intensity statistics.

    ``NormalizeIntensity`` standardizes tensor values by subtracting a mean-like
    term and dividing by a standard-deviation-like term. The statistics can be
    computed over the whole tensor or independently per channel, and can be
    restricted to nonzero voxels or pixels when background values should be
    excluded from normalization.

    Depending on ``input_layout``, this transform supports:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

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
            one statistic per channel. If ``False``, normalize using one set
            of statistics over the full tensor. For ``BHWC`` and ``BDHWC``,
            channel-wise statistics are shared across the batch rather than
            computed independently for each sample. Use a sample layout when
            normalizing one image or volume at a time.
        dtype: Output dtype used for computation and returned tensors.
        input_layout: Channel-last tensor layout. Supported values are
            ``"HWC"``, ``"DHWC"``, ``"BHWC"``, and ``"BDHWC"``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:

        TensorFlow backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import NormalizeIntensity

            transform = NormalizeIntensity(
                keys=["image"],
                nonzero=True,
                channel_wise=False,
                input_layout="HWC",
            )

            image = tf.random.normal((64, 64, 1), seed=7)
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        JAX backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            import jax.numpy as jnp
            from medicai.transforms import NormalizeIntensity

            transform = NormalizeIntensity(
                keys=["image"],
                nonzero=True,
                channel_wise=True,
                input_layout="DHWC",
            )

            image = jax.random.normal(
                jax.random.PRNGKey(7), shape=(32, 64, 64, 1), dtype=jnp.float32
            )
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Torch backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import NormalizeIntensity

            transform = NormalizeIntensity(
                keys=["image"],
                nonzero=True,
                channel_wise=True,
                input_layout="BHWC",
            )

            torch.manual_seed(7)
            batch = torch.randn((2, 64, 64, 1), dtype=torch.float32)
            result = transform({"image": batch})
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
        dtype: Any = "float32",
        *,
        input_layout: str,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.offset = offset
        self.scale = scale
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.dtype = dtype
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            transform_name=type(self).__name__,
        )

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
                    "input_layout": self.input_layout,
                },
                applied=True,
                random=False,
                invertible=False,
            )
        )
        return bundle

    def normalize_tensor(self, tensor: Any) -> Any:
        """Normalize one tensor with the configured statistics policy."""
        self._validate_tensor_layout(tensor)
        tensor = ops.convert_to_tensor(tensor)
        working_dtype = self.dtype or tensor.dtype
        tensor = ops.cast(tensor, dtype=working_dtype)
        if self.channel_wise:
            normalized = self._normalize_channel_wise(tensor)
        else:
            normalized = self._normalize_global(tensor)
        return ops.cast(normalized, working_dtype)

    def _normalize_channel_wise(self, tensor: Any) -> Any:
        mask = (
            ops.not_equal(tensor, 0.0) if self.nonzero else ops.cast(ops.ones_like(tensor), "bool")
        )
        num_dims = len(ops.shape(tensor))
        spatial_axes = tuple(range(num_dims - 1))
        mask_f = ops.cast(mask, tensor.dtype)
        count = ops.sum(mask_f, axis=spatial_axes, keepdims=True)
        safe_count = ops.where(count > 0, count, ops.ones_like(count))

        masked_tensor = ops.where(mask, tensor, ops.zeros_like(tensor))
        mean = ops.sum(masked_tensor, axis=spatial_axes, keepdims=True) / safe_count
        centered = tensor - mean
        masked_sq = ops.where(mask, ops.square(centered), ops.zeros_like(centered))
        std = ops.sqrt(ops.sum(masked_sq, axis=spatial_axes, keepdims=True) / safe_count)

        sub = mean if self.offset is None else ops.cast(self.offset, tensor.dtype)
        div = std if self.scale is None else ops.cast(self.scale, tensor.dtype)
        div = ops.where(ops.equal(div, 0.0), ops.ones_like(div), div)
        normalized = (tensor - sub) / div
        if self.nonzero:
            normalized = ops.where(mask, normalized, tensor)
        return ops.where(count > 0, normalized, tensor)

    def _normalize_global(self, tensor: Any) -> Any:
        mask = (
            ops.not_equal(tensor, 0.0) if self.nonzero else ops.cast(ops.ones_like(tensor), "bool")
        )
        mask_f = ops.cast(mask, tensor.dtype)
        count = ops.sum(mask_f)
        safe_count = ops.where(count > 0, count, ops.ones_like(count))

        masked_tensor = ops.where(mask, tensor, ops.zeros_like(tensor))
        mean = ops.sum(masked_tensor) / safe_count
        centered = tensor - mean
        masked_sq = ops.where(mask, ops.square(centered), ops.zeros_like(centered))
        std = ops.sqrt(ops.sum(masked_sq) / safe_count)

        std = ops.where(ops.equal(std, 0.0), 1.0, std)
        sub = mean if self.offset is None else ops.cast(self.offset, tensor.dtype)
        div = std if self.scale is None else ops.cast(self.scale, tensor.dtype)
        div = ops.where(ops.equal(div, 0.0), 1.0, div)
        normalized = (tensor - sub) / div
        if self.nonzero:
            normalized = ops.where(mask, normalized, tensor)
        return ops.where(count > 0, normalized, tensor)

    def _validate_tensor_layout(self, tensor: Any) -> None:
        """Validate sample or batch channel-last layout for normalization."""
        validate_tensor_matches_layout(
            tensor,
            self.input_layout,
            transform_name=type(self).__name__,
        )

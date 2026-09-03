from numbers import Number
from typing import Any, Sequence, Union

import keras
from keras import ops

from ..base import InvertibleTransform, KeyedTransform, _pop_last_transform_trace
from ..tensor_bundle import TensorBundle
from ..utils import (
    resolve_input_layout,
    validate_tensor_matches_layout,
)


class ShiftIntensity(KeyedTransform, InvertibleTransform):
    """Deterministically shift tensor intensities by an additive offset.

    ``ShiftIntensity`` adds a scalar or broadcastable per-channel offset to
    each selected tensor. It is the deterministic building block used by
    random intensity-shift augmentations and can also be used directly for
    fixed preprocessing adjustments.

    The provided offset must be broadcast-compatible with the selected tensor
    shape. Its inverse subtracts the recorded offset from the same traced keys
    and still honors ``allow_missing_keys``.

    Args:
        keys: Keys of the tensors to shift.
        offset: Scalar offset or per-channel offset tensor broadcastable to
            the selected tensors.
        input_layout: Channel-last tensor layout. Supported values are
            ``"HWC"``, ``"DHWC"``, ``"BHWC"``, and ``"BDHWC"``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:

        TensorFlow backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import ShiftIntensity

            transform = ShiftIntensity(keys=["image"], offset=0.1, input_layout="HWC")

            image = tf.random.normal((64, 64, 1))
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        JAX backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            from medicai.transforms import ShiftIntensity

            transform = ShiftIntensity(keys=["image"], offset=0.1, input_layout="DHWC")

            image = jax.random.normal(
                jax.random.PRNGKey(7), shape=(32, 64, 64, 1)
            )
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Torch backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import ShiftIntensity

            transform = ShiftIntensity(keys=["image"], offset=0.1, input_layout="BHWC")

            torch.manual_seed(7)
            batch = torch.randn((2, 64, 64, 1))
            result = transform({"image": batch})
            output = result["image"]
            print(output.shape)

    Returns:
        ``TensorBundle``: The input bundle with selected tensors shifted in
        place and an invertible trace entry appended.

    Raises:
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
    """

    def __init__(
        self,
        keys: Sequence[str],
        offset: Union[float, Any],
        *,
        input_layout: str,
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.offset = offset
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            transform_name=type(self).__name__,
        )

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        params = self.get_transform_params(bundle)
        present_keys = self.apply_to_present_keys(
            bundle, lambda tensor, _: self.transform_tensor(tensor, params)
        )
        self.record_transform(bundle, self.build_trace_params(params, present_keys))
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = self._get_last_shift_trace(bundle)
        if trace is None:
            return bundle

        offset = trace["params"].get("offset", self.offset)

        def inverse_shift(tensor: Any, _: str) -> Any:
            inverse_offset = (
                -offset if isinstance(offset, Number) else -ops.cast(offset, tensor.dtype)
            )
            return self.shift_tensor(tensor, offset=inverse_offset)

        self.apply_to_present_keys(
            bundle,
            inverse_shift,
            keys=trace["params"].get("keys", []),
        )
        return bundle

    def get_transform_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Prepare forward-pass parameters for this intensity shift."""
        del bundle
        return {
            "offset": self.offset,
            "input_layout": self.input_layout,
        }

    def transform_tensor(self, tensor: Any, params: dict[str, object]) -> Any:
        """Shift one tensor using the prepared transform parameters."""
        self._validate_tensor_layout(tensor)
        return self.shift_tensor(tensor, offset=params["offset"])

    def build_trace_params(
        self,
        params: dict[str, object],
        present_keys: Sequence[str],
    ) -> dict[str, object]:
        """Build invertible trace metadata for the current intensity shift."""
        return {
            "keys": list(present_keys),
            "offset": params["offset"],
            "input_layout": params["input_layout"],
        }

    def shift_tensor(self, tensor: Any, offset: Union[float, Any, None] = None) -> Any:
        """Shift one tensor by a scalar or broadcastable offset.

        This kernel is agnostic to sample vs batch layout because it performs
        only elementwise addition with TensorFlow broadcasting.
        """
        offset_value = self.offset if offset is None else offset
        # Keep Python numeric scalars as scalars. In the Torch backend, casting
        # them first can create a CPU tensor which cannot be added to a CUDA
        # input tensor.
        if isinstance(offset_value, Number):
            return tensor + offset_value

        if keras.config.backend() == "torch":
            # Standalone transforms do not inherit a Keras layer device scope.
            # Keras random tensors can therefore be created on CPU while the
            # input is on CUDA; align only this backend-specific tensor here.
            import torch

            if isinstance(offset_value, torch.Tensor):
                offset_value = offset_value.to(device=tensor.device, dtype=tensor.dtype)
                return tensor + offset_value

        offset_value = ops.cast(offset_value, dtype=tensor.dtype)
        return tensor + offset_value

    def _validate_tensor_layout(self, tensor: Any) -> None:
        """Validate sample or batch channel-last layout for intensity shifting."""
        validate_tensor_matches_layout(
            tensor,
            self.input_layout,
            transform_name=type(self).__name__,
        )

    def _get_last_shift_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)

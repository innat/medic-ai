from typing import Sequence, Union

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform, _pop_last_transform_trace
from ..tensor_bundle import TensorBundle
from ..utils import validate_input_mode, validate_layout


class ShiftIntensity(KeyedTransform, InvertibleTransform):
    """Deterministically shift tensor intensities by an additive offset.

    ``ShiftIntensity`` adds a scalar or broadcastable per-channel offset to
    each selected tensor. It is the deterministic building block used by
    random intensity-shift augmentations and can also be used directly for
    fixed preprocessing adjustments.

    Depending on ``input_mode``, this transform supports:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

    The provided offset must be broadcast-compatible with the selected tensor
    shape. Its inverse subtracts the recorded offset from the same traced keys
    and still honors ``allow_missing_keys``.

    Args:
        keys: Keys of the tensors to shift.
        offset: Scalar offset or per-channel offset tensor broadcastable to
            the selected tensors.
        input_mode: Either ``"sample"`` for ``(H, W, C)`` / ``(D, H, W, C)``
            tensors, or ``"batch"`` for ``(B, H, W, C)`` / ``(B, D, H, W, C)``
            tensors.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Add a fixed offset to a 2D image using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import ShiftIntensity

            transform = ShiftIntensity(keys=["image"], offset=0.1)

            image = tf.random.normal((64, 64, 1))
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Add a fixed offset to a 3D image volume using a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import ShiftIntensity, TensorBundle

            transform = ShiftIntensity(keys=["image"], offset=0.1)

            image = tf.random.normal((32, 64, 64, 1))
            bundle = TensorBundle({"image": image})
            result = transform(bundle)
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
        offset: Union[float, tf.Tensor],
        input_mode: str = "sample",
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.offset = offset
        self.input_mode = validate_input_mode(input_mode, transform_name=type(self).__name__)

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
        self.apply_to_present_keys(
            bundle,
            lambda tensor, _: self.shift_tensor(tensor, offset=-tf.cast(offset, tensor.dtype)),
            keys=trace["params"].get("keys", []),
        )
        return bundle

    def get_transform_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Prepare forward-pass parameters for this intensity shift."""
        del bundle
        return {"offset": self.offset, "input_mode": self.input_mode}

    def transform_tensor(self, tensor: tf.Tensor, params: dict[str, object]) -> tf.Tensor:
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
            "input_mode": params["input_mode"],
        }

    def shift_tensor(
        self, tensor: tf.Tensor, offset: Union[float, tf.Tensor, None] = None
    ) -> tf.Tensor:
        """Shift one tensor by a scalar or broadcastable offset.

        This kernel is agnostic to sample vs batch layout because it performs
        only elementwise addition with TensorFlow broadcasting.
        """
        offset_value = tf.cast(self.offset if offset is None else offset, dtype=tensor.dtype)
        return tensor + offset_value

    def _validate_tensor_layout(self, tensor: tf.Tensor) -> None:
        """Validate sample or batch channel-last layout for intensity shifting."""
        validate_layout(
            tensor,
            input_mode=self.input_mode,
            allowed_spatial_ranks=(2, 3),
            transform_name=type(self).__name__,
        )

    def _get_last_shift_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)

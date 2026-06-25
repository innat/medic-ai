from typing import Sequence, Union

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform, _pop_last_transform_trace
from ..tensor_bundle import TensorBundle


class ShiftIntensity(KeyedTransform, InvertibleTransform):
    """Deterministically shift tensor intensities by an additive offset.

    ``ShiftIntensity`` adds a scalar or broadcastable per-channel offset to
    each selected tensor. It is the deterministic building block used by
    random intensity-shift augmentations and can also be used directly for
    fixed preprocessing adjustments.

    The transform expects channel-last tensors such as ``(H, W, C)`` or
    ``(D, H, W, C)``. The provided offset must be broadcast-compatible with the
    selected tensor shape.

    Args:
        keys: Keys of the tensors to shift.
        offset: Scalar offset or per-channel offset tensor broadcastable to
            the selected tensors.
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
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.offset = offset

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present_keys = self.apply_to_present_keys(
            bundle, lambda tensor, _: self.shift_tensor(tensor)
        )
        self.record_transform(bundle, {"keys": list(present_keys), "offset": self.offset})
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

    def shift_tensor(
        self, tensor: tf.Tensor, offset: Union[float, tf.Tensor, None] = None
    ) -> tf.Tensor:
        """Shift one tensor by a scalar or broadcastable offset."""
        offset_value = tf.cast(self.offset if offset is None else offset, dtype=tensor.dtype)
        return tensor + offset_value

    def _get_last_shift_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)

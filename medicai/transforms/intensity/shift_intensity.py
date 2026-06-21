from typing import Sequence, Union

import tensorflow as tf

from ..base import KeyedTransform
from ..tensor_bundle import TensorBundle


class ShiftIntensity(KeyedTransform):
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
        offsets: Scalar offset or per-channel offset tensor broadcastable to
            the selected tensors.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Add a fixed offset to a 2D image using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import ShiftIntensity

            transform = ShiftIntensity(keys=["image"], offsets=0.1)

            image = tf.random.normal((64, 64, 1))
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Add a fixed offset to a 3D image volume using a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import ShiftIntensity, TensorBundle

            transform = ShiftIntensity(keys=["image"], offsets=0.1)

            image = tf.random.normal((32, 64, 64, 1))
            bundle = TensorBundle({"image": image})
            result = transform(bundle)
            output = result["image"]
            print(output.shape)

    Returns:
        ``TensorBundle``: The input bundle with selected tensors shifted in
        place.

    Raises:
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
    """

    def __init__(
        self,
        keys: Sequence[str],
        offsets: Union[float, tf.Tensor],
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.offsets = offsets

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present_keys = self.apply_to_present_keys(bundle, lambda tensor, _: self.shift_tensor(tensor))
        bundle.push_transform(
            self.build_trace_entry(
                params={"keys": list(present_keys), "offsets": self.offsets},
                applied=True,
                random=False,
                invertible=False,
            )
        )
        return bundle

    def shift_tensor(
        self, tensor: tf.Tensor, offsets: Union[float, tf.Tensor, None] = None
    ) -> tf.Tensor:
        """Shift one tensor by a scalar or broadcastable offset."""
        offset = tf.cast(self.offsets if offsets is None else offsets, dtype=tensor.dtype)
        return tensor + offset

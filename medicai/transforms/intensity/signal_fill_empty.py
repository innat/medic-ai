from typing import Sequence

import tensorflow as tf

from ..base import KeyedTransform
from ..tensor_bundle import TensorBundle


class SignalFillEmpty(KeyedTransform):
    """Replace invalid floating-point signal values in selected tensors.

    ``SignalFillEmpty`` sanitizes tensors by replacing ``NaN`` values and
    positive or negative infinities with finite values. This is useful before
    downstream normalization, resampling, or batching steps that assume valid
    numeric inputs.

    The transform is intended for image-like tensors in channel-last layout,
    such as ``(H, W, C)`` or ``(D, H, W, C)``.

    Args:
        keys: Keys of the tensors to sanitize.
        replacement: Value used for ``NaN`` entries. Positive and negative
            infinity values default to the largest and smallest finite
            ``float32`` values unless overridden in :meth:`nan_to_num`.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Replace invalid values in a 2D image using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import SignalFillEmpty

            transform = SignalFillEmpty(keys=["image"], replacement=0.0)

            image = tf.constant([[[float("nan")], [1.0]]], dtype=tf.float32)
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Sanitize a 3D image volume using a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import SignalFillEmpty, TensorBundle

            transform = SignalFillEmpty(keys=["image"], replacement=0.0)

            image = tf.random.normal((16, 32, 32, 1))
            image = tf.tensor_scatter_nd_update(
                image,
                indices=[[0, 0, 0, 0]],
                updates=[float("nan")],
            )

            bundle = TensorBundle({"image": image})
            result = transform(bundle)
            output = result["image"]
            print(output.shape)

    Returns:
        ``TensorBundle``: The input bundle with selected tensors sanitized in
        place and a non-invertible trace entry appended.

    Raises:
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
    """

    def __init__(
        self,
        keys: Sequence[str],
        replacement: float = 0.0,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.replacement = replacement

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present_keys = self.apply_to_present_keys(bundle, lambda tensor, _: self.nan_to_num(tensor))
        bundle.push_transform(
            self.build_trace_entry(
                params={"keys": list(present_keys), "replacement": self.replacement},
                applied=True,
                random=False,
                invertible=False,
            )
        )
        return bundle

    def nan_to_num(
        self,
        tensor: tf.Tensor,
        nan: float | None = None,
        posinf: float | None = None,
        neginf: float | None = None,
    ) -> tf.Tensor:
        """Replace NaN, positive infinity, and negative infinity values in a tensor."""
        original_dtype = tensor.dtype
        compute_dtype = tf.float32
        tensor = tf.cast(tensor, compute_dtype)
        dtype_limits = tf.experimental.numpy.finfo(original_dtype.as_numpy_dtype)

        nan = self.replacement if nan is None else nan
        posinf = dtype_limits.max if posinf is None else posinf
        neginf = dtype_limits.min if neginf is None else neginf

        tensor = tf.where(tf.math.is_nan(tensor), tf.cast(nan, compute_dtype), tensor)
        tensor = tf.where(
            tf.math.is_inf(tensor) & (tensor > 0),
            tf.cast(posinf, compute_dtype),
            tensor,
        )
        tensor = tf.where(
            tf.math.is_inf(tensor) & (tensor < 0),
            tf.cast(neginf, compute_dtype),
            tensor,
        )
        return tf.cast(tensor, original_dtype)

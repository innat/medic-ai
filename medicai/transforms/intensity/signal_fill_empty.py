from typing import Sequence

import tensorflow as tf

from ..base import KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import validate_input_mode, validate_layout


class SignalFillEmpty(KeyedTransform):
    """Replace invalid floating-point signal values in selected tensors.

    ``SignalFillEmpty`` sanitizes tensors by replacing ``NaN`` values and
    positive or negative infinities with finite values. This is useful before
    downstream normalization, resampling, or batching steps that assume valid
    numeric inputs.

    Depending on ``input_mode``, this transform supports:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

    Args:
        keys: Keys of the tensors to sanitize.
        fill_value: Value used for ``NaN`` entries. Positive and negative
            infinity values default to the largest and smallest finite
            ``float32`` values unless overridden in :meth:`nan_to_num`.
        input_mode: Either ``"sample"`` for ``(H, W, C)`` / ``(D, H, W, C)``
            tensors, or ``"batch"`` for ``(B, H, W, C)`` / ``(B, D, H, W, C)``
            tensors.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Replace invalid values in a 2D image using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import SignalFillEmpty

            transform = SignalFillEmpty(keys=["image"], fill_value=0.0)

            image = tf.constant([[[float("nan")], [1.0]]], dtype=tf.float32)
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Sanitize a 3D image volume using a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import SignalFillEmpty, TensorBundle

            transform = SignalFillEmpty(keys=["image"], fill_value=0.0)

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
        fill_value: float = 0.0,
        input_mode: str = "sample",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.fill_value = fill_value
        self.input_mode = validate_input_mode(input_mode, transform_name=type(self).__name__)

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present_keys = self.apply_to_present_keys(
            bundle, lambda tensor, _: self.transform_tensor(tensor)
        )
        bundle.push_transform(
            self.build_trace_entry(
                params={
                    "keys": list(present_keys),
                    "fill_value": self.fill_value,
                    "input_mode": self.input_mode,
                },
                applied=True,
                random=False,
                invertible=False,
            )
        )
        return bundle

    def transform_tensor(self, tensor: tf.Tensor) -> tf.Tensor:
        """Sanitize one tensor after validating its channel-last layout."""
        self._validate_tensor_layout(tensor)
        return self.nan_to_num(tensor)

    def nan_to_num(
        self,
        tensor: tf.Tensor,
        nan: float | None = None,
        posinf: float | None = None,
        neginf: float | None = None,
    ) -> tf.Tensor:
        """Replace NaN, positive infinity, and negative infinity values in a tensor."""
        tensor = tf.cast(tf.convert_to_tensor(tensor), tf.float32)
        return self.nan_to_num_batch(tensor, nan=nan, posinf=posinf, neginf=neginf)

    def nan_to_num_batch(
        self,
        tensor: tf.Tensor,
        nan: float | None = None,
        posinf: float | None = None,
        neginf: float | None = None,
    ) -> tf.Tensor:
        """Sanitize a tensor with a kernel that is agnostic to sample vs batch layout."""

        nan = self.fill_value if nan is None else nan
        posinf = tf.float32.max if posinf is None else posinf
        neginf = -tf.float32.max if neginf is None else neginf

        tensor = tf.where(tf.math.is_nan(tensor), nan, tensor)
        tensor = tf.where(
            tf.math.is_inf(tensor) & (tensor > 0),
            posinf,
            tensor,
        )
        tensor = tf.where(
            tf.math.is_inf(tensor) & (tensor < 0),
            neginf,
            tensor,
        )
        return tensor

    def _validate_tensor_layout(self, tensor: tf.Tensor) -> None:
        """Validate sample or batch channel-last layout for signal sanitization."""
        validate_layout(
            tensor,
            input_mode=self.input_mode,
            allowed_spatial_ranks=(2, 3),
            transform_name=type(self).__name__,
        )

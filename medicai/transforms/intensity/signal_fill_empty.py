from typing import Sequence

import numpy as np
from keras import ops

from ..base import KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import (
    resolve_input_layout,
    validate_tensor_matches_layout,
)


class SignalFillEmpty(KeyedTransform):
    """Replace invalid floating-point signal values in selected tensors.

    ``SignalFillEmpty`` sanitizes tensors by replacing ``NaN`` values and
    positive or negative infinities with finite values. This is useful before
    downstream normalization, resampling, or batching steps that assume valid
    numeric inputs.

    Depending on ``input_layout``, this transform supports:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

    Args:
        keys: Keys of the tensors to sanitize.
        fill_value: Value used for ``NaN`` entries. Positive and negative
            infinity values default to the largest and smallest finite
            ``float32`` values unless overridden in :meth:`nan_to_num`.
        input_layout: Channel-last tensor layout. Supported values are
            ``"HWC"``, ``"DHWC"``, ``"BHWC"``, and ``"BDHWC"``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Replace invalid values in a 2D image using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import SignalFillEmpty

            transform = SignalFillEmpty(keys=["image"], fill_value=0.0, input_layout="HWC")

            image = tf.constant([[[float("nan")], [1.0]]], dtype=tf.float32)
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Sanitize a 3D image volume using a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import SignalFillEmpty, TensorBundle

            transform = SignalFillEmpty(keys=["image"], fill_value=0.0, input_layout="DHWC")

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
        *,
        input_layout: str,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.fill_value = fill_value
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            transform_name=type(self).__name__,
        )

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present_keys = self.apply_to_present_keys(
            bundle, lambda tensor, _: self.transform_tensor(tensor)
        )
        bundle.push_transform(
            self.build_trace_entry(
                params={
                    "keys": list(present_keys),
                    "fill_value": self.fill_value,
                    "input_layout": self.input_layout,
                },
                applied=True,
                random=False,
                invertible=False,
            )
        )
        return bundle

    def transform_tensor(self, tensor):
        """Sanitize one tensor after validating its channel-last layout."""
        self._validate_tensor_layout(tensor)
        return self.nan_to_num(tensor)

    def nan_to_num(
        self,
        tensor,
        nan: float | None = None,
        posinf: float | None = None,
        neginf: float | None = None,
    ):
        """Replace NaN, positive infinity, and negative infinity values in a tensor."""
        tensor = ops.cast(ops.convert_to_tensor(tensor), "float32")
        return self.nan_to_num_batch(tensor, nan=nan, posinf=posinf, neginf=neginf)

    def nan_to_num_batch(
        self,
        tensor,
        nan: float | None = None,
        posinf: float | None = None,
        neginf: float | None = None,
    ):
        """Sanitize a tensor with a kernel that is agnostic to sample vs batch layout."""

        nan = self.fill_value if nan is None else nan
        max_float32 = float(np.finfo(np.float32).max)
        posinf = max_float32 if posinf is None else posinf
        neginf = -max_float32 if neginf is None else neginf

        tensor = ops.where(ops.isnan(tensor), nan, tensor)
        tensor = ops.where(
            ops.logical_and(ops.isinf(tensor), tensor > 0),
            posinf,
            tensor,
        )
        tensor = ops.where(
            ops.logical_and(ops.isinf(tensor), tensor < 0),
            neginf,
            tensor,
        )
        return tensor

    def _validate_tensor_layout(self, tensor) -> None:
        """Validate sample or batch channel-last layout for signal sanitization."""
        validate_tensor_matches_layout(
            tensor,
            self.input_layout,
            transform_name=type(self).__name__,
        )

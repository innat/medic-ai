from typing import Sequence, Union

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import normalize_axes


class Flip(KeyedTransform, InvertibleTransform):
    """Deterministically flip selected tensors along one or more axes.

    This transform reverses each selected tensor along the requested axes. It
    supports both 2D channel-last tensors ``(H, W, C)`` and 3D channel-last
    tensors ``(D, H, W, C)``.

    Args:
        keys: Keys of the tensors to flip.
        spatial_axis: Axis or axes passed to ``tf.reverse``. If ``None``, the
            transform is a no-op.
        allow_missing_keys: If ``True``, missing keys are skipped.
    """

    def __init__(
        self,
        keys: Sequence[str],
        spatial_axis: Union[int, Sequence[int], None] = None,
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.spatial_axis = spatial_axis

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        if self.spatial_axis is None:
            return bundle

        normalized_axes = {"value": None}

        def apply_flip(tensor: tf.Tensor, _: str) -> tf.Tensor:
            normalized_axes["value"] = self._resolve_axes(tensor)
            return self.flip_tensor(tensor)

        present_keys = self.apply_to_present_keys(bundle, apply_flip)

        if normalized_axes["value"] is not None:
            self.record_transform(
                bundle,
                {
                    "keys": list(present_keys),
                    "spatial_axis": normalized_axes["value"],
                },
            )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        if self.spatial_axis is None:
            return bundle

        self.apply_to_present_keys(bundle, lambda tensor, _: self.flip_tensor(tensor))
        return bundle

    def flip_tensor(self, tensor: tf.Tensor) -> tf.Tensor:
        """Flip one tensor using the configured axes."""
        if self.spatial_axis is None:
            return tensor
        return tf.reverse(tensor, axis=self._resolve_axes(tensor))

    def _resolve_axes(self, tensor: tf.Tensor) -> tuple[int, ...]:
        axes = self.spatial_axis
        if isinstance(axes, int):
            axes = (axes,)
        if axes is None:
            return ()
        return normalize_axes(tuple(axes), rank=tensor.shape.rank, name="spatial_axis")

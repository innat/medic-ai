from typing import Sequence

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import get_spatial_rank, normalize_axes


class Rotate90(KeyedTransform, InvertibleTransform):
    """Rotate selected tensors by multiples of 90 degrees.

    This transform rotates tensors in a plane defined by ``spatial_axes``. It
    supports both 2D channel-last tensors ``(H, W, C)`` and 3D channel-last
    tensors ``(D, H, W, C)``.

    Args:
        keys: Keys of the tensors to rotate.
        k: Number of quarter turns.
        spatial_axes: Two axes defining the rotation plane. If ``None``, the
            transform rotates in the last two spatial dimensions.
        allow_missing_keys: If ``True``, missing keys are skipped.
    """

    def __init__(
        self,
        keys: Sequence[str],
        k: int = 1,
        spatial_axes: Sequence[int] | None = None,
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.k = k
        self.spatial_axes = spatial_axes

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        effective_k = self.k % 4

        if effective_k == 0:
            return bundle

        normalized_axes = {"value": None}

        def apply_rotate(tensor: tf.Tensor, _: str) -> tf.Tensor:
            normalized_axes["value"] = self._resolve_axes(tensor)
            return self.rotate_tensor(tensor)

        present_keys = self.apply_to_present_keys(bundle, apply_rotate)

        if normalized_axes["value"] is not None:
            self.record_transform(
                bundle,
                {
                    "keys": list(present_keys),
                    "k": effective_k,
                    "spatial_axes": normalized_axes["value"],
                },
            )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        effective_k = self.k % 4
        if effective_k == 0:
            return bundle

        inverse_k = (-effective_k) % 4
        self.apply_to_present_keys(bundle, lambda tensor, _: self.rotate_tensor(tensor, k=inverse_k))
        return bundle

    def rotate_tensor(self, tensor: tf.Tensor, k: int | tf.Tensor | None = None) -> tf.Tensor:
        """Rotate one tensor by multiples of 90 degrees."""
        axes = self._resolve_axes(tensor)
        effective_k = tf.math.floormod(
            tf.cast(self.k if k is None else k, tf.int32),
            4,
        )

        return tf.switch_case(
            effective_k,
            branch_fns={
                0: lambda: tensor,
                1: lambda: self._rotate_once(tensor, axes),
                2: lambda: self._rotate_once(self._rotate_once(tensor, axes), axes),
                3: lambda: self._rotate_once(
                    self._rotate_once(self._rotate_once(tensor, axes), axes), axes
                ),
            },
        )

    def _rotate_once(self, tensor: tf.Tensor, axes: tuple[int, int]) -> tf.Tensor:
        perm = [axis for axis in range(tensor.shape.rank) if axis not in axes] + list(axes)
        transposed = tf.transpose(tensor, perm=perm)
        perm_len = len(perm)
        rotated = tf.transpose(
            tf.reverse(transposed, axis=[perm_len - 1]),
            perm=[*range(perm_len - 2), perm_len - 1, perm_len - 2],
        )

        inverse_perm = [0] * len(perm)
        for index, axis in enumerate(perm):
            inverse_perm[axis] = index
        return tf.transpose(rotated, perm=inverse_perm)

    def _resolve_axes(self, tensor: tf.Tensor) -> tuple[int, int]:
        if tensor.shape.rank is None or tensor.shape.rank < 3:
            raise ValueError(
                "Rotate90 expects a channel-last sample tensor with shape (H, W, C) or "
                f"(D, H, W, C). Received shape {tensor.shape}."
            )

        if self.spatial_axes is None:
            spatial_rank = get_spatial_rank(tensor)
            if spatial_rank < 2:
                raise ValueError("Rotate90 requires at least two spatial dimensions.")
            return (spatial_rank - 2, spatial_rank - 1)

        axes = normalize_axes(tuple(self.spatial_axes), rank=tensor.shape.rank, name="spatial_axes")
        if len(axes) != 2:
            raise ValueError("`spatial_axes` must contain exactly two axes.")
        return axes

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import get_spatial_rank, normalize_spatial_axes


class Rotate90(KeyedTransform, InvertibleTransform):
    """Rotate selected tensors by quarter turns in a spatial plane.

    ``Rotate90`` deterministically rotates channel-last sample tensors by
    multiples of 90 degrees. The rotation plane is selected through
    ``spatial_axis`` and can be used for:

    - 2D tensors shaped ``(H, W, C)``
    - 3D tensors shaped ``(D, H, W, C)``

    For 2D tensors, leaving ``spatial_axis=None`` rotates in the image plane.
    For 3D tensors, the default also rotates within the last two spatial
    dimensions, preserving the leading depth axis.

    The transform is invertible because the inverse is another quarter-turn
    rotation with ``(-k) % 4``. Applied parameters are recorded in the
    ``TensorBundle`` trace for inspection and downstream inversion workflows.

    Args:
        keys: Keys of the tensors to rotate.
        k: Number of 90-degree counterclockwise quarter turns.
        spatial_axis: Two axes defining the rotation plane. If ``None``, the
            transform rotates in the last two spatial dimensions.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Rotate a 2D image-label pair by 90 degrees:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Rotate90

            transform = Rotate90(keys=["image", "label"], k=1)

            image = tf.random.normal((128, 128, 1))
            label = tf.random.uniform((128, 128, 1), maxval=2, dtype=tf.int32)

            result = transform({"image": image, "label": label})

        Rotate a tensor and then restore it with ``inverse()``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Rotate90, TensorBundle

            transform = Rotate90(keys=["image"], k=1)

            image = tf.random.normal((64, 64, 1))
            forward = transform({"image": image})
            restored = transform.inverse(forward)
            print(forward["image"].shape)
            print(restored["image"].shape)

        Rotate a 3D image-label pair by 90 degrees:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Rotate90

            transform = Rotate90(keys=["image", "label"], k=1)

            image = tf.random.normal((64, 128, 128, 1))
            label = tf.random.uniform(
                (64, 128, 128, 1), maxval=2, dtype=tf.int32
            )
            result = transform({"image": image, "label": label})
            print(result["image"].shape)
            print(result["label"].shape)

    Returns:
        ``TensorBundle``: The input bundle with rotated tensors and an
        invertible transform trace entry when ``k % 4 != 0``.

    Raises:
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
        ValueError: If a selected tensor is not channel-last 2D or 3D, or if
            ``spatial_axis`` does not resolve to exactly two valid axes.
    """

    def __init__(
        self,
        keys: Sequence[str],
        k: int = 1,
        spatial_axis: Sequence[int] | None = None,
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.k = k
        self.spatial_axis = spatial_axis

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
                    "spatial_axis": normalized_axes["value"],
                },
            )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        effective_k = self.k % 4
        if effective_k == 0:
            return bundle

        inverse_k = (-effective_k) % 4
        self.apply_to_present_keys(
            bundle, lambda tensor, _: self.rotate_tensor(tensor, k=inverse_k)
        )
        return bundle

    def rotate_tensor(self, tensor: tf.Tensor, k: int | tf.Tensor | None = None) -> tf.Tensor:
        """Rotate one tensor by multiples of 90 degrees.

        Args:
            tensor: Channel-last 2D or 3D sample tensor.
            k: Optional quarter-turn override. When ``None``, ``self.k`` is
                used.

        Returns:
            ``tf.Tensor``: The rotated tensor.
        """
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
                2: lambda: tf.reverse(tensor, axis=axes),
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
                f"{type(self).__name__} expects a channel-last sample tensor with shape (H, W, C) or "
                f"(D, H, W, C). Received shape {tensor.shape}."
            )

        if self.spatial_axis is None:
            spatial_rank = get_spatial_rank(tensor)
            if spatial_rank < 2:
                raise ValueError(f"{type(self).__name__} requires at least two spatial dimensions.")
            return (spatial_rank - 2, spatial_rank - 1)

        spatial_rank = get_spatial_rank(tensor)
        axes = normalize_spatial_axes(tuple(self.spatial_axis), spatial_rank, name="spatial_axis")
        if len(axes) != 2:
            raise ValueError("`spatial_axis` must contain exactly two axes.")
        return axes

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import (
    ensure_batch_axis,
    resolve_spatial_axes,
    restore_from_batch_axis,
    validate_input_mode,
    validate_layout,
)


class Rotate90(KeyedTransform, InvertibleTransform):
    """Rotate selected tensors by quarter turns in a spatial plane.

    ``Rotate90`` deterministically rotates channel-last tensors by
    multiples of 90 degrees. The rotation plane is selected through
    ``spatial_axis``. Depending on ``input_mode``, it can be used for:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

    For 2D tensors, leaving ``spatial_axis=None`` rotates in the image plane.
    For 3D tensors, the default also rotates within the last two spatial
    dimensions, preserving the leading depth axis in sample mode or the
    leading batch-plus-depth structure in batch mode.

    The transform is invertible because the inverse is another quarter-turn
    rotation with ``(-k) % 4``. Applied parameters are recorded in the
    ``TensorBundle`` trace for inspection and downstream inversion workflows.

    Args:
        keys: Keys of the tensors to rotate.
        k: Number of 90-degree counterclockwise quarter turns.
        spatial_axis: Two axes defining the rotation plane. If ``None``, the
            transform rotates in the last two spatial dimensions. In batch
            mode, axes are still expressed relative to the spatial dimensions
            only, so the batch axis is never included.
        input_mode: Either ``"sample"`` for ``(H, W, C)`` / ``(D, H, W, C)``
            tensors, or ``"batch"`` for ``(B, H, W, C)`` / ``(B, D, H, W, C)``
            tensors.
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
        input_mode: str = "sample",
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.k = k
        self.spatial_axis = spatial_axis
        self.input_mode = validate_input_mode(input_mode, transform_name=type(self).__name__)

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        params = self.get_transform_params(bundle)
        if not params["applied"]:
            return bundle

        present_keys = self.apply_to_present_keys(
            bundle,
            lambda tensor, key: self.transform_tensor(tensor, key, params),
        )
        self.record_transform(bundle, self.build_trace_params(params, present_keys))
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        params = self.get_transform_params(bundle)
        if not params["applied"]:
            return bundle

        inverse_k = (-params["k"]) % 4
        self.apply_to_present_keys(
            bundle,
            lambda tensor, key: self.transform_tensor(
                tensor,
                key,
                {
                    "applied": True,
                    "k": inverse_k,
                    "spatial_axis": params["spatial_axis"],
                    "input_mode": params["input_mode"],
                },
            ),
        )
        return bundle

    def get_transform_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Prepare forward-pass parameters for this rotation."""
        del bundle
        return {
            "applied": self.k % 4 != 0,
            "k": self.k % 4,
            "spatial_axis": self.spatial_axis,
            "input_mode": self.input_mode,
        }

    def transform_tensor(
        self,
        tensor: tf.Tensor,
        key: str,
        params: dict[str, object],
    ) -> tf.Tensor:
        """Apply the configured quarter-turn kernel to one tensor."""
        del key
        return self.rotate_tensor(
            tensor,
            k=params["k"],
            spatial_axis=params["spatial_axis"],
        )

    def build_trace_params(
        self,
        params: dict[str, object],
        present_keys: Sequence[str],
    ) -> dict[str, object]:
        """Build invertible trace metadata for the current rotation."""
        return {
            "keys": list(present_keys),
            "k": params["k"],
            "spatial_axis": params["spatial_axis"],
            "input_mode": params["input_mode"],
        }

    def rotate_tensor(
        self,
        tensor: tf.Tensor,
        k: int | tf.Tensor | None = None,
        spatial_axis: Sequence[int] | None = None,
    ) -> tf.Tensor:
        """Rotate one tensor by multiples of 90 degrees.

        Args:
            tensor: Channel-last 2D or 3D tensor in sample or batch layout,
                depending on ``self.input_mode``.
            k: Optional quarter-turn override. When ``None``, ``self.k`` is
                used.
            spatial_axis: Optional rotation plane override. When ``None``,
                ``self.spatial_axis`` is used.

        Returns:
            ``tf.Tensor``: The rotated tensor.
        """
        batched_tensor, added_batch_axis = ensure_batch_axis(tensor, input_mode=self.input_mode)
        rotated = self.rotate_batch_tensor(
            batched_tensor,
            k=k,
            spatial_axis=spatial_axis,
        )
        return restore_from_batch_axis(rotated, added_batch_axis)

    def rotate_batch_tensor(
        self,
        tensor: tf.Tensor,
        k: int | tf.Tensor | None = None,
        spatial_axis: Sequence[int] | None = None,
    ) -> tf.Tensor:
        """Rotate a batch-layout tensor by multiples of 90 degrees."""
        axes = self._resolve_axes(
            tensor,
            spatial_axis=spatial_axis,
            input_mode="batch",
        )
        effective_k = tf.math.floormod(tf.cast(self.k if k is None else k, tf.int32), 4)

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

    def _resolve_axes(
        self,
        tensor: tf.Tensor,
        spatial_axis: Sequence[int] | None = None,
        input_mode: str | None = None,
    ) -> tuple[int, int]:
        effective_input_mode = self.input_mode if input_mode is None else input_mode
        layout = validate_layout(
            tensor,
            input_mode=effective_input_mode,
            allowed_spatial_ranks=(2, 3),
        )

        effective_axis = self.spatial_axis if spatial_axis is None else spatial_axis
        if effective_axis is None:
            if layout.spatial_rank < 2:
                raise ValueError(f"{type(self).__name__} requires at least two spatial dimensions.")
            return tuple(layout.spatial_axes[-2:])

        axes = resolve_spatial_axes(
            tensor,
            tuple(effective_axis),
            input_mode=effective_input_mode,
            allowed_spatial_ranks=(2, 3),
            name="spatial_axis",
        )
        if len(axes) != 2:
            raise ValueError("`spatial_axis` must contain exactly two axes.")
        return axes

from __future__ import annotations

from typing import Any, Sequence

from keras import ops

from ..base import InvertibleTransform, KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import (
    ensure_batch_axis_for_layout,
    get_batched_input_layout,
    get_input_layout_info,
    get_tensor_rank,
    resolve_input_layout,
    resolve_input_layout_axes,
    restore_from_batch_axis,
    validate_tensor_matches_layout,
)


def _concrete_quarter_turn(value: Any) -> int | None:
    """Return a concrete quarter-turn value when one is available."""
    if isinstance(value, int):
        return value % 4
    try:
        return int(ops.convert_to_numpy(ops.cast(value, "int32"))) % 4
    except (TypeError, ValueError, RuntimeError):
        return None


class Rotate90(KeyedTransform, InvertibleTransform):
    """Rotate selected tensors by quarter turns in a spatial plane.

    ``Rotate90`` deterministically rotates channel-last tensors by
    multiples of 90 degrees. The rotation plane is selected through
    ``spatial_axis``. Depending on ``input_layout``, it can be used for:

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
        spatial_axis: Two axes defining the rotation plane. When
            ``input_layout`` is provided explicitly, axes refer to the real
            tensor axes of that layout. For example, under ``"BHWC"``, the
            2D image plane is ``(1, 2)``. Under ``"BDHWC"``, common 3D planes
            are axial ``(2, 3)``, coronal ``(1, 3)``, and sagittal ``(1, 2)``.
        input_layout: Channel-last tensor layout. Supported values are
            ``"HWC"``, ``"DHWC"``, ``"BHWC"``, and ``"BDHWC"``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:

        TensorFlow backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import Rotate90

            transform = Rotate90(keys=["image"], k=1, input_layout="HWC")
            image = tf.random.normal((128, 128, 1), seed=7)
            result = transform({"image": image})
            restored = transform.inverse(result)
            print(result["image"].shape, restored["image"].shape)

        JAX backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            from medicai.transforms import Rotate90

            transform = Rotate90(
                keys=["image"], k=1, spatial_axis=(1, 2), input_layout="DHWC"
            )
            image = jax.random.normal(
                jax.random.PRNGKey(7), shape=(32, 64, 64, 1)
            )
            result = transform({"image": image})
            print(result["image"].shape)

        Torch backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import Rotate90

            transform = Rotate90(
                keys=["image"], k=1, spatial_axis=(1, 2), input_layout="BHWC"
            )
            torch.manual_seed(7)
            batch = torch.randn((2, 64, 64, 1))
            result = transform({"image": batch})
            print(result["image"].shape)

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
        *,
        input_layout: str,
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.k = k
        self.spatial_axis = spatial_axis
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            transform_name=type(self).__name__,
        )
        self.layout_info = get_input_layout_info(self.input_layout)

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
        trace = self._get_last_rotate90_trace(bundle)
        if trace is None:
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
                        "input_layout": params["input_layout"],
                    },
                ),
            )
            return bundle

        params = trace["params"]
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
                    "input_layout": params["input_layout"],
                },
            ),
            keys=params.get("keys", []),
        )
        return bundle

    def get_transform_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Prepare forward-pass parameters for this rotation."""
        del bundle
        return {
            "applied": self.k % 4 != 0,
            "k": self.k % 4,
            "spatial_axis": self.spatial_axis,
            "input_layout": self.input_layout,
        }

    def transform_tensor(
        self,
        tensor: Any,
        key: str,
        params: dict[str, object],
    ) -> Any:
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
            "input_layout": params["input_layout"],
        }

    def rotate_tensor(
        self,
        tensor: Any,
        k: int | Any | None = None,
        spatial_axis: Sequence[int] | None = None,
    ) -> Any:
        """Rotate one tensor by multiples of 90 degrees.

        Args:
            tensor: Channel-last 2D or 3D tensor in sample or batch layout,
                depending on ``self.input_layout``.
            k: Optional quarter-turn override. When ``None``, ``self.k`` is
                used.
            spatial_axis: Optional rotation plane override. When ``None``,
                ``self.spatial_axis`` is used.

        Returns:
            Tensor-like: The rotated tensor.
        """
        validate_tensor_matches_layout(
            tensor,
            self.input_layout,
            transform_name=type(self).__name__,
        )
        axes = self._resolve_axes(tensor, spatial_axis=spatial_axis)
        effective_k = _concrete_quarter_turn(self.k if k is None else k)
        if effective_k is not None:
            if effective_k == 0:
                return tensor
            if effective_k == 1:
                return self._rotate_once(tensor, axes)
            if effective_k == 2:
                return ops.flip(tensor, axis=axes)
            return self._rotate_once(self._rotate_once(self._rotate_once(tensor, axes), axes), axes)

        effective_k = ops.mod(ops.cast(self.k if k is None else k, "int32"), 4)
        if not self.layout_info.batched:
            return ops.switch(
                effective_k,
                [
                    lambda: tensor,
                    lambda: self._rotate_once(tensor, axes),
                    lambda: ops.flip(tensor, axis=axes),
                    lambda: self._rotate_once(
                        self._rotate_once(self._rotate_once(tensor, axes), axes), axes
                    ),
                ],
            )
        batched_tensor, added_batch_axis = ensure_batch_axis_for_layout(
            tensor,
            input_layout=self.input_layout,
        )
        rotated = self.rotate_batch_tensor(
            batched_tensor,
            k=k,
            spatial_axis=spatial_axis,
        )
        return restore_from_batch_axis(rotated, added_batch_axis)

    def rotate_batch_tensor(
        self,
        tensor: Any,
        k: int | Any | None = None,
        spatial_axis: Sequence[int] | None = None,
    ) -> Any:
        """Rotate a batch-layout tensor by multiples of 90 degrees."""
        effective_axis = self.spatial_axis if spatial_axis is None else spatial_axis
        if effective_axis is not None and not self.layout_info.batched:
            effective_axis = tuple(axis + 1 for axis in effective_axis)
        axes = self._resolve_axes(
            tensor,
            spatial_axis=effective_axis,
            input_layout=(
                self.input_layout
                if self.layout_info.batched
                else get_batched_input_layout(self.input_layout)
            ),
        )
        effective_k = _concrete_quarter_turn(self.k if k is None else k)
        if effective_k is not None:
            if effective_k == 0:
                return tensor
            if effective_k == 1:
                return self._rotate_once(tensor, axes)
            if effective_k == 2:
                return ops.flip(tensor, axis=axes)
            return self._rotate_once(self._rotate_once(self._rotate_once(tensor, axes), axes), axes)

        effective_k = ops.mod(ops.cast(self.k if k is None else k, "int32"), 4)

        return ops.switch(
            effective_k,
            [
                lambda: tensor,
                lambda: self._rotate_once(tensor, axes),
                lambda: ops.flip(tensor, axis=axes),
                lambda: self._rotate_once(
                    self._rotate_once(self._rotate_once(tensor, axes), axes), axes
                ),
            ],
        )

    def _rotate_once(self, tensor: Any, axes: tuple[int, int]) -> Any:
        perm = [axis for axis in range(get_tensor_rank(tensor)) if axis not in axes] + list(axes)
        transposed = ops.transpose(tensor, axes=perm)
        perm_len = len(perm)
        rotated = ops.transpose(
            ops.flip(transposed, axis=[perm_len - 1]),
            axes=[*range(perm_len - 2), perm_len - 1, perm_len - 2],
        )

        inverse_perm = [0] * len(perm)
        for index, axis in enumerate(perm):
            inverse_perm[axis] = index
        return ops.transpose(rotated, axes=inverse_perm)

    def _resolve_axes(
        self,
        tensor: Any,
        spatial_axis: Sequence[int] | None = None,
        input_layout: str | None = None,
    ) -> tuple[int, int]:
        layout = validate_tensor_matches_layout(
            tensor,
            self.input_layout if input_layout is None else input_layout,
            transform_name=type(self).__name__,
        )

        effective_axis = self.spatial_axis if spatial_axis is None else spatial_axis
        if effective_axis is None:
            if layout.spatial_rank < 2:
                raise ValueError(f"{type(self).__name__} requires at least two spatial dimensions.")
            return tuple(layout.spatial_axes[-2:])
        axes = resolve_input_layout_axes(
            tensor,
            tuple(effective_axis),
            input_layout=self.input_layout if input_layout is None else input_layout,
            name="spatial_axis",
        )
        if len(axes) != 2:
            raise ValueError("`spatial_axis` must contain exactly two axes.")
        return axes

    def _get_last_rotate90_trace(self, bundle: TensorBundle):
        from ..base import _pop_last_transform_trace

        return _pop_last_transform_trace(bundle, type(self).__name__)

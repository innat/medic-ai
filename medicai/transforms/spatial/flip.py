from typing import Any, Sequence, Union

from keras import ops

from ..base import InvertibleTransform, KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import (
    ensure_batch_axis_for_layout,
    get_batched_input_layout,
    get_input_layout_info,
    resolve_input_layout,
    resolve_input_layout_axes,
    restore_from_batch_axis,
)


class Flip(KeyedTransform, InvertibleTransform):
    """Flip selected tensors along one or more spatial axes.

    ``Flip`` deterministically reverses channel-last tensors using the
    backend-native Keras operations. It can be applied to common Medic-AI
    dictionary-style samples such as image-label pairs.

    The transform is invertible. Applying the same flip twice restores
    the original orientation. During :meth:`apply`, the normalized axes are
    recorded in the ``TensorBundle`` transform trace so downstream pipelines
    can inspect what was applied.

    Args:
        keys: Keys of the tensors to flip.
        spatial_axis: Spatial axis or axes to reverse. When ``input_layout``
            is provided explicitly, axes refer to the real tensor axes of that
            layout. For example, under ``"BHWC"``, ``1`` means height and ``2``
            means width; under ``"BDHWC"``, ``1`` means depth, ``2`` means
            height, and ``3`` means width.
        input_layout: Channel-last tensor layout. Supported values are
            ``"HWC"``, ``"DHWC"``, ``"BHWC"``, and ``"BDHWC"``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:

        TensorFlow backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import Flip

            transform = Flip(keys=["image"], spatial_axis=0, input_layout="HWC")
            image = tf.random.normal((64, 64, 1), seed=7)

            result = transform({"image": image})
            restored = transform.inverse(result)
            print(result["image"].shape, restored["image"].shape)

        JAX backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            from medicai.transforms import Flip

            transform = Flip(keys=["image"], spatial_axis=1, input_layout="DHWC")
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
            from medicai.transforms import Flip

            transform = Flip(keys=["image"], spatial_axis=1, input_layout="BHWC")
            torch.manual_seed(7)
            batch = torch.randn((2, 64, 64, 1))
            result = transform({"image": batch})
            print(result["image"].shape)

    Returns:
        ``TensorBundle``: The input bundle with flipped tensors and an
        invertible transform trace entry when a flip is applied.

    Raises:
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
        ValueError: If ``spatial_axis`` contains invalid axes for a selected
            tensor.
    """

    def __init__(
        self,
        keys: Sequence[str],
        spatial_axis: Union[int, Sequence[int], None] = None,
        *,
        input_layout: str,
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        if spatial_axis is None:
            raise ValueError(
                f"{type(self).__name__} requires `spatial_axis`. "
                "Use an explicit identity path instead of a no-op flip."
            )
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
        trace = self._get_last_flip_trace(bundle)
        if trace is None:
            return bundle
        self.apply_to_present_keys(
            bundle,
            lambda tensor, key: self.transform_tensor(tensor, key, trace["params"]),
            keys=trace["params"].get("keys", []),
        )
        return bundle

    def get_transform_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Prepare forward-pass parameters for this flip."""
        del bundle
        return {
            "applied": self.spatial_axis is not None,
            "spatial_axis": self.spatial_axis,
            "input_layout": self.input_layout,
        }

    def transform_tensor(
        self,
        tensor: Any,
        key: str,
        params: dict[str, object],
    ) -> Any:
        """Apply the configured flip kernel to one tensor."""
        del key
        return self.flip_tensor(tensor, spatial_axis=params["spatial_axis"])

    def build_trace_params(
        self,
        params: dict[str, object],
        present_keys: Sequence[str],
    ) -> dict[str, object]:
        """Build invertible trace metadata for the current flip."""
        return {
            "keys": list(present_keys),
            "spatial_axis": params["spatial_axis"],
            "input_layout": params["input_layout"],
        }

    def flip_tensor(
        self,
        tensor: Any,
        spatial_axis: Union[int, Sequence[int], None] = None,
    ) -> Any:
        """Flip one tensor using explicit spatial axes.

        Args:
            tensor: Channel-last 2D or 3D sample tensor to reverse.
            spatial_axis: Spatial axis or axes to flip. When omitted, the
                transform's configured ``spatial_axis`` is used. When the
                effective value is ``None``, the input tensor is returned
                unchanged.

        Returns:
            The flipped tensor, represented by the active Keras backend.
        """
        effective_axis = self.spatial_axis if spatial_axis is None else spatial_axis
        if effective_axis is None:
            return tensor
        if not self.layout_info.batched:
            return ops.flip(tensor, axis=self._resolve_axes(tensor, spatial_axis=effective_axis))
        batched_tensor, added_batch_axis = ensure_batch_axis_for_layout(
            tensor,
            input_layout=self.input_layout,
        )
        flipped = self.flip_batch_tensor(batched_tensor, spatial_axis=effective_axis)
        return restore_from_batch_axis(flipped, added_batch_axis)

    def flip_batch_tensor(
        self,
        tensor: Any,
        spatial_axis: Union[int, Sequence[int], None] = None,
    ) -> Any:
        """Flip a batched tensor.

        Args:
            tensor: Batched channel-last tensor.
            spatial_axis: Spatial axis or axes to flip. For a batched
                ``input_layout``, axes are real tensor axes of that layout. For
                a sample ``input_layout``, axes are sample-relative and are
                offset by one for the temporary batch axis.

        Returns:
            The flipped tensor, represented by the active Keras backend.
        """
        effective_axis = self.spatial_axis if spatial_axis is None else spatial_axis
        if effective_axis is None:
            return tensor
        if not self.layout_info.batched:
            if isinstance(effective_axis, int):
                effective_axis = effective_axis + 1
            else:
                effective_axis = tuple(axis + 1 for axis in effective_axis)
        return ops.flip(
            tensor,
            axis=self._resolve_axes(
                tensor,
                spatial_axis=effective_axis,
                input_layout=(
                    self.input_layout
                    if self.layout_info.batched
                    else get_batched_input_layout(self.input_layout)
                ),
            ),
        )

    def _resolve_axes(
        self,
        tensor: Any,
        spatial_axis: Union[int, Sequence[int], None] = None,
        input_layout: str | None = None,
    ) -> tuple[int, ...]:
        axes = self.spatial_axis if spatial_axis is None else spatial_axis
        if isinstance(axes, int):
            axes = (axes,)
        if axes is None:
            return ()
        return resolve_input_layout_axes(
            tensor,
            tuple(axes),
            input_layout=self.input_layout if input_layout is None else input_layout,
            name="spatial_axis",
        )

    def _get_last_flip_trace(self, bundle: TensorBundle):
        from ..base import _pop_last_transform_trace

        return _pop_last_transform_trace(bundle, type(self).__name__)

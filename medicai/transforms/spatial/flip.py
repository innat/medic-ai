from typing import Sequence, Union

import tensorflow as tf

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

    ``Flip`` deterministically reverses channel-last tensors using
    TensorFlow's ``tf.reverse``. It can be applied to common Medic-AI
    dictionary-style samples such as image-label pairs. Depending on
    ``input_layout``, it supports:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

    The transform is invertible because applying the same flip twice restores
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
        Flip a 3D image-label pair along the depth axis:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Flip

            transform = Flip(keys=["image", "label"], spatial_axis=0, input_layout="DHWC")

            image = tf.random.normal((32, 64, 64, 1))
            label = tf.random.uniform(
                (32, 64, 64, 1), maxval=2, dtype=tf.int32
            )

            result = transform({"image": image, "label": label})

        Restore a flipped tensor with ``inverse()``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Flip, TensorBundle

            transform = Flip(keys=["image"], spatial_axis=1, input_layout="HWC")
            image = tf.random.normal((64, 64, 1))

            forward = transform({"image": image})
            restored = transform.inverse(forward)

            print(forward["image"].shape)
            print(restored["image"].shape)

        Flip a 2D image stored in a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Flip, TensorBundle

            transform = Flip(keys=["image"], spatial_axis=0, input_layout="HWC")
            image = tf.random.normal((64, 64, 1))

            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

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
            if self.spatial_axis is None:
                return bundle
            params = {
                "spatial_axis": self.spatial_axis,
                "input_layout": self.input_layout,
            }
            self.apply_to_present_keys(
                bundle,
                lambda tensor, key: self.transform_tensor(tensor, key, params),
            )
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
        tensor: tf.Tensor,
        key: str,
        params: dict[str, object],
    ) -> tf.Tensor:
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
        tensor: tf.Tensor,
        spatial_axis: Union[int, Sequence[int], None] = None,
    ) -> tf.Tensor:
        """Flip one tensor using explicit spatial axes.

        Args:
            tensor: Channel-last 2D or 3D sample tensor to reverse.
            spatial_axis: Spatial axis or axes to flip. When omitted, the
                transform's configured ``spatial_axis`` is used. When the
                effective value is ``None``, the input tensor is returned
                unchanged.

        Returns:
            ``tf.Tensor``: The flipped tensor.
        """
        effective_axis = self.spatial_axis if spatial_axis is None else spatial_axis
        if effective_axis is None:
            return tensor
        if not self.layout_info.batched:
            return tf.reverse(tensor, axis=self._resolve_axes(tensor, spatial_axis=effective_axis))
        batched_tensor, added_batch_axis = ensure_batch_axis_for_layout(
            tensor,
            input_layout=self.input_layout,
        )
        flipped = self.flip_batch_tensor(batched_tensor, spatial_axis=effective_axis)
        return restore_from_batch_axis(flipped, added_batch_axis)

    def flip_batch_tensor(
        self,
        tensor: tf.Tensor,
        spatial_axis: Union[int, Sequence[int], None] = None,
    ) -> tf.Tensor:
        """Flip a batched tensor using spatial axes expressed without the batch axis."""
        effective_axis = self.spatial_axis if spatial_axis is None else spatial_axis
        if effective_axis is None:
            return tensor
        if not self.layout_info.batched:
            if isinstance(effective_axis, int):
                effective_axis = effective_axis + 1
            else:
                effective_axis = tuple(axis + 1 for axis in effective_axis)
        return tf.reverse(
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
        tensor: tf.Tensor,
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

from __future__ import annotations

from typing import Any, Optional, Sequence

from keras import ops

from ..base import InvertibleTransform, KeyedTransform, _pop_last_transform_trace
from ..tensor_bundle import TensorBundle
from ..utils import (
    resolve_input_layout,
    validate_tensor_matches_layout,
)


class ScaleIntensityRange(KeyedTransform, InvertibleTransform):
    """Linearly map selected tensor intensities from one numeric range to another.

    ``ScaleIntensityRange`` applies an affine intensity transform using the
    source interval ``source_value_range=(min, max)`` and, when provided, the
    target interval ``target_value_range=(min, max)``. This is useful for
    bringing image intensities into a stable range such as ``[0, 1]`` or
    ``[-1, 1]`` before training.

    Args:
        keys: Keys of the tensors to scale.
        source_value_range: Two-element source intensity range ``(min, max)``.
        target_value_range: Optional two-element target intensity range
            ``(min, max)``. If omitted, the normalized ``[0, 1]`` result is kept.
        clip: If ``True`` and ``target_value_range`` is provided, clip the
            output to the target interval after scaling.
        dtype: Output dtype used for computation and returned tensors.
        input_layout: Channel-last tensor layout. Supported values are
            ``"HWC"``, ``"DHWC"``, ``"BHWC"``, and ``"BDHWC"``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:

        TensorFlow backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import ScaleIntensityRange

            transform = ScaleIntensityRange(
                keys=["image"],
                source_value_range=(0.0, 255.0),
                target_value_range=(-1.0, 1.0),
                clip=True,
                input_layout="HWC",
            )

            image = tf.random.uniform((64, 64, 1), minval=0.0, maxval=255.0)
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        JAX backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            import jax.numpy as jnp
            from medicai.transforms import ScaleIntensityRange

            transform = ScaleIntensityRange(
                keys=["image"],
                source_value_range=(-175.0, 250.0),
                target_value_range=(0.0, 1.0),
                clip=True,
                input_layout="DHWC",
            )

            image = jax.random.uniform(
                jax.random.PRNGKey(7),
                shape=(32, 64, 64, 1),
                minval=-175.0,
                maxval=250.0,
                dtype=jnp.float32,
            )
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Torch backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import ScaleIntensityRange

            transform = ScaleIntensityRange(
                keys=["image"],
                source_value_range=(0.0, 255.0),
                target_value_range=(-1.0, 1.0),
                clip=True,
                input_layout="BHWC",
            )

            torch.manual_seed(7)
            batch = torch.rand((2, 64, 64, 1)) * 255.0
            result = transform({"image": batch})
            output = result["image"]
            print(output.shape)

    ``ScaleIntensityRange`` is invertible only for pure affine range mappings.
    In practice that means:

    - ``clip=False``
    - ``source_value_range[0] != source_value_range[1]``
    - when a target range is provided, ``target_value_range[0] != target_value_range[1]``

    If clipping is enabled, or the mapping collapses values to a constant,
    exact inversion is not possible and :meth:`inverse` behaves as a no-op.
    When inversion is available, it replays only the keys recorded in the
    transform trace, restores using the recorded range parameters, and still
    honors ``allow_missing_keys``.

    Returns:
        ``TensorBundle``: The input bundle with selected tensors scaled in
        place and a trace entry appended.

    Raises:
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
    """

    def __init__(
        self,
        keys: Sequence[str],
        source_value_range: Sequence[float],
        target_value_range: Optional[Sequence[float]] = None,
        clip: bool = False,
        dtype: Any = "float32",
        *,
        input_layout: str,
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.source_value_range = self._validate_value_range(
            source_value_range, "source_value_range"
        )
        self.target_value_range = (
            None
            if target_value_range is None
            else self._validate_value_range(target_value_range, "target_value_range")
        )
        self.clip = clip
        self.dtype = dtype
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            transform_name=type(self).__name__,
        )

    @property
    def invertible(self) -> bool:
        if self.clip:
            return False
        if self.source_value_range[1] == self.source_value_range[0]:
            return False
        if self.target_value_range is not None:
            return self.target_value_range[1] != self.target_value_range[0]
        return True

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        params = self.get_transform_params(bundle)
        present_keys = self.apply_to_present_keys(
            bundle, lambda tensor, _: self.transform_tensor(tensor, params)
        )
        self.record_transform(bundle, self.build_trace_params(params, present_keys))
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        if not self.invertible:
            return bundle

        trace = self._get_last_scaling_trace(bundle)
        if trace is None:
            return bundle

        params = trace["params"]
        source_value_range = params.get("source_value_range", self.source_value_range)
        target_value_range = params.get("target_value_range", self.target_value_range)

        self.apply_to_present_keys(
            bundle,
            lambda tensor, _: self.inverse_scale_tensor(
                tensor,
                source_value_range=source_value_range,
                target_value_range=target_value_range,
            ),
            keys=params.get("keys", []),
        )
        return bundle

    def get_transform_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Prepare forward-pass parameters for this range scaling."""
        del bundle
        return {
            "source_value_range": self.source_value_range,
            "target_value_range": self.target_value_range,
            "clip": self.clip,
            "input_layout": self.input_layout,
        }

    def transform_tensor(
        self,
        tensor: Any,
        params: dict[str, object],
    ) -> Any:
        """Scale one tensor using prepared transform parameters."""
        self._validate_tensor_layout(tensor)
        return self.scale_tensor(
            tensor,
            source_value_range=params["source_value_range"],
            target_value_range=params["target_value_range"],
            clip=params["clip"],
        )

    def build_trace_params(
        self,
        params: dict[str, object],
        present_keys: Sequence[str],
    ) -> dict[str, object]:
        """Build invertible trace metadata for the current range scaling."""
        return {
            "keys": list(present_keys),
            "source_value_range": params["source_value_range"],
            "target_value_range": params["target_value_range"],
            "clip": params["clip"],
            "input_layout": params["input_layout"],
        }

    def scale_tensor(
        self,
        tensor: Any,
        source_value_range: Sequence[float] | None = None,
        target_value_range: Sequence[float] | None = None,
        clip: bool | None = None,
    ) -> Any:
        """Scale one sample or batch tensor from source range to target range."""
        tensor = ops.cast(ops.convert_to_tensor(tensor), self.dtype)
        return self.scale_batch_tensor(
            tensor,
            source_value_range=source_value_range,
            target_value_range=target_value_range,
            clip=clip,
        )

    def scale_batch_tensor(
        self,
        tensor: Any,
        source_value_range: Sequence[float] | None = None,
        target_value_range: Sequence[float] | None = None,
        clip: bool | None = None,
    ) -> Any:
        """Scale a tensor with a kernel that is agnostic to sample vs batch layout."""
        src_range = (
            self.source_value_range if source_value_range is None else tuple(source_value_range)
        )
        tgt_range = (
            self.target_value_range if target_value_range is None else tuple(target_value_range)
        )
        in_min, in_max = src_range
        if tgt_range is None:
            out_min = out_max = None
        else:
            out_min, out_max = tgt_range
        should_clip = self.clip if clip is None else clip

        if in_max == in_min:
            result = tensor - in_min if out_min is None else tensor - in_min + out_min
            return ops.cast(result, dtype=self.dtype)

        tensor = (tensor - in_min) / (in_max - in_min)
        if out_min is not None and out_max is not None:
            tensor = tensor * (out_max - out_min) + out_min
        if should_clip and out_min is not None and out_max is not None:
            tensor = ops.clip(tensor, out_min, out_max)
        return ops.cast(tensor, dtype=self.dtype)

    def inverse_scale_tensor(
        self,
        tensor: Any,
        source_value_range: Sequence[float] | None = None,
        target_value_range: Sequence[float] | None = None,
    ) -> Any:
        """Invert one sample or batch tensor from target range back to source range."""
        tensor = ops.cast(ops.convert_to_tensor(tensor), self.dtype)

        src_range = (
            self.source_value_range if source_value_range is None else tuple(source_value_range)
        )
        tgt_range = (
            self.target_value_range if target_value_range is None else tuple(target_value_range)
        )
        in_min, in_max = src_range
        if tgt_range is None:
            out_min = out_max = None
        else:
            out_min, out_max = tgt_range

        if out_min is not None and out_max is not None:
            tensor = (tensor - out_min) / (out_max - out_min)

        tensor = tensor * (in_max - in_min) + in_min
        return ops.cast(tensor, dtype=self.dtype)

    def _validate_tensor_layout(self, tensor: Any) -> None:
        """Validate sample or batch channel-last layout for intensity scaling."""
        validate_tensor_matches_layout(
            tensor,
            self.input_layout,
            transform_name=type(self).__name__,
        )

    def _validate_value_range(
        self,
        value_range: Sequence[float],
        name: str,
    ) -> tuple[float, float]:
        """Validate a two-element intensity range."""
        if len(value_range) != 2:
            raise ValueError(f"`{name}` must contain exactly 2 values, got {len(value_range)}.")
        lower, upper = value_range
        if lower > upper:
            raise ValueError(f"`{name}` must be ordered as (min, max), got ({lower}, {upper}).")
        return (float(lower), float(upper))

    def _get_last_scaling_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)

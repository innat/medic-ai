from __future__ import annotations

from typing import Optional, Sequence

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform, _pop_last_transform_trace
from ..tensor_bundle import TensorBundle
from ..utils import (
    get_legacy_layout_components,
    resolve_input_layout,
    validate_tensor_matches_layout,
)


class ScaleIntensityRange(KeyedTransform, InvertibleTransform):
    """Linearly map selected tensor intensities from one numeric range to another.

    ``ScaleIntensityRange`` applies an affine intensity transform using the
    source interval ``[input_min, input_max]`` and, when provided, the target interval
    ``[output_min, output_max]``. This is useful for bringing image intensities into a
    stable range such as ``[0, 1]`` or ``[-1, 1]`` before training.

    Depending on ``input_layout``, this transform supports:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

    It does not infer source ranges from the data; callers must provide
    medically meaningful source bounds. Integer inputs such as ``uint8`` are
    supported and are cast to ``dtype`` before scaling.

    Args:
        keys: Keys of the tensors to scale.
        input_min: Lower bound of the source intensity range.
        input_max: Upper bound of the source intensity range.
        output_min: Lower bound of the target range. If ``None`` together with
            ``output_max=None``, the normalized ``[0, 1]`` result is kept.
        output_max: Upper bound of the target range.
        clip: If ``True`` and both ``output_min`` and ``output_max`` are provided, clip
            the output to the target interval after scaling.
        dtype: Output dtype used for computation and returned tensors.
        input_layout: Channel-last tensor layout. Supported values are
            ``"HWC"``, ``"DHWC"``, ``"BHWC"``, and ``"BDHWC"``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Map a 2D image into ``[-1, 1]`` using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import ScaleIntensityRange

            transform = ScaleIntensityRange(
                keys=["image"],
                input_min=0.0,
                input_max=255.0,
                output_min=-1.0,
                output_max=1.0,
                clip=True,
                input_layout="HWC",
            )

            image = tf.random.uniform((64, 64, 1), minval=0.0, maxval=255.0)
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Map a 3D image volume from a clipped CT range into ``[0, 1]`` using a
        ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import ScaleIntensityRange, TensorBundle

            transform = ScaleIntensityRange(
                keys=["image"],
                input_min=-175.0,
                input_max=250.0,
                output_min=0.0,
                output_max=1.0,
                clip=True,
                input_layout="DHWC",
            )

            image = tf.random.normal((32, 64, 64, 1))
            bundle = TensorBundle({"image": image})
            result = transform(bundle)
            output = result["image"]
            print(output.shape)

    ``ScaleIntensityRange`` is invertible only for pure affine range mappings.
    In practice that means:

    - `clip=False`
    - `input_min != input_max`
    - when a target range is provided, `output_min != output_max`

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
        input_min: float,
        input_max: float,
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
        clip: bool = False,
        dtype: tf.DType = tf.float32,
        *,
        input_layout: str,
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self._validate_output_range_args(output_min, output_max)
        self.input_min = input_min
        self.input_max = input_max
        self.output_min = output_min
        self.output_max = output_max
        self.clip = clip
        self.dtype = dtype
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            transform_name=type(self).__name__,
        )
        self.input_mode, self.spatial_dims = get_legacy_layout_components(self.input_layout)

    @property
    def invertible(self) -> bool:
        if self.clip:
            return False
        if self.input_max == self.input_min:
            return False
        if self.output_min is not None and self.output_max is not None:
            return self.output_max != self.output_min
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
        input_min = params.get("input_min", self.input_min)
        input_max = params.get("input_max", self.input_max)
        output_min = params.get("output_min", self.output_min)
        output_max = params.get("output_max", self.output_max)

        self.apply_to_present_keys(
            bundle,
            lambda tensor, _: self.inverse_scale_tensor(
                tensor,
                input_min=input_min,
                input_max=input_max,
                output_min=output_min,
                output_max=output_max,
            ),
            keys=params.get("keys", []),
        )
        return bundle

    def get_transform_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Prepare forward-pass parameters for this range scaling."""
        del bundle
        return {
            "input_min": self.input_min,
            "input_max": self.input_max,
            "output_min": self.output_min,
            "output_max": self.output_max,
            "clip": self.clip,
            "input_layout": self.input_layout,
            "input_mode": self.input_mode,
            "spatial_dims": self.spatial_dims,
        }

    def transform_tensor(
        self,
        tensor: tf.Tensor,
        params: dict[str, object],
    ) -> tf.Tensor:
        """Scale one tensor using prepared transform parameters."""
        self._validate_tensor_layout(tensor)
        return self.scale_tensor(
            tensor,
            input_min=params["input_min"],
            input_max=params["input_max"],
            output_min=params["output_min"],
            output_max=params["output_max"],
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
            "input_min": params["input_min"],
            "input_max": params["input_max"],
            "output_min": params["output_min"],
            "output_max": params["output_max"],
            "clip": params["clip"],
            "input_layout": params["input_layout"],
            "input_mode": params["input_mode"],
            "spatial_dims": params["spatial_dims"],
        }

    def scale_tensor(
        self,
        tensor: tf.Tensor,
        input_min: float | None = None,
        input_max: float | None = None,
        output_min: float | None = None,
        output_max: float | None = None,
        clip: bool | None = None,
    ) -> tf.Tensor:
        """Scale one sample or batch tensor from source range to target range."""
        tensor = tf.cast(tf.convert_to_tensor(tensor), self.dtype)
        return self.scale_batch_tensor(
            tensor,
            input_min=input_min,
            input_max=input_max,
            output_min=output_min,
            output_max=output_max,
            clip=clip,
        )

    def scale_batch_tensor(
        self,
        tensor: tf.Tensor,
        input_min: float | None = None,
        input_max: float | None = None,
        output_min: float | None = None,
        output_max: float | None = None,
        clip: bool | None = None,
    ) -> tf.Tensor:
        """Scale a tensor with a kernel that is agnostic to sample vs batch layout."""
        in_min = self.input_min if input_min is None else input_min
        in_max = self.input_max if input_max is None else input_max
        out_min = self.output_min if output_min is None else output_min
        out_max = self.output_max if output_max is None else output_max
        should_clip = self.clip if clip is None else clip

        if in_max == in_min:
            result = (
                tensor - in_min
                if out_min is None
                else tensor - in_min + out_min
            )
            return tf.cast(result, dtype=self.dtype)

        tensor = (tensor - in_min) / (in_max - in_min)
        if out_min is not None and out_max is not None:
            tensor = tensor * (out_max - out_min) + out_min
        if should_clip and out_min is not None and out_max is not None:
            tensor = tf.clip_by_value(tensor, out_min, out_max)
        return tf.cast(tensor, dtype=self.dtype)

    def inverse_scale_tensor(
        self,
        tensor: tf.Tensor,
        input_min: float | None = None,
        input_max: float | None = None,
        output_min: float | None = None,
        output_max: float | None = None,
    ) -> tf.Tensor:
        """Invert one sample or batch tensor from target range back to source range."""
        tensor = tf.cast(tf.convert_to_tensor(tensor), self.dtype)

        in_min = self.input_min if input_min is None else input_min
        in_max = self.input_max if input_max is None else input_max
        out_min = self.output_min if output_min is None else output_min
        out_max = self.output_max if output_max is None else output_max

        if out_min is not None and out_max is not None:
            tensor = (tensor - out_min) / (out_max - out_min)

        tensor = tensor * (in_max - in_min) + in_min
        return tf.cast(tensor, dtype=self.dtype)

    def _validate_tensor_layout(self, tensor: tf.Tensor) -> None:
        """Validate sample or batch channel-last layout for intensity scaling."""
        validate_tensor_matches_layout(
            tensor,
            self.input_layout,
            transform_name=type(self).__name__,
        )

    def _validate_output_range_args(
        self,
        output_min: Optional[float],
        output_max: Optional[float],
    ) -> None:
        """Validate target range argument pairing."""
        if (output_min is None) != (output_max is None):
            raise ValueError(
                "`output_min` and `output_max` must be provided together or both omitted."
            )

    def _get_last_scaling_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)

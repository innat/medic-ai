from __future__ import annotations

from typing import Optional, Sequence

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform
from ..tensor_bundle import TensorBundle


class ScaleIntensityRange(KeyedTransform, InvertibleTransform):
    """Linearly map selected tensor intensities from one numeric range to another.

    ``ScaleIntensityRange`` applies an affine intensity transform using the
    source interval ``[input_min, input_max]`` and, when provided, the target interval
    ``[output_min, output_max]``. This is useful for bringing image intensities into a
    stable range such as ``[0, 1]`` or ``[-1, 1]`` before training.

    The transform expects channel-last image tensors such as ``(H, W, C)`` or
    ``(D, H, W, C)``. It does not infer source ranges from the data; callers
    must provide medically meaningful source bounds.

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
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        if (output_min is None) != (output_max is None):
            raise ValueError(
                "`output_min` and `output_max` must be provided together or both omitted."
            )
        self.input_min = input_min
        self.input_max = input_max
        self.output_min = output_min
        self.output_max = output_max
        self.clip = clip
        self.dtype = dtype

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
        present_keys = self.apply_to_present_keys(
            bundle, lambda tensor, _: self.scale_tensor(tensor)
        )
        self.record_transform(
            bundle,
            {
                "keys": list(present_keys),
                "input_min": self.input_min,
                "input_max": self.input_max,
                "output_min": self.output_min,
                "output_max": self.output_max,
                "clip": self.clip,
            },
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        if not self.invertible:
            return bundle

        trace = self._get_last_scaling_trace(bundle)
        if trace is None:
            return bundle

        present_keys = [key for key in trace["params"].get("keys", []) if key in bundle.data]
        self.apply_to_present_keys(
            bundle, lambda tensor, _: self.inverse_scale_tensor(tensor), keys=present_keys
        )
        return bundle

    def scale_tensor(self, tensor: tf.Tensor) -> tf.Tensor:
        """Scale one tensor from source range to target range."""
        tensor = tf.convert_to_tensor(tensor, dtype=self.dtype)
        if self.input_max == self.input_min:
            result = (
                tensor - self.input_min
                if self.output_min is None
                else tensor - self.input_min + self.output_min
            )
            return tf.cast(result, dtype=self.dtype)

        tensor = (tensor - self.input_min) / (self.input_max - self.input_min)
        if self.output_min is not None and self.output_max is not None:
            tensor = tensor * (self.output_max - self.output_min) + self.output_min
        if self.clip and self.output_min is not None and self.output_max is not None:
            tensor = tf.clip_by_value(tensor, self.output_min, self.output_max)
        return tf.cast(tensor, dtype=self.dtype)

    def inverse_scale_tensor(self, tensor: tf.Tensor) -> tf.Tensor:
        """Invert one tensor from target range back to source range."""
        tensor = tf.convert_to_tensor(tensor, dtype=self.dtype)

        if self.output_min is not None and self.output_max is not None:
            tensor = (tensor - self.output_min) / (self.output_max - self.output_min)

        tensor = tensor * (self.input_max - self.input_min) + self.input_min
        return tf.cast(tensor, dtype=self.dtype)

    def _get_last_scaling_trace(self, bundle: TensorBundle):
        for entry in reversed(bundle.get_applied_transforms()):
            if entry.get("name") == type(self).__name__:
                return entry
        return None

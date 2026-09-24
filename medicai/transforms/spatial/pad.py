from __future__ import annotations

from numbers import Number
from typing import Any, Mapping, Sequence

from keras import ops

from ..base import InvertibleTransform, KeyedTransform, _pop_last_transform_trace
from ..tensor_bundle import TensorBundle
from ..utils import (
    get_input_layout_info,
    get_spatial_shape_for_layout,
    resolve_input_layout,
    validate_tensor_matches_layout,
)

_SUPPORTED_FILL_MODES = {"constant", "reflect", "symmetric"}


class Pad(KeyedTransform, InvertibleTransform):
    """Deterministically pad channel-last image tensors.

    Padding is applied only to spatial axes. The public spatial order is
    ``(H, W)`` for 2D and ``(D, H, W)`` for 3D. Each axis uses a
    ``(front, back)`` pair, where front is the lower-index side of the tensor
    and back is the higher-index side. These terms do not imply anatomical
    orientation. Padding is shared across a batch, while ``fill_mode`` and
    ``fill_value`` may be configured independently for each key.

    This transform is intended for deterministic preprocessing rather than
    augmentation. It is useful before a crop or model whose input dimensions
    must satisfy a minimum size or divisibility requirement. Use
    :class:`PadIfNeeded` when padding should be computed from the input shape.

    Args:
        keys: Keys of tensors to pad.
        padding: A scalar, one symmetric value per spatial axis, or one
            ``(front, back)`` pair per spatial axis.
        fill_mode: Padding mode or a key-to-mode mapping. The portable modes
            are ``"constant"``, ``"reflect"``, and ``"symmetric"``.
        fill_value: Constant padding value or a key-to-value mapping. It is
            used only when the corresponding fill mode is ``"constant"``.
        input_layout: Channel-last tensor layout: ``HWC``, ``DHWC``, ``BHWC``,
            or ``BDHWC``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    The fill modes are ``"constant"``, ``"reflect"``, and
    ``"symmetric"``. For segmentation, use a background class such as ``0``
    for labels and choose an image background value appropriate for the image
    modality.

    ``inverse()`` removes exactly the recorded padding and restores each
    selected tensor's original spatial shape. It is also suitable for placing
    a padded model prediction back onto the original canvas.

    Example:

        TensorFlow backend:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Pad

            transform = Pad(
                keys=["image", "label"],
                padding=((4, 4), (8, 8)),  # top/bottom, left/right
                fill_value={"image": -1000.0, "label": 0},
                input_layout="HWC",
            )
            result = transform(
                {
                    "image": tf.zeros((64, 64, 1)),
                    "label": tf.zeros((64, 64, 1), dtype=tf.int32),
                }
            )

        JAX backend:

        .. code-block:: python

            import jax.numpy as jnp
            from medicai.transforms import Pad

            transform = Pad(
                keys=["image"],
                padding=((2, 2), (8, 8), (8, 8)),
                input_layout="DHWC",
            )
            result = transform({"image": jnp.zeros((32, 64, 64, 1))})

        Torch backend:

        .. code-block:: python

            import torch
            from medicai.transforms import Pad

            transform = Pad(
                keys=["image"],
                padding=((1, 1), (4, 4), (4, 4)),
                input_layout="BDHWC",
            )
            batch = torch.zeros((2, 16, 32, 32, 1))
            result = transform({"image": batch})

    Returns:
        ``TensorBundle``: The padded tensors with an invertible transform trace.

    Raises:
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
        ValueError: If the padding specification, input layout, or fill mode
            is invalid.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        padding: int | Sequence[int] | Sequence[Sequence[int]],
        *,
        fill_mode: str | Mapping[str, str] = "constant",
        fill_value: Any | Mapping[str, Any] = 0,
        input_layout: str,
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            transform_name=type(self).__name__,
        )
        self.layout_info = get_input_layout_info(self.input_layout)
        self.padding = self._normalize_padding(padding)
        self.fill_mode = self._normalize_per_key_option(fill_mode, "fill_mode")
        self.fill_value = self._normalize_per_key_option(fill_value, "fill_value")

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        original_shapes: dict[str, Any] = {}

        def apply_padding(tensor: Any, key: str) -> Any:
            validate_tensor_matches_layout(
                tensor,
                self.input_layout,
                transform_name=type(self).__name__,
            )
            original_shapes[key] = get_spatial_shape_for_layout(
                tensor,
                input_layout=self.input_layout,
            )
            return self._pad_tensor(tensor, key)

        present_keys = self.apply_to_present_keys(bundle, apply_padding)
        if present_keys:
            self.record_transform(
                bundle,
                {
                    "keys": list(present_keys),
                    "padding": self.padding,
                    "original_shapes": original_shapes,
                    "input_layout": self.input_layout,
                },
            )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = _pop_last_transform_trace(bundle, type(self).__name__)
        if trace is None:
            return bundle

        original_shapes = trace["params"].get("original_shapes", {})
        padding = trace["params"].get("padding", self.padding)

        def remove_padding(tensor: Any, key: str) -> Any:
            original_shape = original_shapes.get(key)
            if original_shape is None:
                return tensor
            return self._unpad_tensor(tensor, original_shape, padding)

        self.apply_to_present_keys(
            bundle,
            remove_padding,
            keys=trace["params"].get("keys", []),
        )
        return bundle

    def _pad_tensor(
        self,
        tensor: Any,
        key: str,
        padding: Any | None = None,
    ) -> Any:
        mode = self.fill_mode[key]
        paddings = self._full_paddings(self.padding if padding is None else padding)
        if mode == "constant":
            return ops.pad(
                tensor,
                paddings,
                mode=mode,
                constant_values=self.fill_value[key],
            )
        return ops.pad(tensor, paddings, mode=mode)

    def _unpad_tensor(self, tensor: Any, original_shape: Any, padding: Any) -> Any:
        spatial_padding = ops.convert_to_tensor(padding, dtype="int32")
        spatial_front = ops.take(spatial_padding, 0, axis=1)
        current_shape = ops.shape(tensor)

        if self.layout_info.batched:
            starts = ops.concatenate(
                [ops.zeros((1,), dtype="int32"), spatial_front, ops.zeros((1,), dtype="int32")],
                axis=0,
            )
            size = ops.concatenate(
                [
                    ops.reshape(current_shape[0], (1,)),
                    original_shape,
                    ops.reshape(current_shape[-1], (1,)),
                ],
                axis=0,
            )
        else:
            starts = ops.concatenate(
                [spatial_front, ops.zeros((1,), dtype="int32")],
                axis=0,
            )
            size = ops.concatenate(
                [original_shape, ops.reshape(current_shape[-1], (1,))],
                axis=0,
            )

        return ops.slice(tensor, start_indices=starts, shape=size)

    def _full_paddings(self, padding: Any) -> Any:
        spatial_padding = ops.convert_to_tensor(padding, dtype="int32")
        zero = ops.zeros((1, 2), dtype="int32")
        if self.layout_info.batched:
            return ops.concatenate([zero, spatial_padding, zero], axis=0)
        return ops.concatenate([spatial_padding, zero], axis=0)

    def _normalize_padding(self, padding: Any) -> tuple[tuple[int, int], ...]:
        rank = self.layout_info.spatial_rank
        if isinstance(padding, Number):
            values = [int(padding)] * rank
            pairs = tuple((value, value) for value in values)
        else:
            values = tuple(padding)
            if len(values) == rank and all(isinstance(value, Number) for value in values):
                pairs = tuple((int(value), int(value)) for value in values)
            elif len(values) == rank and all(
                isinstance(value, (tuple, list)) and len(value) == 2 for value in values
            ):
                pairs = tuple((int(value[0]), int(value[1])) for value in values)
            else:
                raise ValueError(
                    f"`padding` must be a scalar, {rank} axis values, or {rank} "
                    f"(front, back) pairs for {self.input_layout}."
                )

        if any(side < 0 for pair in pairs for side in pair):
            raise ValueError("`padding` values must be non-negative.")
        return pairs

    def _normalize_per_key_option(self, value: Any, name: str) -> dict[str, Any]:
        if isinstance(value, Mapping):
            missing = set(self.keys) - set(value)
            if missing:
                raise ValueError(f"`{name}` is missing values for keys: {sorted(missing)}")
            result = {key: value[key] for key in self.keys}
        else:
            result = {key: value for key in self.keys}

        if name == "fill_mode":
            invalid = {mode for mode in result.values() if mode not in _SUPPORTED_FILL_MODES}
            if invalid:
                supported = ", ".join(sorted(_SUPPORTED_FILL_MODES))
                raise ValueError(
                    f"`fill_mode` values must be one of {supported}; received {sorted(invalid)}."
                )
        return result


class PadIfNeeded(Pad):
    """Pad tensors to a minimum shape or divisibility constraint.

    Padding is deterministic and shared across selected keys. Required padding
    is split as evenly as possible between the front and back of each spatial
    axis; when the total is odd, the back receives the extra element. This is a
    preprocessing transform, not a random augmentation.

    Args:
        keys: Keys of tensors to pad.
        min_target_shape: Minimum spatial shape in ``(H, W)`` or ``(D, H, W)``
            order. The output is never smaller than this shape.
        divisible_by: Optional positive divisor per spatial axis. The output
            shape is padded to be divisible by these values.
        fill_mode: Padding mode or a key-to-mode mapping. The portable modes
            are ``"constant"``, ``"reflect"``, and ``"symmetric"``.
        fill_value: Constant padding value or a key-to-value mapping.
        input_layout: Channel-last tensor layout: ``HWC``, ``DHWC``, ``BHWC``,
            or ``BDHWC``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    If the input already satisfies all constraints, the tensors are unchanged
    apart from an invertible no-op trace entry. ``inverse()`` restores the
    original tensors using the recorded padding.

    Example:

        TensorFlow backend for a 3D segmentation sample:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import PadIfNeeded

            pad = PadIfNeeded(
                keys=["image", "label"],
                min_target_shape=(128, 128, 128),
                divisible_by=(16, 16, 16),
                fill_value={"image": 0.0, "label": 0},
                input_layout="DHWC",
            )
            image = tf.zeros((96, 112, 120, 1))
            label = tf.zeros((96, 112, 120, 1), dtype=tf.int32)
            result = pad({"image": image, "label": label})

        JAX backend for a 2D sample:

        .. code-block:: python

            import jax.numpy as jnp
            from medicai.transforms import PadIfNeeded

            pad = PadIfNeeded(
                keys=["image"],
                min_target_shape=(224, 224),
                divisible_by=32,
                input_layout="HWC",
            )
            image = jnp.zeros((197, 211, 3))
            result = pad({"image": image})

        Torch backend for a 2D batch:

        .. code-block:: python

            import torch
            from medicai.transforms import PadIfNeeded

            pad = PadIfNeeded(
                keys=["image"],
                min_target_shape=(224, 224),
                divisible_by=32,
                input_layout="BHWC",
            )
            batch = torch.zeros((4, 197, 211, 3))
            result = pad({"image": batch})

    Returns:
        ``TensorBundle``: The padded tensors with an invertible transform trace.

    Raises:
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
        ValueError: If neither target constraint is provided or if a target
            shape or divisor is invalid for the selected spatial rank.
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        min_target_shape: int | Sequence[int] | None = None,
        divisible_by: int | Sequence[int] | None = None,
        *,
        fill_mode: str | Mapping[str, str] = "constant",
        fill_value: Any | Mapping[str, Any] = 0,
        input_layout: str,
        allow_missing_keys: bool = False,
    ):
        if min_target_shape is None and divisible_by is None:
            raise ValueError("At least one of `min_target_shape` or `divisible_by` is required.")

        super().__init__(
            keys=keys,
            padding=0,
            fill_mode=fill_mode,
            fill_value=fill_value,
            input_layout=input_layout,
            allow_missing_keys=allow_missing_keys,
        )
        rank = self.layout_info.spatial_rank
        self.min_target_shape = self._normalize_target_shape(min_target_shape, rank)
        self.divisible_by = self._normalize_divisible_by(divisible_by, rank)

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        original_shapes: dict[str, Any] = {}
        padding: Any | None = None

        def apply_required_padding(tensor: Any, key: str) -> Any:
            nonlocal padding
            validate_tensor_matches_layout(
                tensor,
                self.input_layout,
                transform_name=type(self).__name__,
            )
            original_shapes[key] = get_spatial_shape_for_layout(
                tensor,
                input_layout=self.input_layout,
            )
            if padding is None:
                padding = self._compute_padding(tensor)
            return self._pad_tensor(tensor, key, padding)

        present_keys = self.apply_to_present_keys(bundle, apply_required_padding)
        if present_keys:
            self.record_transform(
                bundle,
                {
                    "keys": list(present_keys),
                    "padding": padding,
                    "original_shapes": original_shapes,
                    "input_layout": self.input_layout,
                },
            )
        return bundle

    def _compute_padding(self, tensor: Any) -> Any:
        spatial_shape = get_spatial_shape_for_layout(
            tensor,
            input_layout=self.input_layout,
        )
        static_shape = tuple(tensor.shape[axis] for axis in self.layout_info.spatial_axes)
        if all(dimension is not None for dimension in static_shape):
            current = [int(dimension) for dimension in static_shape]
            target = [
                max(value, minimum)
                for value, minimum in zip(current, self.min_target_shape or (0,) * len(current))
            ]
            if self.divisible_by is not None:
                target = [
                    ((value + divisor - 1) // divisor) * divisor
                    for value, divisor in zip(target, self.divisible_by)
                ]
            return tuple(
                (total // 2, total - total // 2)
                for total in (
                    target_value - current_value
                    for target_value, current_value in zip(target, current)
                )
            )

        target = spatial_shape
        if self.min_target_shape is not None:
            target = ops.maximum(
                target,
                ops.convert_to_tensor(self.min_target_shape, dtype="int32"),
            )
        if self.divisible_by is not None:
            divisors = ops.convert_to_tensor(self.divisible_by, dtype="int32")
            target = ((target + divisors - 1) // divisors) * divisors
        total = target - spatial_shape
        front = total // 2
        return ops.stack([front, total - front], axis=1)

    def _normalize_target_shape(
        self,
        value: int | Sequence[int] | None,
        rank: int,
    ) -> tuple[int, ...] | None:
        if value is None:
            return None
        if isinstance(value, Number):
            values = (int(value),) * rank
        else:
            values = tuple(int(item) for item in value)
        if len(values) != rank or any(item <= 0 for item in values):
            raise ValueError(
                f"`min_target_shape` must contain {rank} positive values for {self.input_layout}."
            )
        return values

    def _normalize_divisible_by(
        self,
        value: int | Sequence[int] | None,
        rank: int,
    ) -> tuple[int, ...] | None:
        if value is None:
            return None
        if isinstance(value, Number):
            values = (int(value),) * rank
        else:
            values = tuple(int(item) for item in value)
        if len(values) != rank or any(item <= 0 for item in values):
            raise ValueError(
                f"`divisible_by` must contain {rank} positive values for {self.input_layout}."
            )
        return values

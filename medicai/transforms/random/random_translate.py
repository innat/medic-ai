"""Backend-neutral random translations for channel-last medical tensors."""

from typing import Any, Sequence

import keras
from keras import ops

from ..base import RandomTransform, _pop_last_transform_trace
from .affine import sample_affine_volume
from ..tensor_bundle import TensorBundle
from ..utils import (
    ensure_batch_axis_for_layout,
    get_input_layout_info,
    resolve_input_layout,
    restore_from_batch_axis,
    validate_tensor_matches_layout,
)
_DEFAULT_PROB = 0.5
_DEFAULT_FACTOR = 0.0
_DEFAULT_INTERPOLATION = None
_DEFAULT_FILL_MODE = "constant"
_DEFAULT_FILL_VALUE = 0.0
_INTERPOLATION_MODES = {"bilinear", "nearest"}
_FILL_MODES = {"constant", "nearest", "reflect", "wrap", "mirror"}


def _factor_range(value: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Each translation factor range must contain two values.")
        low, high = float(value[0]), float(value[1])
    else:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"Translation factors must be non-negative. Received {value}.")
        low, high = -value, value
    if low > high:
        raise ValueError("Translation factor ranges must be ordered as (min, max).")
    return low, high


def _resolve_per_key(keys, value, default_fn, name):
    if value is None:
        return {key: default_fn(key, index) for index, key in enumerate(keys)}
    if isinstance(value, dict):
        missing = [key for key in keys if key not in value]
        if missing:
            raise ValueError(f"`{name}` is missing entries for keys: {missing}.")
        return {key: value[key] for key in keys}
    if isinstance(value, (tuple, list)):
        if len(value) != len(keys):
            raise ValueError(f"`{name}` must have one value per key.")
        return dict(zip(keys, value, strict=True))
    return {key: value for key in keys}


def _resolve_factor_ranges(factor, spatial_rank):
    """Normalize scalar or Cartesian-axis translation factors."""
    axes = ("x", "y") if spatial_rank == 2 else ("z", "x", "y")
    if isinstance(factor, dict):
        unknown = set(factor) - set(axes)
        if unknown:
            raise ValueError(
                f"Translation factor axes must be drawn from {axes}; "
                f"received {sorted(unknown)}."
            )
        return {
            axis: _factor_range(factor[axis]) if axis in factor else (0.0, 0.0)
            for axis in axes
        }

    value_range = _factor_range(factor)
    return {axis: value_range for axis in axes}


def _translation_matrix_2d(offsets: Any) -> Any:
    """Build Keras image-kernel matrices from ``(y, x)`` offsets."""
    zeros = ops.zeros_like(offsets[:, 0])
    ones = ops.ones_like(zeros)
    x = -offsets[:, 1]
    y = -offsets[:, 0]
    return ops.stack(
        [ones, zeros, x, zeros, ones, y, zeros, zeros],
        axis=-1,
    )


def _translation_matrix_3d(offsets: Any) -> Any:
    """Build output-to-input matrices from ``(z, y, x)`` offsets."""
    batch = ops.shape(offsets)[0]
    identity = ops.eye(3, dtype=offsets.dtype)
    identity = ops.broadcast_to(identity, (batch, 3, 3))
    inverse_offsets = -offsets
    return ops.concatenate(
        [identity, ops.expand_dims(inverse_offsets, axis=-1)], axis=-1
    )


def _translate_one_volume(volume, inverse_matrix, interpolation, fill_mode, fill_value):
    """Sample one volume with a 3D affine matrix containing translation."""
    return sample_affine_volume(
        volume,
        inverse_matrix[:, :3],
        interpolation,
        fill_mode,
        fill_value,
        translation=inverse_matrix[:, 3],
    )


class RandomTranslate(RandomTransform):
    """Randomly translate channel-last 2D images or 3D volumes.

    Factors are relative to the corresponding spatial dimension. A scalar
    factor samples symmetrically, so ``0.1`` permits offsets in ``[-0.1, 0.1]``
    of the dimension size. Parameters are sampled independently per batch item
    and shared across all selected keys.

    Public axis names are ``x`` and ``y`` for 2D, and ``z``, ``x``, and ``y``
    for 3D. They map to channel-last tensor axes ``W``, ``D``, and ``H``
    as ``x -> W``, ``y -> H``, and ``z -> D``; the internal sampler uses
    tensor order ``(D, H, W)``.

    Args:
        keys: Tensor keys to translate together.
        factor: A scalar or two-value relative translation range applied to all
            spatial axes, or a mapping with ``x``/``y`` entries for 2D and
            ``z``/``x``/``y`` entries for 3D. Missing mapping entries are
            treated as identity translations.
        prob: Per-sample probability of applying the translation.
        interpolation: One mode, one mode per key, or a key-to-mode mapping.
        fill_mode: Boundary behavior for newly exposed values.
        fill_value: Constant boundary value when ``fill_mode="constant"``.
        input_layout: One of ``HWC``, ``DHWC``, ``BHWC``, or ``BDHWC``.
        seed: Optional integer or Keras seed generator.
        allow_missing_keys: If ``True``, missing requested keys are skipped.
    """

    def __init__(
        self,
        keys: Sequence[str],
        factor: float | Sequence[float] | dict[str, Any] = _DEFAULT_FACTOR,
        prob: float = _DEFAULT_PROB,
        interpolation=_DEFAULT_INTERPOLATION,
        fill_mode=_DEFAULT_FILL_MODE,
        fill_value=_DEFAULT_FILL_VALUE,
        *,
        input_layout: str,
        seed: int | keras.random.SeedGenerator | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob, seed=seed)
        from ..base import _normalize_keys

        self.keys = _normalize_keys(keys)
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            allowed_layouts=("HWC", "DHWC", "BHWC", "BDHWC"),
            transform_name=type(self).__name__,
        )
        self.layout_info = get_input_layout_info(self.input_layout)
        self.allow_missing_keys = allow_missing_keys
        self.ranges = _resolve_factor_ranges(factor, self.layout_info.spatial_rank)

        self.interpolation = _resolve_per_key(
            self.keys,
            interpolation,
            lambda _, index: "bilinear" if index == 0 else "nearest",
            "interpolation",
        )
        self.fill_mode = _resolve_per_key(
            self.keys,
            fill_mode,
            lambda *_: _DEFAULT_FILL_MODE,
            "fill_mode",
        )
        self.fill_value = _resolve_per_key(
            self.keys,
            fill_value,
            lambda *_: _DEFAULT_FILL_VALUE,
            "fill_value",
        )
        for key in self.keys:
            mode = str(self.interpolation[key]).lower()
            if mode not in _INTERPOLATION_MODES:
                raise ValueError(f"Unsupported interpolation for key {key!r}.")
            boundary = str(self.fill_mode[key]).lower()
            if boundary not in _FILL_MODES:
                raise ValueError(f"Unsupported fill_mode {boundary!r}.")
            self.interpolation[key] = mode
            self.fill_mode[key] = boundary

    @property
    def invertible(self) -> bool:
        return True

    def _sample_offsets(self, batch_size, dtype="float32"):
        apply_mask = ops.cast(
            self.random_uniform(
                shape=(batch_size,), minval=0.0, maxval=1.0, dtype="float32"
            )
            < self.prob,
            dtype,
        )
        offsets = {}
        for axis, (low, high) in self.ranges.items():
            offsets[axis] = (
                self.random_uniform(
                    shape=(batch_size,), minval=low, maxval=high, dtype=dtype
                )
                * apply_mask
            )
        return offsets, ops.any(apply_mask > 0)

    def _offset_tensor(self, offsets, spatial_shape):
        if self.layout_info.spatial_rank == 2:
            return ops.stack([offsets["y"], offsets["x"]], axis=-1) * ops.cast(
                ops.convert_to_tensor(spatial_shape), offsets["y"].dtype
            )
        return ops.stack([offsets["z"], offsets["y"], offsets["x"]], axis=-1) * ops.cast(
            ops.convert_to_tensor(spatial_shape), offsets["z"].dtype
        )

    def _apply_tensor(self, tensor, key, offsets):
        batched, added_batch = ensure_batch_axis_for_layout(
            tensor, input_layout=self.input_layout, allowed_spatial_ranks=(2, 3)
        )
        validate_tensor_matches_layout(
            batched,
            "BHWC" if self.layout_info.spatial_rank == 2 else "BDHWC",
            transform_name=type(self).__name__,
        )
        spatial_shape = ops.shape(batched)[1:-1]
        offset_tensor = self._offset_tensor(offsets, spatial_shape)
        if self.layout_info.spatial_rank == 2:
            matrices = _translation_matrix_2d(offset_tensor)
            output = ops.image.affine_transform(
                ops.cast(batched, "float32"),
                matrices,
                interpolation=self.interpolation[key],
                fill_mode=self.fill_mode[key],
                fill_value=self.fill_value[key],
            )
        else:
            matrices = _translation_matrix_3d(offset_tensor)

            def translate_one(args):
                volume, matrix = args
                return _translate_one_volume(
                    volume,
                    matrix,
                    self.interpolation[key],
                    self.fill_mode[key],
                    self.fill_value[key],
                )

            output = ops.vectorized_map(translate_one, (batched, matrices))
        return restore_from_batch_axis(output, added_batch)

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present = []
        for key in self.keys:
            if key in bundle.data:
                present.append(key)
            elif not self.allow_missing_keys:
                raise KeyError(f"Key {key!r} not found in input data.")
        if not present:
            return bundle

        reference = bundle.data[present[0]]
        validate_tensor_matches_layout(
            reference,
            self.input_layout,
            transform_name=type(self).__name__,
        )
        batch_size = ops.shape(reference)[0] if self.layout_info.batched else 1
        offsets, applied = self._sample_offsets(batch_size)
        for key in present:
            bundle.data[key] = self._apply_tensor(bundle.data[key], key, offsets)
        self.record_random_transform(
            bundle,
            params={
                "keys": present,
                "offsets": offsets,
                "input_layout": self.input_layout,
            },
            applied=applied,
            kernel="random_translate",
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = _pop_last_transform_trace(bundle, type(self).__name__)
        if trace is None:
            return bundle
        offsets = {
            axis: -value for axis, value in trace["params"]["offsets"].items()
        }
        for key in trace["params"]["keys"]:
            if key in bundle.data:
                bundle.data[key] = self._apply_tensor(bundle.data[key], key, offsets)
        return bundle

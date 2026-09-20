"""Backend-neutral random zoom for channel-last medical tensors."""

from typing import Any, Sequence

import keras
from keras import ops

from ..base import RandomTransform, _normalize_keys, _pop_last_transform_trace
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
_DEFAULT_ZOOM_FACTOR = 0.0
_DEFAULT_INTERPOLATION = None
_DEFAULT_FILL_MODE = "constant"
_DEFAULT_FILL_VALUE = 0.0
_INTERPOLATION_MODES = {"bilinear", "nearest"}
_FILL_MODES = {"constant", "nearest", "reflect", "wrap", "mirror"}


def _factor_range(value: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Each zoom factor range must contain two values.")
        low, high = float(value[0]), float(value[1])
    else:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"Zoom factors must be non-negative. Received {value}.")
        low, high = -value, value
    if low > high:
        raise ValueError("Zoom factor ranges must be ordered as (min, max).")
    if low <= -1.0:
        raise ValueError("Zoom factors must keep the sampled scale above zero.")
    return low, high


def _resolve_axis_ranges(factor, spatial_rank):
    axes = ("x", "y") if spatial_rank == 2 else ("z", "x", "y")
    if isinstance(factor, dict):
        unknown = set(factor) - set(axes)
        if unknown:
            raise ValueError(
                f"Zoom factor axes must be drawn from {axes}; "
                f"received {sorted(unknown)}."
            )
        return {
            axis: _factor_range(factor[axis]) if axis in factor else (0.0, 0.0)
            for axis in axes
        }
    value_range = _factor_range(factor)
    return {axis: value_range for axis in axes}


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


def _zoom_matrix_2d(scales, spatial_shape):
    """Build output-to-input matrices for centered 2D zooms."""
    scale_y = 1.0 / scales[:, 0]
    scale_x = 1.0 / scales[:, 1]
    center = ops.cast(
        (ops.convert_to_tensor(spatial_shape, dtype="float32") - 1.0) / 2.0,
        scales.dtype,
    )
    translate_x = center[1] * (1.0 - scale_x)
    translate_y = center[0] * (1.0 - scale_y)
    return ops.stack(
        [
            scale_x,
            ops.zeros_like(scale_x),
            translate_x,
            ops.zeros_like(scale_y),
            scale_y,
            translate_y,
            ops.zeros_like(scale_x),
            ops.zeros_like(scale_x),
        ],
        axis=-1,
    )


def _zoom_one_volume(volume, scales, interpolation, fill_mode, fill_value):
    """Sample one 3D volume using a centered output-to-input zoom."""
    inverse_linear = ops.diag(1.0 / scales)
    return sample_affine_volume(
        volume, inverse_linear, interpolation, fill_mode, fill_value
    )


class RandomZoom(RandomTransform):
    """Randomly zoom channel-last 2D images or 3D volumes around their center.

    ``zoom_factor`` follows the Keras-style relative convention. A scalar
    ``0.2`` samples a factor in ``[-0.2, 0.2]`` independently for each active
    axis, then converts it to a scale of ``1 + factor``. A mapping can provide
    independent factors for Cartesian axes: ``x`` and ``y`` for 2D, or
    ``z``, ``x``, and ``y`` for 3D. Missing axes remain unchanged.

    Parameters are sampled independently for each batch item and shared across
    all selected keys, preserving image/label alignment.

    Args:
        keys: Tensor keys to zoom together.
        zoom_factor: A scalar or two-value factor range, or an axis mapping.
        prob: Per-sample probability of applying the zoom.
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
        zoom_factor: float | Sequence[float] | dict[str, Any] = _DEFAULT_ZOOM_FACTOR,
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
        self.keys = _normalize_keys(keys)
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            allowed_layouts=("HWC", "DHWC", "BHWC", "BDHWC"),
            transform_name=type(self).__name__,
        )
        self.layout_info = get_input_layout_info(self.input_layout)
        self.allow_missing_keys = allow_missing_keys
        self.ranges = _resolve_axis_ranges(
            zoom_factor, self.layout_info.spatial_rank
        )
        self.interpolation = _resolve_per_key(
            self.keys,
            interpolation,
            lambda _, index: "bilinear" if index == 0 else "nearest",
            "interpolation",
        )
        self.fill_mode = _resolve_per_key(
            self.keys, fill_mode, lambda *_: _DEFAULT_FILL_MODE, "fill_mode"
        )
        self.fill_value = _resolve_per_key(
            self.keys, fill_value, lambda *_: _DEFAULT_FILL_VALUE, "fill_value"
        )
        for key in self.keys:
            mode = str(self.interpolation[key]).lower()
            boundary = str(self.fill_mode[key]).lower()
            if mode not in _INTERPOLATION_MODES:
                raise ValueError(f"Unsupported interpolation for key {key!r}.")
            if boundary not in _FILL_MODES:
                raise ValueError(f"Unsupported fill_mode {boundary!r}.")
            self.interpolation[key] = mode
            self.fill_mode[key] = boundary

    @property
    def invertible(self) -> bool:
        return True

    def _sample_scales(self, batch_size, dtype="float32"):
        apply_mask = ops.cast(
            self.random_uniform(
                shape=(batch_size,), minval=0.0, maxval=1.0, dtype="float32"
            )
            < self.prob,
            dtype,
        )
        factors = {}
        for axis, (low, high) in self.ranges.items():
            sampled = self.random_uniform(
                shape=(batch_size,), minval=low, maxval=high, dtype=dtype
            )
            factors[axis] = sampled * apply_mask
        scales = {axis: 1.0 + value for axis, value in factors.items()}
        return scales, ops.any(apply_mask > 0)

    def _scale_tensor(self, tensor, key, scales):
        batched, added_batch = ensure_batch_axis_for_layout(
            tensor, input_layout=self.input_layout, allowed_spatial_ranks=(2, 3)
        )
        validate_tensor_matches_layout(
            batched,
            "BHWC" if self.layout_info.spatial_rank == 2 else "BDHWC",
            transform_name=type(self).__name__,
        )
        if self.layout_info.spatial_rank == 2:
            scale_tensor = ops.stack([scales["y"], scales["x"]], axis=-1)
            matrix = _zoom_matrix_2d(scale_tensor, ops.shape(batched)[1:-1])
            output = ops.image.affine_transform(
                ops.cast(batched, "float32"),
                matrix,
                interpolation=self.interpolation[key],
                fill_mode=self.fill_mode[key],
                fill_value=self.fill_value[key],
            )
        else:
            scale_tensor = ops.stack(
                [scales["z"], scales["y"], scales["x"]], axis=-1
            )

            def zoom_one(args):
                volume, sample_scales = args
                return _zoom_one_volume(
                    volume,
                    sample_scales,
                    self.interpolation[key],
                    self.fill_mode[key],
                    self.fill_value[key],
                )

            output = ops.vectorized_map(zoom_one, (batched, scale_tensor))
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
            reference, self.input_layout, transform_name=type(self).__name__
        )
        batch_size = ops.shape(reference)[0] if self.layout_info.batched else 1
        scales, applied = self._sample_scales(batch_size)
        for key in present:
            bundle.data[key] = self._scale_tensor(bundle.data[key], key, scales)
        self.record_random_transform(
            bundle,
            params={"keys": present, "scales": scales, "input_layout": self.input_layout},
            applied=applied,
            kernel="random_zoom",
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = _pop_last_transform_trace(bundle, type(self).__name__)
        if trace is None:
            return bundle
        inverse_scales = {
            axis: 1.0 / value for axis, value in trace["params"]["scales"].items()
        }
        for key in trace["params"]["keys"]:
            if key in bundle.data:
                bundle.data[key] = self._scale_tensor(
                    bundle.data[key], key, inverse_scales
                )
        return bundle

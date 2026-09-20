"""Backend-neutral random shear for channel-last medical tensors."""

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
_DEFAULT_SHEAR_FACTOR = 0.0
_DEFAULT_INTERPOLATION = None
_DEFAULT_FILL_MODE = "constant"
_DEFAULT_FILL_VALUE = 0.0
_INTERPOLATION_MODES = {"bilinear", "nearest"}
_FILL_MODES = {"constant", "nearest", "reflect", "wrap", "mirror"}
_AXES_2D = ("xy", "yx")
_AXES_3D = ("zy", "zx", "yz", "yx", "xz", "xy")


def _factor_range(value: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Each shear factor range must contain two values.")
        low, high = float(value[0]), float(value[1])
    else:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"Shear factors must be non-negative. Received {value}.")
        low, high = -value, value
    if low > high:
        raise ValueError("Shear factor ranges must be ordered as (min, max).")
    return low, high


def _resolve_shear_ranges(factor, spatial_rank):
    axes = _AXES_2D if spatial_rank == 2 else _AXES_3D
    if isinstance(factor, dict):
        unknown = set(factor) - set(axes)
        if unknown:
            raise ValueError(
                f"Shear factor axes must be drawn from {axes}; "
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


def _inverse_linear_2d(coefficients):
    """Return centered output-to-input matrices for 2D shear coefficients."""
    xy = coefficients[:, 0]
    yx = coefficients[:, 1]
    linear = ops.stack(
        [
            ops.ones_like(xy),
            yx,
            xy,
            ops.ones_like(xy),
        ],
        axis=-1,
    )
    return ops.linalg.inv(ops.reshape(linear, (-1, 2, 2)))


def _shear_matrix_2d(coefficients, spatial_shape, inverse=False):
    inverse_linear = _inverse_linear_2d(coefficients)
    center = ops.cast(
        (ops.convert_to_tensor(spatial_shape, dtype="float32") - 1.0) / 2.0,
        coefficients.dtype,
    )
    translation = center - ops.einsum("bij,j->bi", inverse_linear, center)
    matrix = ops.stack(
        [
            inverse_linear[:, 1, 1],
            inverse_linear[:, 1, 0],
            translation[:, 1],
            inverse_linear[:, 0, 1],
            inverse_linear[:, 0, 0],
            translation[:, 0],
            ops.zeros_like(translation[:, 0]),
            ops.zeros_like(translation[:, 0]),
        ],
        axis=-1,
    )
    if inverse:
        matrix = ops.linalg.inv(ops.reshape(matrix, (-1, 3, 3)))
        matrix = ops.reshape(matrix, (-1, 9))
    return matrix


def _inverse_linear_3d(coefficients, inverse=False):
    """Return output-to-input linear matrices for 3D shear coefficients."""
    zy, zx, yz, yx, xz, xy = [coefficients[:, index] for index in range(6)]
    linear = ops.stack(
        [
            ops.ones_like(zy),
            zy,
            zx,
            yz,
            ops.ones_like(zy),
            yx,
            xz,
            xy,
            ops.ones_like(zy),
        ],
        axis=-1,
    )
    matrix = ops.linalg.inv(ops.reshape(linear, (-1, 3, 3)))
    return ops.linalg.inv(matrix) if inverse else matrix


def _shear_one_volume(volume, inverse_linear, interpolation, fill_mode, fill_value):
    return sample_affine_volume(
        volume, inverse_linear, interpolation, fill_mode, fill_value
    )


class RandomShear(RandomTransform):
    """Randomly shear channel-last 2D images or 3D volumes.

    ``shear_factor`` is a dimensionless matrix coefficient. A scalar samples
    every supported axis-pair coefficient symmetrically. Mappings can select
    individual terms: ``xy`` and ``yx`` are supported for 2D; ``zy``, ``zx``,
    ``yz``, ``yx``, ``xz``, and ``xy`` are supported for 3D. The first letter
    identifies the output row and the second identifies the input column.

    Args:
        keys: Tensor keys to shear together.
        shear_factor: A scalar, two-value range, or axis-pair mapping.
        prob: Per-sample probability of applying the shear.
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
        shear_factor: float | Sequence[float] | dict[str, Any] = _DEFAULT_SHEAR_FACTOR,
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
        self.ranges = _resolve_shear_ranges(
            shear_factor, self.layout_info.spatial_rank
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

    def _sample_coefficients(self, batch_size, dtype="float32"):
        apply_mask = ops.cast(
            self.random_uniform(
                shape=(batch_size,), minval=0.0, maxval=1.0, dtype="float32"
            )
            < self.prob,
            dtype,
        )
        coefficients = {}
        for axis_pair, (low, high) in self.ranges.items():
            coefficients[axis_pair] = (
                self.random_uniform(
                    shape=(batch_size,), minval=low, maxval=high, dtype=dtype
                )
                * apply_mask
            )
        return coefficients, ops.any(apply_mask > 0)

    def _apply_tensor(self, tensor, key, coefficients, inverse=False):
        batched, added_batch = ensure_batch_axis_for_layout(
            tensor, input_layout=self.input_layout, allowed_spatial_ranks=(2, 3)
        )
        validate_tensor_matches_layout(
            batched,
            "BHWC" if self.layout_info.spatial_rank == 2 else "BDHWC",
            transform_name=type(self).__name__,
        )
        if self.layout_info.spatial_rank == 2:
            values = ops.stack([coefficients["xy"], coefficients["yx"]], axis=-1)
            matrix = _shear_matrix_2d(
                values, ops.shape(batched)[1:-1], inverse=inverse
            )
            output = ops.image.affine_transform(
                ops.cast(batched, "float32"),
                matrix,
                interpolation=self.interpolation[key],
                fill_mode=self.fill_mode[key],
                fill_value=self.fill_value[key],
            )
        else:
            values = ops.stack(
                [coefficients[axis] for axis in _AXES_3D], axis=-1
            )
            matrices = _inverse_linear_3d(values, inverse=inverse)

            def shear_one(args):
                volume, matrix = args
                return _shear_one_volume(
                    volume,
                    matrix,
                    self.interpolation[key],
                    self.fill_mode[key],
                    self.fill_value[key],
                )

            output = ops.vectorized_map(shear_one, (batched, matrices))
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
        coefficients, applied = self._sample_coefficients(batch_size)
        for key in present:
            bundle.data[key] = self._apply_tensor(bundle.data[key], key, coefficients)
        self.record_random_transform(
            bundle,
            params={
                "keys": present,
                "coefficients": coefficients,
                "input_layout": self.input_layout,
            },
            applied=applied,
            kernel="random_shear",
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = _pop_last_transform_trace(bundle, type(self).__name__)
        if trace is None:
            return bundle
        for key in trace["params"]["keys"]:
            if key in bundle.data:
                bundle.data[key] = self._apply_tensor(
                    bundle.data[key],
                    key,
                    trace["params"]["coefficients"],
                    inverse=True,
                )
        return bundle

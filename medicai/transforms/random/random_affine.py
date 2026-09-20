"""Single-pass random affine transforms for channel-last medical tensors."""

from typing import Any, Sequence

import keras
from keras import ops

from ..base import RandomTransform, _normalize_keys, _pop_last_transform_trace
from ..tensor_bundle import TensorBundle
from ..utils import (
    ensure_batch_axis_for_layout,
    get_input_layout_info,
    resolve_input_layout,
    restore_from_batch_axis,
    validate_tensor_matches_layout,
)
from .affine import (
    centered_affine_matrix,
    compose_affine_matrices,
    invert_affine_matrix,
    sample_affine_volume,
)

_INTERPOLATION_MODES = {
    2: {"bilinear", "nearest"},
    3: {"trilinear", "nearest"},
}
_FILL_MODES = {"constant", "nearest", "reflect", "wrap", "mirror"}


def _range(value, name):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Each {name} range must contain two values.")
        low, high = map(float, value)
    else:
        value = float(value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative. Received {value}.")
        low, high = -value, value
    if low > high:
        raise ValueError(f"{name} ranges must be ordered as (min, max).")
    return low, high


def _axis_ranges(value, axes, name):
    if value is None:
        return {axis: (0.0, 0.0) for axis in axes}
    if isinstance(value, dict):
        unknown = set(value) - set(axes)
        if unknown:
            raise ValueError(f"{name} axes must be drawn from {axes}.")
        return {
            axis: _range(value[axis], name) if axis in value else (0.0, 0.0)
            for axis in axes
        }
    value_range = _range(value, name)
    return {axis: value_range for axis in axes}


def _per_key(keys, value, default, name):
    if value is None:
        return {key: default(key, index) for index, key in enumerate(keys)}
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


def _sample(transform, ranges, batch_size, dtype, gate):
    values = {
        name: transform.random_uniform(
            shape=(batch_size,), minval=low, maxval=high, dtype=dtype
        )
        * gate
        for name, (low, high) in ranges.items()
    }
    return values


def _rotation_2d(angle):
    c, s = ops.cos(angle), ops.sin(angle)
    return ops.stack(
        [
            ops.stack([c, -s], axis=-1),
            ops.stack([s, c], axis=-1),
        ],
        axis=-2,
    )


def _rotation_3d(angles):
    z, y, x = angles[:, 0], angles[:, 1], angles[:, 2]
    one, zero = ops.ones_like(z), ops.zeros_like(z)
    cz, sz, cy, sy, cx, sx = (
        ops.cos(z), ops.sin(z), ops.cos(y), ops.sin(y), ops.cos(x), ops.sin(x)
    )
    rz = ops.stack(
        [ops.stack([one, zero, zero], -1), ops.stack([zero, cz, -sz], -1),
         ops.stack([zero, sz, cz], -1)], -2
    )
    ry = ops.stack(
        [ops.stack([cy, zero, sy], -1), ops.stack([zero, one, zero], -1),
         ops.stack([-sy, zero, cy], -1)], -2
    )
    rx = ops.stack(
        [ops.stack([cx, -sx, zero], -1), ops.stack([sx, cx, zero], -1),
         ops.stack([zero, zero, one], -1)], -2
    )
    return compose_affine_matrices(rz, ry, rx)


def _shear_2d(values):
    xy, yx = values[:, 0], values[:, 1]
    return ops.stack(
        [
            ops.stack([ops.ones_like(xy), yx], -1),
            ops.stack([xy, ops.ones_like(xy)], -1),
        ],
        -2,
    )


def _shear_3d(values):
    zy, zx, yz, yx, xz, xy = [values[:, i] for i in range(6)]
    one = ops.ones_like(zy)
    return ops.stack(
        [ops.stack([one, zy, zx], -1), ops.stack([yz, one, yx], -1),
         ops.stack([xz, xy, one], -1)], -2
    )


def _matrix_to_keras_2d(matrix):
    return ops.stack(
        [matrix[:, 1, 1], matrix[:, 1, 0], matrix[:, 1, 2],
         matrix[:, 0, 1], matrix[:, 0, 0], matrix[:, 0, 2],
         ops.zeros_like(matrix[:, 0, 0]), ops.zeros_like(matrix[:, 0, 0])],
        -1,
    )


def _centered_parts(matrix, spatial_shape):
    rank = len(spatial_shape)
    linear = matrix[:, :rank, :rank]
    center = ops.cast(
        (ops.convert_to_tensor(spatial_shape, dtype="float32") - 1.0) / 2.0,
        matrix.dtype,
    )
    translation = matrix[:, :rank, rank] + ops.einsum(
        "bij,j->bi", linear, center
    ) - center
    return linear, translation


class RandomAffine(RandomTransform):
    """Apply rotation, zoom, translation, and shear in one resampling pass.

    The forward matrix is composed as ``translation @ rotation @ shear @
    scale``. Its inverse is used for sampling, and the realized matrices are
    recorded for inverse execution. Parameters are sampled independently per
    batch item and shared across selected image and label keys.
    """

    def __init__(
        self,
        keys: Sequence[str],
        rotation_factor=None,
        zoom_factor=None,
        translation_factor=None,
        shear_factor=None,
        prob=0.5,
        interpolation=None,
        fill_mode="constant",
        fill_value=0.0,
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
        if self.layout_info.spatial_rank == 2:
            self.rotation_ranges = (
                {"z": _range(rotation_factor, "rotation_factor")}
                if rotation_factor is not None
                else {"z": (0.0, 0.0)}
            )
            axes = ("y", "x")
            shear_axes = ("xy", "yx")
        else:
            self.rotation_ranges = _axis_ranges(rotation_factor, ("z", "y", "x"), "rotation_factor")
            axes = ("z", "y", "x")
            shear_axes = ("zy", "zx", "yz", "yx", "xz", "xy")
        self.zoom_ranges = _axis_ranges(zoom_factor, axes, "zoom_factor")
        self.translation_ranges = _axis_ranges(translation_factor, axes, "translation_factor")
        self.shear_ranges = _axis_ranges(shear_factor, shear_axes, "shear_factor")
        self.interpolation = _per_key(
            self.keys, interpolation,
            lambda _, index: (
                "bilinear" if self.layout_info.spatial_rank == 2 else "trilinear"
            )
            if index == 0
            else "nearest",
            "interpolation",
        )
        self.fill_mode = _per_key(
            self.keys, fill_mode, lambda *_: "constant", "fill_mode"
        )
        self.fill_value = _per_key(
            self.keys, fill_value, lambda *_: 0.0, "fill_value"
        )
        for key in self.keys:
            self.interpolation[key] = str(self.interpolation[key]).lower()
            self.fill_mode[key] = str(self.fill_mode[key]).lower()
            if self.interpolation[key] not in _INTERPOLATION_MODES[
                self.layout_info.spatial_rank
            ]:
                raise ValueError(f"Unsupported interpolation for key {key!r}.")
            if self.fill_mode[key] not in _FILL_MODES:
                raise ValueError(f"Unsupported fill_mode {self.fill_mode[key]!r}.")

    @property
    def invertible(self):
        return True

    def _matrices(self, spatial_shape, batch_size):
        gate = ops.cast(
            self.random_uniform(
                shape=(batch_size,), minval=0.0, maxval=1.0, dtype="float32"
            )
            < self.prob,
            "float32",
        )
        rotation = _sample(self, self.rotation_ranges, batch_size, "float32", gate)
        zoom = _sample(self, self.zoom_ranges, batch_size, "float32", gate)
        translation = _sample(
            self, self.translation_ranges, batch_size, "float32", gate
        )
        shear = _sample(self, self.shear_ranges, batch_size, "float32", gate)
        applied = ops.any(gate > 0)
        rank = self.layout_info.spatial_rank
        if rank == 2:
            linear_rotation = _rotation_2d(rotation["z"])
            linear_shear = _shear_2d(ops.stack([shear["xy"], shear["yx"]], -1))
            linear_scale = ops.eye(2, dtype="float32") * ops.reshape(
                ops.stack([1.0 + zoom["y"], 1.0 + zoom["x"]], -1), (-1, 2, 1)
            )
            offset = ops.stack(
                [
                    translation["y"] * spatial_shape[0],
                    translation["x"] * spatial_shape[1],
                ],
                -1,
            )
        else:
            linear_rotation = _rotation_3d(
                ops.stack([rotation["z"], rotation["y"], rotation["x"]], -1)
            )
            linear_shear = _shear_3d(
                ops.stack(
                    [shear[name] for name in ("zy", "zx", "yz", "yx", "xz", "xy")],
                    -1,
                )
            )
            linear_scale = ops.eye(3, dtype="float32") * ops.reshape(
                ops.stack([1.0 + zoom["z"], 1.0 + zoom["y"], 1.0 + zoom["x"]], -1), (-1, 3, 1)
            )
            offset = ops.stack(
                [
                    translation["z"] * spatial_shape[0],
                    translation["y"] * spatial_shape[1],
                    translation["x"] * spatial_shape[2],
                ],
                -1,
            )
        identity = ops.eye(rank, dtype="float32")
        forward = compose_affine_matrices(
            centered_affine_matrix(identity, spatial_shape, offset),
            centered_affine_matrix(linear_rotation, spatial_shape),
            centered_affine_matrix(linear_shear, spatial_shape),
            centered_affine_matrix(linear_scale, spatial_shape),
        )
        return forward, invert_affine_matrix(forward), applied

    def _apply_tensor(self, tensor, key, matrix):
        batched, added_batch = ensure_batch_axis_for_layout(
            tensor, input_layout=self.input_layout, allowed_spatial_ranks=(2, 3)
        )
        spatial_shape = ops.shape(batched)[1:-1]
        if self.layout_info.spatial_rank == 2:
            output = ops.image.affine_transform(
                ops.cast(batched, "float32"), _matrix_to_keras_2d(matrix),
                interpolation=self.interpolation[key], fill_mode=self.fill_mode[key],
                fill_value=self.fill_value[key],
            )
        else:
            linear, translation = _centered_parts(matrix, spatial_shape)

            def sample(args):
                volume, sample_matrix, sample_offset = args
                return sample_affine_volume(
                    volume, sample_matrix, self.interpolation[key],
                    self.fill_mode[key], self.fill_value[key], sample_offset
                )

            output = ops.vectorized_map(sample, (batched, linear, translation))
        return restore_from_batch_axis(output, added_batch)

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present = [key for key in self.keys if key in bundle.data]
        if not self.allow_missing_keys and len(present) != len(self.keys):
            raise KeyError(
                "Keys not found in input data: "
                f"{[key for key in self.keys if key not in bundle.data]}."
            )
        if not present:
            return bundle
        reference = bundle.data[present[0]]
        validate_tensor_matches_layout(
            reference, self.input_layout, transform_name=type(self).__name__
        )
        batched, _ = ensure_batch_axis_for_layout(
            reference,
            input_layout=self.input_layout,
            allowed_spatial_ranks=(2, 3),
        )
        forward, inverse, applied = self._matrices(ops.shape(batched)[1:-1], ops.shape(batched)[0])
        for key in present:
            bundle.data[key] = self._apply_tensor(bundle.data[key], key, inverse)
        self.record_random_transform(
            bundle,
            params={
                "keys": present,
                "forward_matrix": forward,
                "inverse_matrix": inverse,
                "input_layout": self.input_layout,
            },
            applied=applied,
            kernel="random_affine",
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = _pop_last_transform_trace(bundle, type(self).__name__)
        if trace is None:
            return bundle
        for key in trace["params"]["keys"]:
            if key in bundle.data:
                bundle.data[key] = self._apply_tensor(
                    bundle.data[key], key, trace["params"]["forward_matrix"]
                )
        return bundle

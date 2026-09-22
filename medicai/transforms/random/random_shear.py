"""Backend-neutral random shear for channel-last medical tensors."""

from typing import Any, Sequence

import keras
from keras import ops

from ..base import (
    RandomTransform,
    _normalize_keys,
    _pop_last_transform_trace,
    _validate_last_transform_keys,
)
from ..tensor_bundle import TensorBundle
from ..utils import (
    ensure_batch_axis_for_layout,
    get_input_layout_info,
    resolve_input_layout,
    restore_from_batch_axis,
    validate_tensor_matches_layout,
)
from .affine import (
    apply_plane_affine_3d,
    normalize_resampling_options,
    resolve_axis_ranges,
    resolve_per_key,
    sample_affine_volumes,
    validate_fixed_shear_ranges,
)

_DEFAULT_PROB = 0.5
_DEFAULT_SHEAR_FACTOR = 0.0
_DEFAULT_INTERPOLATION = None
_DEFAULT_FILL_MODE = "constant"
_DEFAULT_FILL_VALUE = 0.0
_INTERPOLATION_MODES = {
    2: {"bilinear", "nearest"},
    3: {"trilinear", "nearest"},
}
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
        matrix = ops.stack(
            [
                matrix[:, 0],
                matrix[:, 1],
                matrix[:, 2],
                matrix[:, 3],
                matrix[:, 4],
                matrix[:, 5],
                ops.zeros_like(matrix[:, 0]),
                ops.zeros_like(matrix[:, 0]),
                ops.ones_like(matrix[:, 0]),
            ],
            axis=-1,
        )
        matrix = ops.linalg.inv(ops.reshape(matrix, (-1, 3, 3)))
        matrix = ops.stack(
            [
                matrix[:, 0, 0],
                matrix[:, 0, 1],
                matrix[:, 0, 2],
                matrix[:, 1, 0],
                matrix[:, 1, 1],
                matrix[:, 1, 2],
                matrix[:, 2, 0],
                matrix[:, 2, 1],
                matrix[:, 2, 2],
            ],
            axis=-1,
        )
        matrix = ops.stack(
            [
                matrix[:, 0],
                matrix[:, 1],
                matrix[:, 2],
                matrix[:, 3],
                matrix[:, 4],
                matrix[:, 5],
                matrix[:, 6],
                matrix[:, 7],
            ],
            axis=-1,
        )
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


class RandomShear(RandomTransform):
    """Randomly shear channel-last 2D images or 3D volumes.

    Shearing changes the angle between spatial axes while preserving the
    channel-last image or volume grid. The transform uses the rank-appropriate
    affine resampling path for 2D and 3D inputs, and samples parameters
    independently for each batch item while sharing them across selected keys.

    This keeps images, masks, and labels spatially aligned while allowing each
    batch item to receive a different random shear.

    Resampling uses ``float32`` internally and restores the original input
    dtype. Use floating-point image tensors with bilinear or trilinear
    interpolation to preserve fractional values, and use nearest interpolation
    for discrete labels. Integer images with linear interpolation may lose
    fractional values when cast back.

    .. note::

        On the TensorFlow backend, 2D shearing uses the affine image kernel
        backed by ``tf.raw_ops.ImageProjectiveTransformV3``, which is not
        currently XLA-compatible. A 3D shear confined to the H-W plane uses
        the same folded 2D path and has the same limitation. Full 3D shearing
        uses the general coordinate-sampling path instead. Eager and
        ``tf.data`` graph execution remain supported. However, for Jax and Torch
        backends, this limitation does not apply; they are XLA-compatible.

    Args:
        keys: Tensor keys to shear together.
        factor: A scalar, a ``(min, max)`` range, or an axis-pair mapping. A
            scalar or range is applied independently to every supported shear
            coefficient: ``xy`` and ``yx`` for 2D, or ``zy``, ``zx``, ``yz``,
            ``yx``, ``xz``, and ``xy`` for 3D. The axis names follow the
            channel-last ``[D]HWC`` order, where ``z`` maps to ``D``, ``y`` to
            ``H``, and ``x`` to ``W``. A mapping can enable only selected
            coefficients; omitted coefficients remain zero. For example,
            ``{"zy": 0.05}`` shears the 3D ``z`` output relative to ``y``.
        prob: Per-sample probability of applying the shear.
        interpolation: One mode, one mode per key, or a key-to-mode mapping.
        fill_mode: Boundary behavior for newly exposed values.
        fill_value: Constant boundary value when ``fill_mode="constant"``.
        input_layout: One of ``HWC``, ``DHWC``, ``BHWC``, or ``BDHWC``. The
            optional batch and depth dimensions follow ``B[D]HWC``.
        seed: Optional integer or Keras seed generator.
        allow_missing_keys: If ``True``, missing requested keys are skipped.

    Example:

        The selected 3D coefficients apply small shears: up to 5% for the
        depth-related terms and up to 10% for the in-plane terms. Unspecified
        coefficients remain zero, which avoids introducing unintended planes
        of deformation::

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import RandomShear

            transform = RandomShear(
                keys=["image", "label"],
                factor={"zy": 0.05, "zx": 0.05, "xy": 0.1, "yx": 0.1},
                interpolation={"image": "trilinear", "label": "nearest"},
                input_layout="BDHWC",
                prob=0.5,
                seed=7,
            )
            image = tf.random.normal((2, 32, 64, 64, 1), seed=7)
            label = tf.zeros_like(image)
            result = transform({"image": image, "label": label})

        This 2D example uses explicit ``(min, max)`` ranges for both in-plane
        shear coefficients. Each coefficient can be sampled between -5% and
        +10%, allowing motion in either cross-axis direction::

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            from medicai.transforms import RandomShear

            transform = RandomShear(
                keys=["image"],
                factor={"xy": (-0.05, 0.1), "yx": (-0.05, 0.1)},
                input_layout="BHWC", seed=7
            )
            image = jax.random.normal(jax.random.PRNGKey(7), (8, 128, 128, 3))
            result = transform({"image": image})

        The scalar factor applies a symmetric shear range of up to 10% to all
        supported 2D shear coefficients::

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import RandomShear

            transform = RandomShear(
                keys=["image"], factor=0.1, input_layout="BHWC", seed=7
            )
            image = torch.randn((8, 128, 128, 3))
            result = transform({"image": image})
    """

    def __init__(
        self,
        keys: Sequence[str],
        factor: float | Sequence[float] | dict[str, Any] = _DEFAULT_SHEAR_FACTOR,
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
        axes = _AXES_2D if self.layout_info.spatial_rank == 2 else _AXES_3D
        self.ranges = resolve_axis_ranges(
            factor, axes, "Shear factor", lambda value, _: _factor_range(value)
        )
        validate_fixed_shear_ranges(self.ranges)
        self.interpolation = resolve_per_key(
            self.keys,
            interpolation,
            lambda _, index: (
                ("bilinear" if self.layout_info.spatial_rank == 2 else "trilinear")
                if index == 0
                else "nearest"
            ),
            "interpolation",
        )
        self.fill_mode = resolve_per_key(
            self.keys, fill_mode, lambda *_: _DEFAULT_FILL_MODE, "fill_mode"
        )
        self.fill_value = resolve_per_key(
            self.keys, fill_value, lambda *_: _DEFAULT_FILL_VALUE, "fill_value"
        )
        self.interpolation, self.fill_mode = normalize_resampling_options(
            self.keys,
            self.interpolation,
            self.fill_mode,
            self.layout_info.spatial_rank,
            _INTERPOLATION_MODES,
            _FILL_MODES,
        )

    @property
    def invertible(self) -> bool:
        return True

    def _sample_coefficients(self, batch_size, dtype="float32"):
        apply_mask = ops.cast(
            self.random_uniform(shape=(batch_size,), minval=0.0, maxval=1.0, dtype="float32")
            < self.prob,
            dtype,
        )
        coefficients = {}
        for axis_pair, (low, high) in self.ranges.items():
            coefficients[axis_pair] = (
                self.random_uniform(shape=(batch_size,), minval=low, maxval=high, dtype=dtype)
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
            matrix = _shear_matrix_2d(values, ops.shape(batched)[1:-1], inverse=inverse)
            output = ops.image.affine_transform(
                ops.cast(batched, "float32"),
                matrix,
                interpolation=self.interpolation[key],
                fill_mode=self.fill_mode[key],
                fill_value=self.fill_value[key],
            )
        else:
            values = ops.stack([coefficients[axis] for axis in _AXES_3D], axis=-1)
            matrices = _inverse_linear_3d(values, inverse=inverse)

            active_pairs = {
                pair for pair, (low, high) in self.ranges.items() if low != 0.0 or high != 0.0
            }
            if keras.config.backend() != "torch" and active_pairs and active_pairs <= {"xy", "yx"}:
                plane_axes = ("y", "x")
                plane_values = ops.stack(
                    [
                        coefficients["xy"],
                        coefficients["yx"],
                    ],
                    axis=-1,
                )
                plane_matrices = _shear_matrix_2d(
                    plane_values,
                    ops.stack([ops.shape(batched)[2], ops.shape(batched)[3]]),
                    inverse=inverse,
                )
                output = apply_plane_affine_3d(
                    batched,
                    plane_matrices,
                    plane_axes,
                    interpolation=self.interpolation[key],
                    fill_mode=self.fill_mode[key],
                    fill_value=self.fill_value[key],
                )
                output = ops.cast(output, tensor.dtype)
                return restore_from_batch_axis(output, added_batch)

            output = sample_affine_volumes(
                batched,
                matrices,
                self.interpolation[key],
                self.fill_mode[key],
                self.fill_value[key],
            )
        output = ops.cast(output, tensor.dtype)
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
        _validate_last_transform_keys(bundle, type(self).__name__, self.allow_missing_keys)
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

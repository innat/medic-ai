"""Backend-neutral random scaling for channel-last medical tensors."""

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
)

_DEFAULT_PROB = 0.5
_DEFAULT_SCALE_FACTOR = 0.0
_DEFAULT_INTERPOLATION = None
_DEFAULT_FILL_MODE = "constant"
_DEFAULT_FILL_VALUE = 0.0
_INTERPOLATION_MODES = {
    2: {"bilinear", "nearest"},
    3: {"trilinear", "nearest"},
}
_FILL_MODES = {"constant", "nearest", "reflect", "wrap", "mirror"}


def _factor_range(value: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Each scale factor range must contain two values.")
        low, high = float(value[0]), float(value[1])
    else:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"Scale factors must be non-negative. Received {value}.")
        low, high = -value, value
    if low > high:
        raise ValueError("Scale factor ranges must be ordered as (min, max).")
    if low <= -1.0:
        raise ValueError("Scale factors must keep the sampled scale above zero.")
    return low, high


def _scale_matrix_2d(scales, spatial_shape):
    """Build output-to-input matrices for centered 2D scaling."""
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


class RandomScale(RandomTransform):
    """Randomly scale channel-last 2D images or 3D volumes around their center.

    Scaling changes the spatial size of image structures relative to the
    image or volume center. The transform uses the rank-appropriate affine
    resampling path for 2D and 3D inputs, and samples parameters independently
    for each batch item while sharing them across selected keys.

    This keeps images, masks, and labels spatially aligned while allowing each
    batch item to receive a different random scale.

    Resampling uses ``float32`` internally and restores the original input
    dtype. Use floating-point image tensors with bilinear or trilinear
    interpolation to preserve fractional values, and use nearest interpolation
    for discrete labels. Integer images with linear interpolation may lose
    fractional values when cast back.

    .. warning::

        Ranges that allow scales very close to zero can make inverse affine
        sampling numerically unstable. Use moderate positive scale ranges.

    .. note::

        On the TensorFlow backend, 2D scaling uses the affine image kernel
        backed by ``tf.raw_ops.ImageProjectiveTransformV3``, which is not
        currently XLA-compatible. A 3D scale that changes only the H-W plane
        uses the same folded 2D path and has the same limitation. Full 3D
        scaling uses the general coordinate-sampling path instead. Eager and
        ``tf.data`` graph execution remain supported. However, for Jax and Torch
        backends, this limitation does not apply, they are XLA-compatible.

    Args:
        keys: Tensor keys to scale together.
        factor: A scalar, a ``(min, max)`` range, or an axis mapping. A
            scalar or range is applied independently to every spatial axis:
            ``(y, x)`` for 2D inputs and ``(z, y, x)`` for 3D inputs. The
            public axis names follow the channel-last ``[D]HWC`` order, so
            ``z`` maps to ``D``, ``y`` to ``H``, and ``x`` to ``W``. A scalar
            ``0.2`` samples relative scale factors from ``[-0.2, 0.2]`` and
            uses scales in ``[0.8, 1.2]``. A mapping can restrict scaling to
            selected axes; omitted axes keep scale ``1``. For example,
            ``{"x": 0.1, "y": 0.0}`` scales only the 2D image width.
        prob: Per-sample probability of applying the scaling.
        interpolation: One mode, one mode per key, or a key-to-mode mapping.
        fill_mode: Boundary behavior for newly exposed values.
        fill_value: Constant boundary value when ``fill_mode="constant"``.
        input_layout: One of ``HWC``, ``DHWC``, ``BHWC``, or ``BDHWC``. The
            optional batch and depth dimensions follow ``B[D]HWC``.
        seed: Optional integer or Keras seed generator.
        allow_missing_keys: If ``True``, missing requested keys are skipped.

    Example:

        The 3D example permits a smaller depth scaling (about +/-10%) and
        slightly stronger in-plane scaling (about +/-15%). The label uses
        nearest-neighbor interpolation so its class values are preserved::

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import RandomScale

            transform = RandomScale(
                keys=["image", "label"],
                factor={"z": 0.1, "y": 0.15, "x": 0.15},
                interpolation={"image": "trilinear", "label": "nearest"},
                input_layout="BDHWC",
                prob=0.5,
                seed=7,
            )
            image = tf.random.normal((2, 32, 64, 64, 1), seed=7)
            label = tf.zeros_like(image)
            result = transform({"image": image, "label": label})


        This 2D example uses explicit ``(min, max)`` ranges: height is scaled
        between -10% and +20%, while width is scaled between -5% and +15%.
        The batch and channel dimensions are not scaled::

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            from medicai.transforms import RandomScale

            transform = RandomScale(
                keys=["image"],
                factor={"y": (-0.1, 0.2), "x": (-0.05, 0.15)},
                input_layout="BHWC",
                prob=0.5,
                seed=7,
            )
            image = jax.random.normal(jax.random.PRNGKey(7), (8, 128, 128, 3))
            result = transform({"image": image})

        A scalar factor applies the same +/-10% relative scaling range to both
        spatial axes of this 2D batch::

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import RandomScale

            transform = RandomScale(
                keys=["image"], factor=0.1, input_layout="BHWC", seed=7
            )
            image = torch.randn((8, 128, 128, 3))
            result = transform({"image": image})
    """

    def __init__(
        self,
        keys: Sequence[str],
        factor: float | Sequence[float] | dict[str, Any] = _DEFAULT_SCALE_FACTOR,
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
        axes = ("y", "x") if self.layout_info.spatial_rank == 2 else ("z", "y", "x")
        self.ranges = resolve_axis_ranges(
            factor, axes, "Scale factor", lambda value, _: _factor_range(value)
        )
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

    def _sample_scales(self, batch_size, dtype="float32"):
        apply_mask = ops.cast(
            self.random_uniform(shape=(batch_size,), minval=0.0, maxval=1.0, dtype="float32")
            < self.prob,
            dtype,
        )
        factors = {}
        for axis, (low, high) in self.ranges.items():
            sampled = self.random_uniform(shape=(batch_size,), minval=low, maxval=high, dtype=dtype)
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
            matrix = _scale_matrix_2d(scale_tensor, ops.shape(batched)[1:-1])
            output = ops.image.affine_transform(
                ops.cast(batched, "float32"),
                matrix,
                interpolation=self.interpolation[key],
                fill_mode=self.fill_mode[key],
                fill_value=self.fill_value[key],
            )
        else:
            scale_tensor = ops.stack([scales["z"], scales["y"], scales["x"]], axis=-1)

            active_axes = {
                axis for axis, (low, high) in self.ranges.items() if low != 0.0 or high != 0.0
            }
            if keras.config.backend() != "torch" and active_axes and active_axes <= {"x", "y"}:
                matrix = _scale_matrix_2d(
                    ops.stack([scales["y"], scales["x"]], axis=-1),
                    ops.shape(batched)[2:4],
                )
                output = apply_plane_affine_3d(
                    batched,
                    matrix,
                    ("y", "x"),
                    interpolation=self.interpolation[key],
                    fill_mode=self.fill_mode[key],
                    fill_value=self.fill_value[key],
                )
                output = ops.cast(output, tensor.dtype)
                return restore_from_batch_axis(output, added_batch)

            inverse_matrices = ops.eye(3, dtype=scale_tensor.dtype) / ops.reshape(
                scale_tensor, (-1, 3, 1)
            )
            output = sample_affine_volumes(
                batched,
                inverse_matrices,
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
        scales, applied = self._sample_scales(batch_size)
        for key in present:
            bundle.data[key] = self._scale_tensor(bundle.data[key], key, scales)
        self.record_random_transform(
            bundle,
            params={"keys": present, "scales": scales, "input_layout": self.input_layout},
            applied=applied,
            kernel="random_scale",
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        _validate_last_transform_keys(bundle, type(self).__name__, self.allow_missing_keys)
        trace = _pop_last_transform_trace(bundle, type(self).__name__)
        if trace is None:
            return bundle
        inverse_scales = {axis: 1.0 / value for axis, value in trace["params"]["scales"].items()}
        for key in trace["params"]["keys"]:
            if key in bundle.data:
                bundle.data[key] = self._scale_tensor(bundle.data[key], key, inverse_scales)
        return bundle

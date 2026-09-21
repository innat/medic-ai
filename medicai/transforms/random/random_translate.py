"""Backend-neutral random translations for channel-last medical tensors."""

from typing import Any, Sequence

import keras
from keras import ops

from ..base import RandomTransform, _pop_last_transform_trace
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
_DEFAULT_FACTOR = 0.0
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
    return ops.concatenate([identity, ops.expand_dims(inverse_offsets, axis=-1)], axis=-1)


class RandomTranslate(RandomTransform):
    """Randomly translate channel-last 2D images or 3D volumes.

    Translation moves image content along one or more spatial axes without
    changing its scale or orientation. The transform uses the rank-appropriate
    affine resampling path for 2D and 3D inputs, and samples parameters
    independently for each batch item while sharing them across selected keys.

    This preserves image, mask, and label alignment while allowing each batch
    item to receive a different random displacement.

    Resampling uses ``float32`` internally and restores the original input
    dtype. Use floating-point image tensors with bilinear or trilinear
    interpolation to preserve fractional values, and use nearest interpolation
    for discrete labels. Integer images with linear interpolation may lose
    fractional values when cast back.

    .. note::

        On the TensorFlow backend, 2D translation uses the affine image kernel
        backed by ``tf.raw_ops.ImageProjectiveTransformV3``, which is not
        currently XLA-compatible. A 3D translation confined to the H-W plane
        uses the same folded 2D path and has the same limitation. Full 3D
        translation uses the general coordinate-sampling path instead. Eager
        and ``tf.data`` graph execution remain supported. However, for Jax and
        Torch backends, this limitation does not apply; they are XLA-compatible.

    Args:
        keys: Tensor keys to translate together.
        factor: A scalar, a ``(min, max)`` relative translation range, or an
            axis mapping. A scalar or range is applied independently to every
            spatial axis: ``(y, x)`` for 2D inputs and ``(z, y, x)`` for 3D
            inputs. The public axis names follow the channel-last ``[D]HWC``
            order, so ``z`` maps to ``D``, ``y`` to ``H``, and ``x`` to ``W``.
            The sampled value is relative to that axis length; for example,
            ``0.1`` permits a displacement in ``[-0.1 * size, 0.1 * size]``.
            A mapping can restrict translation to selected axes, while omitted
            axes remain unchanged.
        prob: Per-sample probability of applying the translation.
        interpolation: One mode, one mode per key, or a key-to-mode mapping.
        fill_mode: Boundary behavior for newly exposed values.
        fill_value: Constant boundary value when ``fill_mode="constant"``.
        input_layout: One of ``HWC``, ``DHWC``, ``BHWC``, or ``BDHWC``. The
            optional batch and depth dimensions follow ``B[D]HWC``.
        seed: Optional integer or Keras seed generator.
        allow_missing_keys: If ``True``, missing requested keys are skipped.

    Example:

        This 3D example allows up to 5% relative translation through depth and
        up to 10% through height and width. The same displacement is used for
        the image and label, while nearest interpolation protects label IDs::

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import RandomTranslate

            transform = RandomTranslate(
                keys=["image", "label"],
                factor={"z": 0.05, "y": 0.1, "x": 0.1},
                interpolation={"image": "trilinear", "label": "nearest"},
                input_layout="BDHWC",
                prob=0.5,
                seed=7,
            )
            image = tf.random.normal((2, 32, 64, 64, 1), seed=7)
            label = tf.zeros_like(image)
            result = transform({"image": image, "label": label})

        This 2D example uses explicit ``(min, max)`` ranges: height can move
        between -10% and +5% of its size, while width can move between -5% and
        +10% of its size::

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            from medicai.transforms import RandomTranslate

            transform = RandomTranslate(
                keys=["image"],
                factor={"y": (-0.1, 0.05), "x": (-0.05, 0.1)},
                input_layout="BHWC", seed=7
            )
            image = jax.random.normal(jax.random.PRNGKey(7), (8, 128, 128, 3))
            result = transform({"image": image})

        The scalar factor permits a symmetric translation range of up to 10%
        along both spatial axes::

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import RandomTranslate

            transform = RandomTranslate(
                keys=["image"], factor=0.1, input_layout="BHWC", seed=7
            )
            image = torch.randn((8, 128, 128, 3))
            result = transform({"image": image})
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
        axes = ("y", "x") if self.layout_info.spatial_rank == 2 else ("z", "y", "x")
        self.ranges = resolve_axis_ranges(
            factor, axes, "Translation factor", lambda value, _: _factor_range(value)
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
            self.keys,
            fill_mode,
            lambda *_: _DEFAULT_FILL_MODE,
            "fill_mode",
        )
        self.fill_value = resolve_per_key(
            self.keys,
            fill_value,
            lambda *_: _DEFAULT_FILL_VALUE,
            "fill_value",
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

    def _sample_offsets(self, batch_size, dtype="float32"):
        apply_mask = ops.cast(
            self.random_uniform(shape=(batch_size,), minval=0.0, maxval=1.0, dtype="float32")
            < self.prob,
            dtype,
        )
        offsets = {}
        for axis, (low, high) in self.ranges.items():
            offsets[axis] = (
                self.random_uniform(shape=(batch_size,), minval=low, maxval=high, dtype=dtype)
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

            active_axes = {
                axis for axis, (low, high) in self.ranges.items() if low != 0.0 or high != 0.0
            }
            if keras.config.backend() != "torch" and active_axes and active_axes <= {"x", "y"}:
                output = apply_plane_affine_3d(
                    batched,
                    _translation_matrix_2d(offset_tensor[:, 1:]),
                    ("y", "x"),
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
        offsets = {axis: -value for axis, value in trace["params"]["offsets"].items()}
        for key in trace["params"]["keys"]:
            if key in bundle.data:
                bundle.data[key] = self._apply_tensor(bundle.data[key], key, offsets)
        return bundle

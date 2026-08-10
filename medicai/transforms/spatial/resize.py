from __future__ import annotations

from typing import Mapping, Sequence

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform, _pop_last_transform_trace
from ..tensor_bundle import TensorBundle
from ..utils import (
    ensure_batch_axis,
    ensure_spatial_tuple,
    get_spatial_shape,
    restore_from_batch_axis,
    resize_volumes,
    validate_input_mode,
    validate_layout,
    validate_spatial_dims,
)


class Resize(KeyedTransform, InvertibleTransform):
    """Resize selected tensors to a requested spatial shape.

    ``Resize`` resamples channel-last tensors to a fixed 2D or 3D spatial
    shape while preserving the channel dimension. It supports both image-like
    tensors and label-like tensors, with interpolation configured per key so
    continuous images can use linear interpolation while discrete labels can
    use nearest-neighbor interpolation.

    Depending on ``input_mode``, this transform supports:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

    ``Resize`` is invertible in the limited sense that it records the original
    spatial shape and can resize the transformed result back to that shape via
    :meth:`inverse`. As with any resampling transform, the restored tensor is
    shape-consistent but may not be numerically identical to the original.

    Args:
        keys: Keys of tensors to resize.
        interpolation: Interpolation mode specified as a single string, a sequence
            aligned with ``keys``, or a mapping from key to interpolation mode. Valid modes
            are ``"bilinear"`` and ``"nearest"`` for 2D targets, and
            ``"trilinear"`` and ``"nearest"`` for 3D targets. This argument
            is required so callers explicitly choose a rank-appropriate mode.
        target_shape: Target spatial shape. Must be length 2 for 2D resizing
            or length 3 for 3D resizing. This argument is required so callers
            explicitly define the intended output rank.
        input_mode: Either ``"sample"`` for ``(H, W, C)`` / ``(D, H, W, C)``
            tensors, or ``"batch"`` for ``(B, H, W, C)`` / ``(B, D, H, W, C)``
            tensors.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Resize a 2D image-label pair using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Resize

            transform = Resize(
                keys=["image", "label"],
                interpolation=("bilinear", "nearest"),
                target_shape=(128, 128),
            )

            image = tf.random.normal((96, 96, 1))
            label = tf.random.uniform(
                (96, 96, 1), maxval=2, dtype=tf.int32
            )

            result = transform({"image": image, "label": label})
            resized_image = result["image"]
            resized_label = result["label"]

        Resize a 3D image volume using a ``TensorBundle`` and restore its
        original shape:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Resize, TensorBundle

            transform = Resize(
                keys=["image"],
                interpolation="trilinear",
                target_shape=(32, 64, 64),
            )

            image = tf.random.normal((48, 96, 96, 1))
            bundle = TensorBundle({"image": image})
            forward = transform(bundle)
            restored = transform.inverse(forward)

            print(forward["image"].shape)
            print(restored["image"].shape)

        Resize an image-label pair and restore the original spatial size:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Resize, TensorBundle

            transform = Resize(
                keys=["image", "label"],
                interpolation=("bilinear", "nearest"),
                target_shape=(48, 48),
            )

            image = tf.random.normal((96, 96, 1))
            label = tf.random.uniform(
                (96, 96, 1), maxval=2, dtype=tf.int32
            )
            forward = transform({"image": image, "label": label})
            restored = transform.inverse(forward)

            print(forward["image"].shape, forward["label"].shape)
            print(restored["image"].shape, restored["label"].shape)

    Returns:
        ``TensorBundle``: The input bundle with resized tensors, recorded
        original shapes, and an invertible transform trace entry appended.

    Raises:
        ValueError: If ``target_shape`` is not 2D or 3D, or if an invalid
            interpolation mode is provided for the requested dimensionality.
        TypeError: If ``interpolation`` is not a string, sequence, or mapping.
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
    """

    def __init__(
        self,
        keys: Sequence[str],
        interpolation: str | Sequence[str] | Mapping[str, str],
        target_shape: Sequence[int],
        input_mode: str = "sample",
        spatial_dims: int | None = None,
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.target_shape = tuple(target_shape)
        self.input_mode = validate_input_mode(input_mode, transform_name=type(self).__name__)
        self.spatial_dims = validate_spatial_dims(spatial_dims, transform_name=type(self).__name__)

        ndim = len(self.target_shape)
        if ndim not in (2, 3):
            raise ValueError(f"`target_shape` must be 2D or 3D, got {ndim}D.")

        valid_interpolations = {"bilinear", "nearest"} if ndim == 2 else {"trilinear", "nearest"}

        if isinstance(interpolation, str):
            self.interpolation = {key: interpolation for key in keys}
        elif isinstance(interpolation, (tuple, list)):
            if len(interpolation) != len(keys):
                raise ValueError("Length of 'interpolation' must match length of 'keys'.")
            self.interpolation = dict(zip(keys, interpolation))
        elif isinstance(interpolation, dict):
            missing_keys = set(keys) - set(interpolation.keys())
            if missing_keys:
                raise ValueError(f"Missing interpolation mode for keys: {sorted(missing_keys)}")
            self.interpolation = dict(interpolation)
        else:
            raise TypeError("'interpolation' must be a string, tuple, list, or dict.")

        for key, resize_interpolation in self.interpolation.items():
            if resize_interpolation not in valid_interpolations:
                raise ValueError(
                    f"Invalid interpolation '{resize_interpolation}' for {ndim}D input. "
                    f"Allowed: {sorted(valid_interpolations)} (key='{key}')."
                )

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        params = self.get_transform_params(bundle)
        original_shapes = {}

        def apply_resize(tensor: tf.Tensor, key: str) -> tf.Tensor:
            original_shapes[key] = self._get_original_spatial_shape(tensor)
            return self.transform_tensor(tensor, key, params)

        present_keys = self.apply_to_present_keys(bundle, apply_resize)
        self.record_transform(bundle, self.build_trace_params(params, present_keys, original_shapes))
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = self._get_last_resize_trace(bundle)
        if trace is None:
            return bundle

        original_shapes = trace["params"].get("original_shapes", {})

        def apply_inverse_resize(tensor: tf.Tensor, key: str) -> tf.Tensor:
            target_shape = original_shapes.get(key)
            if target_shape is None:
                return tensor
            return self.resize_tensor(tensor, key, target_shape=target_shape)

        self.apply_to_present_keys(
            bundle,
            apply_inverse_resize,
            keys=trace["params"].get("keys", []),
        )
        return bundle

    def get_transform_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Prepare forward-pass parameters for this resize."""
        del bundle
        return {
            "target_shape": self.target_shape,
            "input_mode": self.input_mode,
            "spatial_dims": self.spatial_dims,
        }

    def transform_tensor(
        self,
        tensor: tf.Tensor,
        key: str,
        params: dict[str, object],
    ) -> tf.Tensor:
        """Resize one tensor with the configured target shape."""
        return self.resize_tensor(tensor, key, target_shape=params["target_shape"])

    def build_trace_params(
        self,
        params: dict[str, object],
        present_keys: Sequence[str],
        original_shapes: Mapping[str, tf.Tensor],
    ) -> dict[str, object]:
        """Build invertible trace metadata for the current resize."""
        return {
            "keys": list(present_keys),
            "target_shape": params["target_shape"],
            "input_mode": params["input_mode"],
            "spatial_dims": params["spatial_dims"],
            "original_shapes": original_shapes,
            "interpolation": {key: self.interpolation[key] for key in present_keys},
        }

    def resize_tensor(
        self,
        tensor: tf.Tensor,
        key: str,
        target_shape: Sequence[int] | tf.Tensor,
    ) -> tf.Tensor:
        """Resize one tensor to the requested spatial shape."""
        if isinstance(target_shape, tf.Tensor):
            target_shape_tensor = target_shape
            target_rank = target_shape.shape[0]
            if target_rank is None:
                raise ValueError("`target_shape` tensor must have a statically known length.")
        else:
            target_rank = len(target_shape)
            target_shape_tensor = tf.convert_to_tensor(
                ensure_spatial_tuple(target_shape, target_rank, "target_shape"),
                dtype=tf.int32,
            )

        layout = self._resolve_layout(tensor, target_rank)
        batched_tensor, added_batch_axis = ensure_batch_axis(
            tensor,
            input_mode=self.input_mode,
            spatial_dims=self.spatial_dims,
            allowed_spatial_ranks=(target_rank,),
        )
        resized = self.resize_batch_tensor(
            batched_tensor,
            key,
            target_shape=target_shape_tensor,
            spatial_rank=layout.spatial_rank,
        )
        return restore_from_batch_axis(resized, added_batch_axis)

    def resize_batch_tensor(
        self,
        tensor: tf.Tensor,
        key: str,
        target_shape: tf.Tensor,
        spatial_rank: int | None = None,
    ) -> tf.Tensor:
        """Resize one batch-layout tensor to the requested spatial shape."""
        effective_spatial_rank = spatial_rank
        if effective_spatial_rank is None:
            layout = validate_layout(
                tensor,
                input_mode="batch",
                allowed_spatial_ranks=(2, 3),
                transform_name=type(self).__name__,
            )
            effective_spatial_rank = layout.spatial_rank

        if effective_spatial_rank == 2:
            return self._resize_2d(tensor, key, target_shape)
        if effective_spatial_rank == 3:
            return self._resize_3d(tensor, key, target_shape)
        raise ValueError(
            f"{type(self).__name__} supports only 2D or 3D tensors, got spatial rank "
            f"{effective_spatial_rank}."
        )

    def _resize_2d(self, tensor: tf.Tensor, key: str, target_shape: tf.Tensor) -> tf.Tensor:
        return tf.image.resize(tensor, target_shape, method=self.interpolation.get(key))

    def _resize_3d(self, tensor: tf.Tensor, key: str, target_shape: tf.Tensor) -> tf.Tensor:
        resized = resize_volumes(
            tensor,
            target_shape[0],
            target_shape[1],
            target_shape[2],
            method=self.interpolation.get(key),
            align_corners=False,
        )
        return resized

    def _resolve_layout(self, tensor: tf.Tensor, target_rank: int):
        """Validate the current tensor layout against ``input_mode`` and ``target_shape``."""
        return validate_layout(
            tensor,
            input_mode=self.input_mode,
            allowed_spatial_ranks=(target_rank,),
            spatial_dims=self.spatial_dims,
            transform_name=type(self).__name__,
        )

    def _get_original_spatial_shape(self, tensor: tf.Tensor) -> tf.Tensor:
        """Extract the original spatial shape using the configured target rank."""
        self._resolve_layout(tensor, len(self.target_shape))
        return get_spatial_shape(tensor, input_mode=self.input_mode)

    def _get_last_resize_trace(self, bundle: TensorBundle) -> dict | None:
        return _pop_last_transform_trace(bundle, type(self).__name__)

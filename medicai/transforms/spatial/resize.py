from __future__ import annotations

from typing import Mapping, Sequence

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import ensure_spatial_tuple, get_tensor_rank


class Resize(KeyedTransform, InvertibleTransform):
    """Resize selected tensors to a requested spatial shape.

    ``Resize`` resamples channel-last tensors to a fixed 2D or 3D spatial
    shape while preserving the channel dimension. It supports both image-like
    tensors and label-like tensors, with interpolation configured per key so
    continuous images can use linear interpolation while discrete labels can
    use nearest-neighbor interpolation.

    This transform supports:

    - 2D tensors shaped ``(H, W, C)``
    - 3D tensors shaped ``(D, H, W, C)``

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
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.target_shape = tuple(target_shape)

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
        original_shapes = {}

        def apply_resize(tensor: tf.Tensor, key: str) -> tf.Tensor:
            original_shapes[key] = self._get_original_spatial_shape(tensor)
            return self.resize_tensor(tensor, key, target_shape=self.target_shape)

        present_keys = self.apply_to_present_keys(bundle, apply_resize)
        self.record_transform(
            bundle,
            {
                "keys": list(present_keys),
                "target_shape": self.target_shape,
                "original_shapes": original_shapes,
                "interpolation": {key: self.interpolation[key] for key in present_keys},
            },
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = self._get_last_resize_trace(bundle)
        if trace is None:
            return bundle

        original_shapes = trace["params"].get("original_shapes", {})
        present_keys = [key for key in trace["params"].get("keys", []) if key in bundle.data]

        def apply_inverse_resize(tensor: tf.Tensor, key: str) -> tf.Tensor:
            target_shape = original_shapes.get(key)
            if target_shape is None:
                return tensor
            return self.resize_tensor(tensor, key, target_shape=target_shape)

        self.apply_to_present_keys(bundle, apply_inverse_resize, keys=present_keys)
        return bundle

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

        spatial_rank = self._resolve_spatial_rank(tensor, target_rank)
        if spatial_rank == 2:
            return self._resize_2d(tensor, key, target_shape_tensor)
        if spatial_rank == 3:
            return self._resize_3d(tensor, key, target_shape_tensor)
        raise ValueError(
            f"{type(self).__name__} supports only 2D or 3D tensors, got spatial rank "
            f"{spatial_rank}."
        )

    def _resize_2d(self, tensor: tf.Tensor, key: str, target_shape: tf.Tensor) -> tf.Tensor:
        return tf.image.resize(tensor, target_shape, method=self.interpolation.get(key))

    def _resize_3d(self, tensor: tf.Tensor, key: str, target_shape: tf.Tensor) -> tf.Tensor:
        added_batch = tensor.shape.rank == 4
        if added_batch:
            tensor = tensor[None, ...]
        resized = resize_volumes(
            tensor,
            target_shape[0],
            target_shape[1],
            target_shape[2],
            method=self.interpolation.get(key),
            align_corners=False,
        )
        return resized[0] if added_batch else resized

    def _resolve_spatial_rank(self, tensor: tf.Tensor, target_rank: int) -> int:
        """Resolve whether a tensor is unbatched or batched for the requested target rank."""
        tensor_rank = get_tensor_rank(tensor)
        if target_rank not in (2, 3):
            raise ValueError(f"`target_shape` must be 2D or 3D, got {target_rank}D.")

        if tensor_rank == target_rank + 1:
            return target_rank
        if tensor_rank == target_rank + 2:
            return target_rank

        raise ValueError(
            f"{type(self).__name__} expects a channel-last tensor shaped either as an unbatched "
            "sample "
            f"or batched sample compatible with target spatial rank {target_rank}. "
            f"Received shape {tensor.shape}."
        )

    def _get_original_spatial_shape(self, tensor: tf.Tensor) -> tf.Tensor:
        """Extract the original spatial shape using the configured target rank."""
        target_rank = len(self.target_shape)
        tensor_rank = get_tensor_rank(tensor)

        if tensor_rank == target_rank + 1:
            return tf.shape(tensor)[:target_rank]
        if tensor_rank == target_rank + 2:
            return tf.shape(tensor)[1 : 1 + target_rank]

        raise ValueError(
            f"{type(self).__name__} expects a channel-last tensor shaped either as an unbatched "
            "sample "
            f"or batched sample compatible with target spatial rank {target_rank}. "
            f"Received shape {tensor.shape}."
        )

    def _get_last_resize_trace(self, bundle: TensorBundle) -> dict | None:
        for entry in reversed(bundle.get_applied_transforms()):
            if entry.get("name") == type(self).__name__:
                return entry
        return None


# This could be temporary: issue: https://github.com/keras-team/keras/issues/21785
def resize_volumes(volumes, depth, height, width, method="trilinear", align_corners=False):
    def trilinear_resize(volumes, depth, height, width, align_corners):
        original_dtype = volumes.dtype
        volumes = tf.cast(volumes, "float32")
        in_d = tf.shape(volumes)[1]
        in_h = tf.shape(volumes)[2]
        in_w = tf.shape(volumes)[3]

        if align_corners:
            z_coords = tf.linspace(0.0, tf.cast(in_d - 1, "float32"), depth)
            y_coords = tf.linspace(0.0, tf.cast(in_h - 1, "float32"), height)
            x_coords = tf.linspace(0.0, tf.cast(in_w - 1, "float32"), width)
        else:
            scale_d = tf.cast(in_d, "float32") / tf.cast(depth, "float32")
            scale_h = tf.cast(in_h, "float32") / tf.cast(height, "float32")
            scale_w = tf.cast(in_w, "float32") / tf.cast(width, "float32")

            z_coords = (tf.range(depth, dtype="float32") + 0.5) * scale_d - 0.5
            y_coords = (tf.range(height, dtype="float32") + 0.5) * scale_h - 0.5
            x_coords = (tf.range(width, dtype="float32") + 0.5) * scale_w - 0.5

            z_coords = tf.clip_by_value(z_coords, 0.0, tf.cast(in_d - 1, "float32"))
            y_coords = tf.clip_by_value(y_coords, 0.0, tf.cast(in_h - 1, "float32"))
            x_coords = tf.clip_by_value(x_coords, 0.0, tf.cast(in_w - 1, "float32"))

        def interpolate_1d(input_vol, coords, axis):
            idx0 = tf.cast(tf.floor(coords), "int32")
            idx1 = tf.minimum(idx0 + 1, tf.shape(input_vol)[axis] - 1)

            values0 = tf.gather(input_vol, idx0, axis=axis)
            values1 = tf.gather(input_vol, idx1, axis=axis)

            weight1 = coords - tf.cast(idx0, "float32")
            weight0 = 1.0 - weight1

            new_shape = [1] * 5
            new_shape[axis] = tf.shape(coords)[0]
            weight0 = tf.reshape(weight0, new_shape)
            weight1 = tf.reshape(weight1, new_shape)
            return weight0 * values0 + weight1 * values1

        interp_d = interpolate_1d(volumes, z_coords, axis=1)
        interp_h = interpolate_1d(interp_d, y_coords, axis=2)
        interp_w = interpolate_1d(interp_h, x_coords, axis=3)
        return tf.cast(interp_w, original_dtype)

    def nearest(volumes, depth, height, width):
        shape = tf.shape(volumes)
        bs, d, h, w, c = shape[0], shape[1], shape[2], shape[3], shape[4]

        z = tf.linspace(0.0, tf.cast(d - 1, "float32"), depth)
        z = tf.cast(tf.round(z), "int32")
        z = tf.clip_by_value(z, 0, d - 1)

        y = tf.linspace(0.0, tf.cast(h - 1, "float32"), height)
        y = tf.cast(tf.round(y), "int32")
        y = tf.clip_by_value(y, 0, h - 1)

        x = tf.linspace(0.0, tf.cast(w - 1, "float32"), width)
        x = tf.cast(tf.round(x), "int32")
        x = tf.clip_by_value(x, 0, w - 1)

        z_grid, y_grid, x_grid = tf.meshgrid(z, y, x, indexing="ij")
        z_grid = tf.reshape(z_grid, (-1,))
        y_grid = tf.reshape(y_grid, (-1,))
        x_grid = tf.reshape(x_grid, (-1,))

        batch_idx = tf.repeat(tf.range(bs), tf.shape(z_grid)[0])
        z_grid = tf.tile(z_grid, [bs])
        y_grid = tf.tile(y_grid, [bs])
        x_grid = tf.tile(x_grid, [bs])

        flat = tf.reshape(volumes, (bs * d * h * w, c))
        indices = (batch_idx * d * h * w) + (z_grid * h * w) + (y_grid * w) + x_grid
        result = tf.gather(flat, indices, axis=0)
        return tf.reshape(result, (bs, depth, height, width, c))

    if method == "trilinear":
        return trilinear_resize(volumes, depth, height, width, align_corners)
    if method == "nearest":
        return nearest(volumes, depth, height, width)
    raise ValueError(f"Unsupported resize method: {method}")

from __future__ import annotations

from typing import Mapping, Sequence

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import ensure_spatial_tuple, get_spatial_rank


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
        mode: Interpolation mode specified as a single string, a sequence
            aligned with ``keys``, or a mapping from key to mode. Valid modes
            are ``"bilinear"`` and ``"nearest"`` for 2D targets, and
            ``"trilinear"`` and ``"nearest"`` for 3D targets. This argument
            is required so callers explicitly choose a rank-appropriate mode.
        spatial_shape: Target spatial shape. Must be length 2 for 2D resizing
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
                mode=("bilinear", "nearest"),
                spatial_shape=(128, 128),
            )

            image = tf.random.normal((96, 96, 1))
            label = tf.random.uniform((96, 96, 1), maxval=2, dtype=tf.int32)

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
                mode="trilinear",
                spatial_shape=(32, 64, 64),
            )

            image = tf.random.normal((48, 96, 96, 1))
            bundle = TensorBundle({"image": image})

            forward = transform(bundle)
            restored = transform.inverse(forward)

            print(forward["image"].shape)
            print(restored["image"].shape)

    Returns:
        ``TensorBundle``: The input bundle with resized tensors, recorded
        original shapes, and an invertible transform trace entry appended.

    Raises:
        ValueError: If ``spatial_shape`` is not 2D or 3D, or if an invalid
            interpolation mode is provided for the requested dimensionality.
        TypeError: If ``mode`` is not a string, sequence, or mapping.
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
    """

    def __init__(
        self,
        keys: Sequence[str],
        mode: str | Sequence[str] | Mapping[str, str],
        spatial_shape: Sequence[int],
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.spatial_shape = tuple(spatial_shape)

        ndim = len(self.spatial_shape)
        if ndim not in (2, 3):
            raise ValueError(f"`spatial_shape` must be 2D or 3D, got {ndim}D.")

        valid_modes = {"bilinear", "nearest"} if ndim == 2 else {"trilinear", "nearest"}

        if isinstance(mode, str):
            self.mode = {key: mode for key in keys}
        elif isinstance(mode, (tuple, list)):
            if len(mode) != len(keys):
                raise ValueError("Length of 'mode' must match length of 'keys'.")
            self.mode = dict(zip(keys, mode))
        elif isinstance(mode, dict):
            self.mode = dict(mode)
        else:
            raise TypeError("'mode' must be a string, tuple, list, or dict.")

        for key, resize_mode in self.mode.items():
            if resize_mode not in valid_modes:
                raise ValueError(
                    f"Invalid mode '{resize_mode}' for {ndim}D input. "
                    f"Allowed: {sorted(valid_modes)} (key='{key}')."
                )

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        original_shapes = {}

        def apply_resize(tensor: tf.Tensor, key: str) -> tf.Tensor:
            original_shapes[key] = tf.shape(tensor)[: get_spatial_rank(tensor)]
            return self.resize_tensor(tensor, key, spatial_shape=self.spatial_shape)

        present_keys = self.apply_to_present_keys(bundle, apply_resize)
        self.record_transform(
            bundle,
            {
                "keys": list(present_keys),
                "spatial_shape": self.spatial_shape,
                "original_shapes": original_shapes,
                "mode": {key: self.mode[key] for key in present_keys},
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
            return self.resize_tensor(tensor, key, spatial_shape=target_shape)

        self.apply_to_present_keys(bundle, apply_inverse_resize, keys=present_keys)
        return bundle

    def resize_tensor(
        self,
        tensor: tf.Tensor,
        key: str,
        spatial_shape: Sequence[int] | tf.Tensor,
    ) -> tf.Tensor:
        """Resize one tensor to the requested spatial shape."""
        spatial_rank = get_spatial_rank(tensor)
        if isinstance(spatial_shape, tf.Tensor):
            target_shape = spatial_shape
        else:
            target_shape = tf.convert_to_tensor(
                ensure_spatial_tuple(spatial_shape, spatial_rank, "spatial_shape"),
                dtype=tf.int32,
            )

        if spatial_rank == 2:
            return self._resize_2d(tensor, key, target_shape)
        if spatial_rank == 3:
            return self._resize_3d(tensor, key, target_shape)
        raise ValueError(f"Resize supports only 2D or 3D tensors, got spatial rank {spatial_rank}.")

    def _resize_2d(self, tensor: tf.Tensor, key: str, spatial_shape: tf.Tensor) -> tf.Tensor:
        return tf.image.resize(tensor, spatial_shape, method=self.mode.get(key))

    def _resize_3d(self, tensor: tf.Tensor, key: str, spatial_shape: tf.Tensor) -> tf.Tensor:
        tensor = tensor[None, ...]
        resized = resize_volumes(
            tensor,
            spatial_shape[0],
            spatial_shape[1],
            spatial_shape[2],
            method=self.mode.get(key),
            align_corners=False,
        )
        return resized[0]

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

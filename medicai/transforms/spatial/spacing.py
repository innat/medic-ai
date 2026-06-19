import warnings
from typing import Mapping, Sequence

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import get_spatial_rank
from .resize import resize_volumes


class Spacing(KeyedTransform, InvertibleTransform):
    """Resample tensors to a target physical spacing.

    This transform remains explicitly 3D-only and expects channel-last sample
    tensors shaped like ``(D, H, W, C)``.
    """

    def __init__(
        self,
        keys: Sequence[str] = ("image", "label"),
        pixdim: Sequence[float] = (1.0, 1.0, 1.0),
        mode: str | Sequence[str] | Mapping[str, str] = ("trilinear", "nearest"),
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.pixdim = tuple(pixdim)

        if len(self.pixdim) != 3:
            raise ValueError(f"`pixdim` must be 3D for Spacing, got {len(self.pixdim)}D.")

        valid_modes = {"trilinear", "nearest"}
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
                    f"Invalid mode '{resize_mode}' for 3D input. "
                    f"Allowed: {sorted(valid_modes)} (key='{key}')."
                )

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        affine = bundle.meta.get("affine")
        if affine is not None:
            original_spacing = self.get_spacing_from_affine(affine)
        else:
            original_spacing = tf.constant((1.0, 1.0, 1.0), dtype=tf.float32)
            warnings.warn(
                "Affine matrix is not provided. Using default spacing (1.0, 1.0, 1.0)."
            )

        original_shapes = {}

        def apply_spacing(tensor: tf.Tensor, key: str) -> tf.Tensor:
            spatial_rank = get_spatial_rank(tensor)
            if spatial_rank != 3:
                raise ValueError(
                    f"Spacing currently supports only 3D tensors; got spatial rank {spatial_rank} "
                    f"for shape {tensor.shape}."
                )
            original_shapes[key] = tf.shape(tensor)[:3]
            return self.spacing_resample(
                tensor,
                original_spacing=original_spacing,
                desired_spacing=tf.constant(self.pixdim, dtype=tf.float32),
                mode=self.mode[key],
            )

        present_keys = self.apply_to_present_keys(bundle, apply_spacing)
        bundle.meta["pixdim"] = tf.constant(self.pixdim, dtype=tf.float32)
        self.record_transform(
            bundle,
            {
                "keys": list(present_keys),
                "pixdim": self.pixdim,
                "original_spacing": original_spacing,
                "original_shapes": original_shapes,
                "mode": {key: self.mode[key] for key in present_keys},
            },
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = self._get_last_spacing_trace(bundle)
        if trace is None:
            return bundle

        params = trace["params"]
        original_spacing = params.get("original_spacing")
        original_shapes = params.get("original_shapes", {})
        if original_spacing is None:
            return bundle

        def apply_inverse_spacing(tensor: tf.Tensor, key: str) -> tf.Tensor:
            if key not in original_shapes:
                return tensor
            resized = self.spacing_resample(
                tensor,
                original_spacing=tf.constant(self.pixdim, dtype=tf.float32),
                desired_spacing=tf.cast(original_spacing, tf.float32),
                mode=self.mode[key],
            )
            target_shape = original_shapes[key]
            return self._resize_to_shape(resized, target_shape, self.mode[key])

        present_keys = [key for key in params.get("keys", []) if key in bundle.data]
        self.apply_to_present_keys(bundle, apply_inverse_spacing, keys=present_keys)
        bundle.meta["pixdim"] = tf.cast(original_spacing, tf.float32)
        return bundle

    def get_spacing_from_affine(self, affine: tf.Tensor) -> tf.Tensor:
        """Calculate voxel spacing from a 4x4 affine matrix."""
        affine = tf.cast(affine, tf.float32)
        return tf.stack([tf.norm(affine[:3, i]) for i in range(3)], axis=0)

    def spacing_resample(
        self,
        tensor: tf.Tensor,
        original_spacing: tf.Tensor,
        desired_spacing: tf.Tensor,
        mode: str,
    ) -> tf.Tensor:
        """Resample one 3D tensor to the desired physical spacing."""
        scale = tf.cast(original_spacing, tf.float32) / tf.cast(desired_spacing, tf.float32)
        original_shape = tf.cast(tf.shape(tensor)[:3], tf.float32)
        new_shape = tf.cast(tf.round(original_shape * scale), tf.int32)
        return self._resize_to_shape(tensor, new_shape, mode)

    def _resize_to_shape(self, tensor: tf.Tensor, spatial_shape: tf.Tensor, mode: str) -> tf.Tensor:
        tensor = tensor[None, ...]
        resized = resize_volumes(
            tensor,
            spatial_shape[0],
            spatial_shape[1],
            spatial_shape[2],
            method=mode,
            align_corners=False,
        )
        return resized[0]

    def _get_last_spacing_trace(self, bundle: TensorBundle) -> dict | None:
        for entry in reversed(bundle.get_applied_transforms()):
            if entry.get("name") == type(self).__name__:
                return entry
        return None

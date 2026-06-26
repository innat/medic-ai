import warnings
from typing import Mapping, Sequence

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform, _pop_last_transform_trace
from ..tensor_bundle import TensorBundle
from ..utils import (
    SpatialResample,
    compute_destination_affine,
    compute_output_shape,
    get_spatial_rank,
    is_axis_aligned_affine,
    round_half_up,
    spacing_from_affine,
)
from .resize import resize_volumes


class Spacing(KeyedTransform, InvertibleTransform):
    """Resample tensors to a target physical spacing.

    ``Spacing`` uses voxel spacing derived from the input affine matrix to
    resample 3D channel-last tensors to a requested physical resolution. It is
    intended for volumetric medical images and corresponding labels that follow
    Medic-AI's ``(D, H, W, C)`` tensor convention.

    The transform reads the source spacing from ``bundle.meta["affine"]``. If
    no affine is provided, it falls back to spacing ``(1.0, 1.0, 1.0)`` and
    emits a warning. When affine metadata is available, both
    ``bundle.meta["pixdim"]`` and ``bundle.meta["affine"]`` are updated to
    describe the resampled output space.

    ``Spacing`` is invertible in the sense that it records the original voxel
    spacing, affine, and spatial shapes so :meth:`inverse` can resample tensors
    back to their prior shape and restore the original spatial metadata.
    Because this is a resampling operation, the restored tensor is not
    guaranteed to be numerically identical to the original input.

    Args:
        keys: Keys of tensors to resample.
        pixdim: Target voxel spacing given as ``(depth, height, width)``
            spacing values.
        interpolation: Interpolation mode specified as a single string, a sequence
            aligned with ``keys``, or a mapping from key to mode. Valid 3D
            modes are ``"trilinear"`` and ``"nearest"``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Resample a 3D image-label pair to isotropic spacing using a raw Python
        dictionary and affine metadata:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Spacing

            transform = Spacing(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                interpolation=("trilinear", "nearest"),
            )

            image = tf.random.normal((24, 128, 128, 1))
            label = tf.random.uniform((24, 128, 128, 1), maxval=2, dtype=tf.int32)
            affine = tf.constant(
                [
                    [0.0, 0.0, 2.5, 0.0],
                    [0.0, 0.8, 0.0, 0.0],
                    [3.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=tf.float32,
            )

            result = transform({"image": image, "label": label}, {"affine": affine})

        Resample a 3D image volume using a ``TensorBundle`` and restore its
        prior spacing:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Spacing, TensorBundle

            transform = Spacing(keys=["image"], pixdim=(1.0, 1.0, 1.0))

            image = tf.random.normal((24, 64, 64, 1))
            affine = tf.constant(
                [
                    [0.0, 0.0, 2.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=tf.float32,
            )

            bundle = TensorBundle({"image": image}, {"affine": affine})
            forward = transform(bundle)
            restored = transform.inverse(forward)

        Resample an image-label pair and then map it back to the original
        spatial shape:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Spacing, TensorBundle

            transform = Spacing(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                interpolation=("trilinear", "nearest"),
            )

            image = tf.random.normal((12, 32, 32, 1))
            label = tf.random.uniform((12, 32, 32, 1), maxval=2, dtype=tf.int32)
            affine = tf.eye(4)

            bundle = TensorBundle({"image": image, "label": label}, {"affine": affine})
            forward = transform(bundle)
            restored = transform.inverse(forward)

    Returns:
        ``TensorBundle``: The input bundle with resampled tensors, updated
        spacing metadata, and an invertible transform trace entry appended.

    Raises:
        ValueError: If ``pixdim`` is not length 3, if a tensor is not 3D
            spatially, or if an invalid interpolation mode is provided.
        TypeError: If ``interpolation`` is not a string, sequence, or mapping.
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
    """

    def __init__(
        self,
        keys: Sequence[str] = ("image", "label"),
        pixdim: Sequence[float] = (1.0, 1.0, 1.0),
        interpolation: str | Sequence[str] | Mapping[str, str] = ("trilinear", "nearest"),
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.pixdim = tuple(pixdim)

        if len(self.pixdim) != 3:
            raise ValueError(
                f"`pixdim` must be 3D for {type(self).__name__}, got {len(self.pixdim)}D."
            )
        if any(p <= 0 for p in self.pixdim):
            raise ValueError(
                f"All elements of `pixdim` must be strictly positive, got {self.pixdim}."
            )

        valid_modes = {"trilinear", "nearest"}
        if isinstance(interpolation, str):
            self.interpolation = {key: interpolation for key in keys}
        elif isinstance(interpolation, (tuple, list)):
            if len(keys) == 1 and len(interpolation) >= 1:
                self.interpolation = {keys[0]: interpolation[0]}
            elif len(interpolation) != len(keys):
                raise ValueError("Length of 'interpolation' must match length of 'keys'.")
            else:
                self.interpolation = dict(zip(keys, interpolation))
        elif isinstance(interpolation, dict):
            missing_keys = set(keys) - set(interpolation.keys())
            if missing_keys:
                raise ValueError(f"Missing interpolation mode for keys: {sorted(missing_keys)}")
            self.interpolation = dict(interpolation)
        else:
            raise TypeError("'interpolation' must be a string, tuple, list, or dict.")

        for key, resize_interpolation in self.interpolation.items():
            if resize_interpolation not in valid_modes:
                raise ValueError(
                    f"Invalid interpolation '{resize_interpolation}' for 3D input. "
                    f"Allowed: {sorted(valid_modes)} (key='{key}')."
                )
        self._spatial_resample = SpatialResample()

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        self._validate_bundle_is_3d(bundle)
        affine = bundle.meta.get("affine")
        if affine is not None:
            original_spacing = self.get_spacing_from_affine(affine)
        else:
            original_spacing = tf.constant((1.0, 1.0, 1.0), dtype=tf.float32)
            warnings.warn(
                "Affine matrix is not provided. Using default spacing (1.0, 1.0, 1.0).",
                stacklevel=2,
            )

        original_shapes = {}
        destination_affine = None
        target_pixdim = tf.constant(self.pixdim, dtype=tf.float32)
        use_fast_resize_path = False
        if affine is not None:
            affine = tf.cast(affine, tf.float32)
            destination_affine = compute_destination_affine(affine, target_pixdim)
            use_fast_resize_path = bool(tf.get_static_value(is_axis_aligned_affine(affine)))

        def apply_spacing(tensor: tf.Tensor, key: str) -> tf.Tensor:
            spatial_rank = get_spatial_rank(tensor)
            if spatial_rank != 3:
                raise ValueError(
                    f"{type(self).__name__} currently supports only 3D tensors; got spatial rank "
                    f"{spatial_rank} "
                    f"for shape {tensor.shape}."
                )
            original_shapes[key] = tf.shape(tensor)[:3]
            if destination_affine is not None and not use_fast_resize_path:
                output_shape = compute_output_shape(
                    input_shape=original_shapes[key],
                    src_affine=affine,
                    dst_affine=destination_affine,
                    align_corners=False,
                )
                return self._spatial_resample(
                    tensor=tensor,
                    src_affine=affine,
                    dst_affine=destination_affine,
                    output_shape=output_shape,
                    interpolation=self.interpolation[key],
                    padding_mode="constant",
                    fill_value=0.0,
                )
            return self.spacing_resample(
                tensor,
                original_spacing=original_spacing,
                desired_spacing=target_pixdim,
                interpolation=self.interpolation[key],
            )

        present_keys = self.apply_to_present_keys(bundle, apply_spacing)
        bundle.meta["pixdim"] = target_pixdim
        original_affine = None
        if affine is not None:
            original_affine = tf.identity(tf.cast(affine, tf.float32))
            bundle.meta["affine"] = tf.cast(destination_affine, tf.float32)
        self.record_transform(
            bundle,
            {
                "keys": list(present_keys),
                "pixdim": self.pixdim,
                "original_spacing": original_spacing,
                "original_affine": original_affine,
                "output_affine": (
                    tf.identity(destination_affine) if destination_affine is not None else None
                ),
                "original_shapes": original_shapes,
                "interpolation": {key: self.interpolation[key] for key in present_keys},
                "used_fast_resize_path": use_fast_resize_path,
            },
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = self._get_last_spacing_trace(bundle)
        if trace is None:
            return bundle

        params = trace["params"]
        original_spacing = params.get("original_spacing")
        original_affine = params.get("original_affine")
        output_affine = params.get("output_affine")
        original_shapes = params.get("original_shapes", {})
        interpolation = params.get("interpolation", self.interpolation)
        used_fast_resize_path = params.get("used_fast_resize_path", False)
        if original_spacing is None:
            return bundle

        if original_affine is not None and output_affine is not None and not used_fast_resize_path:
            tensors_by_shape: dict[tuple[int, ...], dict[str, tf.Tensor]] = {}
            interpolation_by_shape: dict[tuple[int, ...], dict[str, str]] = {}
            can_group = True

            for key in params.get("keys", []):
                if key not in bundle.data:
                    if self.allow_missing_keys:
                        continue
                    raise KeyError(f"Key '{key}' not found in input data.")
                if key not in original_shapes:
                    continue
                target_shape = original_shapes[key]
                static_shape = tf.get_static_value(target_shape)
                if static_shape is None:
                    can_group = False
                    break
                shape_key = tuple(int(v) for v in static_shape)
                tensors_by_shape.setdefault(shape_key, {})[key] = bundle.data[key]
                interpolation_by_shape.setdefault(shape_key, {})[key] = interpolation[key]

            if can_group:
                for shape_key, tensors in tensors_by_shape.items():
                    output_shape = tf.constant(shape_key, dtype=tf.int32)
                    restored = self._spatial_resample.resample_many(
                        tensors=tensors,
                        src_affine=output_affine,
                        dst_affine=original_affine,
                        output_shape=output_shape,
                        interpolation=interpolation_by_shape[shape_key],
                        padding_mode="constant",
                        fill_value=0.0,
                    )
                    for key, value in restored.items():
                        bundle.data[key] = value

                bundle.meta["pixdim"] = tf.cast(original_spacing, tf.float32)
                if original_affine is not None:
                    bundle.meta["affine"] = tf.cast(original_affine, tf.float32)
                return bundle

            for key in params.get("keys", []):
                if key not in bundle.data:
                    if self.allow_missing_keys:
                        continue
                    raise KeyError(f"Key '{key}' not found in input data.")
                if key not in original_shapes:
                    continue
                bundle.data[key] = self._spatial_resample(
                    tensor=bundle.data[key],
                    src_affine=output_affine,
                    dst_affine=original_affine,
                    output_shape=original_shapes[key],
                    interpolation=interpolation[key],
                    padding_mode="constant",
                    fill_value=0.0,
                )

            bundle.meta["pixdim"] = tf.cast(original_spacing, tf.float32)
            if original_affine is not None:
                bundle.meta["affine"] = tf.cast(original_affine, tf.float32)
            return bundle

        def apply_inverse_spacing(tensor: tf.Tensor, key: str) -> tf.Tensor:
            if key not in original_shapes:
                return tensor
            target_shape = original_shapes[key]
            return self._resize_to_shape(tensor, target_shape, interpolation[key])

        self.apply_to_present_keys(
            bundle,
            apply_inverse_spacing,
            keys=params.get("keys", []),
        )
        bundle.meta["pixdim"] = tf.cast(original_spacing, tf.float32)
        if original_affine is not None:
            bundle.meta["affine"] = tf.cast(original_affine, tf.float32)
        return bundle

    def _validate_bundle_is_3d(self, bundle: TensorBundle) -> None:
        """Validate that present tensors use Medic-AI 3D sample layout."""
        for key in self.keys:
            if key not in bundle.data:
                continue
            tensor = bundle.data[key]
            spatial_rank = get_spatial_rank(tensor)
            if spatial_rank != 3:
                raise ValueError(
                    f"{type(self).__name__} supports only 3D channel-last tensors shaped "
                    f"(D, H, W, C). Key '{key}' has shape {tensor.shape} with spatial rank "
                    f"{spatial_rank}."
                )

    def get_spacing_from_affine(self, affine: tf.Tensor) -> tf.Tensor:
        """Calculate voxel spacing from a 4x4 affine matrix."""
        return spacing_from_affine(affine)

    def spacing_resample(
        self,
        tensor: tf.Tensor,
        original_spacing: tf.Tensor,
        desired_spacing: tf.Tensor,
        interpolation: str,
    ) -> tf.Tensor:
        """Resample one 3D tensor to the desired physical spacing."""
        scale = tf.cast(original_spacing, tf.float32) / tf.cast(desired_spacing, tf.float32)
        original_shape = tf.cast(tf.shape(tensor)[:3], tf.float32)
        new_shape = tf.maximum(
            tf.cast(round_half_up(original_shape * scale), tf.int32),
            1,
        )
        return self._resize_to_shape(tensor, new_shape, interpolation)

    def _resize_to_shape(
        self, tensor: tf.Tensor, spatial_shape: tf.Tensor, interpolation: str
    ) -> tf.Tensor:
        tensor = tensor[None, ...]
        resized = resize_volumes(
            tensor,
            spatial_shape[0],
            spatial_shape[1],
            spatial_shape[2],
            method=interpolation,
            align_corners=False,
        )
        return resized[0]

    def _get_last_spacing_trace(self, bundle: TensorBundle) -> dict | None:
        return _pop_last_transform_trace(bundle, type(self).__name__)

from numbers import Integral
from typing import Sequence

import keras
from keras import ops

from ..base import RandomTransform, _apply_if_applied, _normalize_keys
from ..tensor_bundle import TensorBundle
from ..utils import (
    get_input_layout_info,
    get_tensor_rank,
    resolve_input_layout,
    validate_tensor_matches_layout,
)

_FILL_MODES = {"constant", "gaussian"}
_DEFAULT_PROB = 0.5
_DEFAULT_FILL_MODE = "constant"
_DEFAULT_FILL_VALUE = 0.0
_GAUSSIAN_NOISE_STD = 1.0


class RandomCutOut(RandomTransform):
    """Apply random CutOut augmentation to 2D or 3D image tensors.

    ``RandomCutOut`` samples one or more rectangular masks and replaces the
    corresponding image regions with either a constant value or Gaussian
    noise. For 2D inputs, each mask removes pixels. For 3D inputs, each mask
    removes a bounded D-H-W cuboid; no individual slice mode is used.

    Args:
        keys: A single key containing the image tensor to modify. The key may
            be provided as a one-element sequence or a string.
        mask_size: Spatial mask size for each cutout window. Use ``(H, W)``
            for 2D layouts and ``(D, H, W)`` for 3D layouts.
        num_cuts: Number of cutout windows to sample.
        prob: Probability of applying cutout.
        fill_mode: Either ``"constant"`` or ``"gaussian"``. Gaussian fill
            uses standard-normal noise rescaled to the input image value range.
        fill_value: Constant fill value used when ``fill_mode="constant"``.
        input_layout: Channel-last tensor layout. Supported values are
            ``"HWC"``, ``"DHWC"``, ``"BHWC"``, and ``"BDHWC"``.
            In batch layouts, the Bernoulli apply decision and cutout mask are
            sampled independently for each sample.
        seed: Optional random seed. Supports ``None``, an integer seed, or a
            ``keras.random.SeedGenerator``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:

        TensorFlow backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import RandomCutOut

            transform = RandomCutOut(
                keys=["image"],
                mask_size=(16, 16),
                num_cuts=2,
                prob=0.5,
                input_layout="HWC",
            )

            image = tf.random.normal((64, 64, 1))
            label = tf.cast(image > 0, tf.int32)
            result = transform({"image": image, "label": label})
            output = result["image"]
            print(output.shape)

        JAX backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            import jax.numpy as jnp
            from medicai.transforms import RandomCutOut

            transform = RandomCutOut(
                keys=["image"],
                mask_size=(8, 16, 16),
                num_cuts=2,
                prob=0.5,
                input_layout="DHWC",
            )

            image = jax.random.normal(
                jax.random.PRNGKey(7), shape=(32, 64, 64, 1)
            )
            label = jnp.asarray(image > 0, dtype=jnp.int32)
            result = transform({"image": image, "label": label})
            output = result["image"]
            print(output.shape)

        Torch backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import RandomCutOut

            transform = RandomCutOut(
                keys=["image"],
                mask_size=(16, 16),
                num_cuts=2,
                prob=0.5,
                input_layout="BHWC",
            )

            torch.manual_seed(7)
            image = torch.randn((2, 64, 64, 1))
            label = (image > 0).to(torch.int32)
            result = transform({"image": image, "label": label})
            output = result["image"]
            print(output.shape)
    """

    def __init__(
        self,
        keys: Sequence[str] | str,
        mask_size: Sequence[int],
        num_cuts: int,
        prob: float = _DEFAULT_PROB,
        fill_mode: str = _DEFAULT_FILL_MODE,
        fill_value: float = _DEFAULT_FILL_VALUE,
        *,
        input_layout: str,
        seed: int | keras.random.SeedGenerator | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob, seed=seed)

        normalized_keys = _normalize_keys(keys)
        if len(normalized_keys) != 1:
            raise ValueError(
                "`keys` must contain exactly one image key. " f"Got length {len(keys)}."
            )

        if not isinstance(num_cuts, Integral) or isinstance(num_cuts, bool) or num_cuts <= 0:
            raise ValueError("`num_cuts` must be a positive integer.")

        if fill_mode not in _FILL_MODES:
            raise ValueError(
                f'`fill_mode` must be either "gaussian" or "constant". Got {fill_mode}.'
            )
        self.image_key = normalized_keys[0]
        self.num_cuts = int(num_cuts)
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            transform_name=type(self).__name__,
        )
        self.layout_info = get_input_layout_info(self.input_layout)
        self.mask_size = self._validate_mask_size(mask_size)
        self.allow_missing_keys = allow_missing_keys

    def _validate_mask_size(self, mask_size: Sequence[int]) -> tuple[int, ...]:
        """Validate a spatial mask size against the configured input layout."""
        spatial_rank = self.layout_info.spatial_rank
        if not isinstance(mask_size, (list, tuple)) or len(mask_size) != spatial_rank:
            expected = "(H, W)" if spatial_rank == 2 else "(D, H, W)"
            raise ValueError(
                f"`mask_size` must be a sequence of {spatial_rank} integers: {expected}."
            )

        if not all(
            isinstance(value, Integral) and not isinstance(value, bool) and value > 0
            for value in mask_size
        ):
            raise ValueError("All values in `mask_size` must be positive integers.")
        return tuple(int(value) for value in mask_size)

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        params = self.get_random_params(bundle)
        if params["skip"]:
            return bundle
        return self.apply_with_params(bundle, params)

    def get_random_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Sample one Bernoulli decision for the selected image key."""
        if self.image_key not in bundle.data:
            if self.allow_missing_keys:
                return {"skip": True}
            raise KeyError(f"Key '{self.image_key}' not found in input data.")

        image = bundle.data[self.image_key]
        layout = validate_tensor_matches_layout(
            image,
            self.input_layout,
            transform_name=type(self).__name__,
        )
        spatial_rank = layout.spatial_rank

        if self.layout_info.batched:
            batch_size = ops.shape(image)[0]
            should_apply = (
                self.random_uniform(
                    shape=(batch_size,),
                    minval=0.0,
                    maxval=1.0,
                    dtype="float32",
                )
                < self.prob
            )
        else:
            should_apply = self.sample_should_apply()

        centers = self._sample_cutout_centers(image, spatial_rank)
        noise = (
            self.random_normal(
                shape=ops.shape(image),
                stddev=_GAUSSIAN_NOISE_STD,
                dtype=image.dtype,
            )
            if self.fill_mode == "gaussian"
            else ops.zeros_like(image)
        )
        return {
            "skip": False,
            "image": image,
            "spatial_rank": spatial_rank,
            "should_apply": should_apply,
            "centers": centers,
            "noise": noise,
        }

    def apply_with_params(
        self,
        bundle: TensorBundle,
        params: dict[str, object],
    ) -> TensorBundle:
        """Apply the sampled cutout configuration to the selected image key."""
        if self.layout_info.batched:
            bundle.data[self.image_key] = self.apply_batch_cutout(
                params["image"],
                params["spatial_rank"],
                params["centers"],
                params["noise"],
                params["should_apply"],
            )
        else:
            bundle.data[self.image_key] = _apply_if_applied(
                params["should_apply"],
                lambda: self.apply_sample_cutout(
                    params["image"],
                    params["spatial_rank"],
                    params["centers"],
                    params["noise"],
                ),
                lambda: params["image"],
            )
        self.record_random_transform(
            bundle,
            params=self.build_trace_params(params),
            applied=params["should_apply"],
            kernel="cutout_mask",
        )
        return bundle

    def build_trace_params(self, params: dict[str, object]) -> dict[str, object]:
        """Build random trace metadata for the current cutout operation."""
        return {
            "keys": [self.image_key],
            "mask_size": self.mask_size,
            "num_cuts": self.num_cuts,
            "fill_mode": self.fill_mode,
            "should_apply": params["should_apply"],
            "input_layout": self.input_layout,
        }

    def apply_sample_cutout(
        self,
        image,
        spatial_rank: int,
        centers,
        noise,
    ):
        """Apply cutout to one sample tensor using sampled mask parameters."""
        mask = self.generate_cutout_mask(image, spatial_rank, centers)
        return self.apply_cutout(image, mask, noise)

    def apply_batch_cutout(
        self,
        images,
        spatial_rank: int,
        centers,
        noise,
        should_apply,
    ):
        """Apply cutout independently to each sample of a batch."""
        mask = self.generate_batch_cutout_mask(images, spatial_rank, centers)
        cutout = self.apply_cutout(images, mask, noise)
        apply_shape = [ops.shape(images)[0]] + [1] * (get_tensor_rank(images) - 1)
        return ops.where(
            ops.reshape(ops.cast(should_apply, "bool"), apply_shape),
            cutout,
            images,
        )

    def apply_cutout(self, image, mask, noise):
        """Apply a generated cutout mask to the image tensor."""
        mask_bool = ops.cast(mask, "bool")
        if self.fill_mode == "gaussian":
            reduction_axes = (
                tuple(range(1, get_tensor_rank(image))) if self.layout_info.batched else None
            )
            im_min = ops.min(image, axis=reduction_axes, keepdims=True)
            im_max = ops.max(image, axis=reduction_axes, keepdims=True)
            nz_min = ops.min(noise, axis=reduction_axes, keepdims=True)
            nz_max = ops.max(noise, axis=reduction_axes, keepdims=True)
            fill = (im_max - im_min) * (noise - nz_min) / (nz_max - nz_min + 1e-8) + im_min
        else:
            fill = ops.zeros_like(image) + ops.cast(self.fill_value, image.dtype)
        return ops.where(mask_bool, image, fill)

    def generate_cutout_mask(self, volume, spatial_rank: int, centers):
        """Generate a cutout mask for a 2D or 3D sample tensor."""
        if spatial_rank == 2:
            return self._cutout_mask_2d(volume, centers)
        return self._cutout_mask_volume_wise(volume, centers)

    def generate_batch_cutout_mask(self, images, spatial_rank: int, centers):
        """Generate all batch masks through broadcasted spatial comparisons."""
        if spatial_rank == 2:
            return self._cutout_mask_2d_batch(images, centers)

        return self._cutout_mask_volume_batch(images, centers)

    def _sample_cutout_centers(self, image, spatial_rank: int):
        """Sample all cutout centers before conditional application."""
        shape = ops.shape(image)
        if self.layout_info.batched:
            batch_size = shape[0]
            prefix = [batch_size, self.num_cuts]
        else:
            prefix = [self.num_cuts]
        if all(isinstance(value, int) for value in prefix):
            center_shape = tuple(prefix) + (spatial_rank,)
        else:
            prefix = ops.stack(
                [ops.cast(value, "int32") for value in prefix],
                axis=0,
            )
            center_shape = ops.concatenate(
                [prefix, ops.convert_to_tensor([spatial_rank], dtype="int32")], axis=0
            )
        if spatial_rank == 3:
            spatial_shape = ops.stack([shape[-4], shape[-3], shape[-2]])
        else:
            spatial_shape = ops.stack([shape[-3], shape[-2]])
        random_unit = self.random_uniform(
            shape=center_shape,
            minval=0.0,
            maxval=1.0,
            dtype="float32",
        )
        return ops.cast(ops.floor(random_unit * ops.cast(spatial_shape, "float32")), "int32")

    def _cutout_mask_2d(self, image, centers):
        shape = ops.shape(image)
        height, width = shape[0], shape[1]
        mask_h, mask_w = self.mask_size
        y_lo = mask_h // 2
        y_hi = mask_h - y_lo
        x_lo = mask_w // 2
        x_hi = mask_w - x_lo
        y = ops.arange(height)[None, :]
        x = ops.arange(width)[None, :]
        cy, cx = centers[:, 0], centers[:, 1]
        y_mask = (y >= cy[:, None] - y_lo) & (y < cy[:, None] + y_hi)
        x_mask = (x >= cx[:, None] - x_lo) & (x < cx[:, None] + x_hi)
        cut_any = ops.any(y_mask[:, :, None] & x_mask[:, None, :], axis=0)
        return ops.logical_not(cut_any)[..., None]

    def _cutout_mask_2d_batch(self, images, centers):
        shape = ops.shape(images)
        height, width = shape[1], shape[2]
        mask_h, mask_w = self.mask_size
        y_lo = mask_h // 2
        y_hi = mask_h - y_lo
        x_lo = mask_w // 2
        x_hi = mask_w - x_lo
        y = ops.arange(height)[None, None, :]
        x = ops.arange(width)[None, None, :]
        cy, cx = centers[..., 0], centers[..., 1]
        y_mask = (y >= cy[..., None] - y_lo) & (y < cy[..., None] + y_hi)
        x_mask = (x >= cx[..., None] - x_lo) & (x < cx[..., None] + x_hi)
        cut_any = ops.any(y_mask[..., :, None] & x_mask[..., None, :], axis=1)
        return ops.logical_not(cut_any)[..., None]

    def _cutout_mask_volume_wise(self, volume, centers):
        shape = ops.shape(volume)
        depth, height, width = shape[0], shape[1], shape[2]
        mask_d, mask_h, mask_w = self.mask_size
        z_lo = mask_d // 2
        z_hi = mask_d - z_lo
        y_lo = mask_h // 2
        y_hi = mask_h - y_lo
        x_lo = mask_w // 2
        x_hi = mask_w - x_lo
        z = ops.arange(depth)[None, :]
        y = ops.arange(height)[None, :]
        x = ops.arange(width)[None, :]
        cz, cy, cx = centers[:, 0], centers[:, 1], centers[:, 2]
        z_mask = (z >= cz[:, None] - z_lo) & (z < cz[:, None] + z_hi)
        y_mask = (y >= cy[:, None] - y_lo) & (y < cy[:, None] + y_hi)
        x_mask = (x >= cx[:, None] - x_lo) & (x < cx[:, None] + x_hi)
        cut_any = ops.any(
            z_mask[:, :, None, None] & y_mask[:, None, :, None] & x_mask[:, None, None, :],
            axis=0,
        )
        return ops.logical_not(cut_any)[..., None]

    def _cutout_mask_volume_batch(self, volumes, centers):
        shape = ops.shape(volumes)
        depth, height, width = shape[1], shape[2], shape[3]
        mask_d, mask_h, mask_w = self.mask_size
        z_lo = mask_d // 2
        z_hi = mask_d - z_lo
        y_lo = mask_h // 2
        y_hi = mask_h - y_lo
        x_lo = mask_w // 2
        x_hi = mask_w - x_lo
        z = ops.arange(depth)[None, None, :]
        y = ops.arange(height)[None, None, :]
        x = ops.arange(width)[None, None, :]
        cz, cy, cx = centers[..., 0], centers[..., 1], centers[..., 2]
        z_mask = (z >= cz[..., None] - z_lo) & (z < cz[..., None] + z_hi)
        y_mask = (y >= cy[..., None] - y_lo) & (y < cy[..., None] + y_hi)
        x_mask = (x >= cx[..., None] - x_lo) & (x < cx[..., None] + x_hi)
        cut_any = ops.any(
            z_mask[..., :, None, None] & y_mask[..., None, :, None] & x_mask[..., None, None, :],
            axis=1,
        )
        return ops.logical_not(cut_any)[..., None]

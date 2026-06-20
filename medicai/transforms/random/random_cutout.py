from typing import Sequence

import tensorflow as tf

from ..base import RandomTransform
from ..tensor_bundle import TensorBundle
from ..utils import get_spatial_rank


class RandomCutOut(RandomTransform):
    """Apply random CutOut augmentation to a volumetric image tensor.

    ``RandomCutOut`` samples one or more rectangular masks and replaces the
    corresponding image regions with either a constant value or Gaussian
    noise. It currently operates on 3D channel-last samples shaped
    ``(D, H, W, C)`` and uses the paired label tensor to optionally avoid
    masking invalid regions.

    Args:
        keys: Two keys containing the image tensor and label tensor.
        mask_size: Height-width mask size for each cutout window.
        num_cuts: Number of cutout windows to sample.
        prob: Probability of applying cutout.
        fill_mode: Either ``"constant"`` or ``"gaussian"``.
        fill_value: Constant fill value used when ``fill_mode="constant"``.
        gaussian_std: Standard deviation for Gaussian fill noise.
        invalid_label: Optional label value marking invalid regions.
        cutout_mode: Either ``"slice"`` for slice-wise masks or ``"volume"``
            for the same mask across all depth slices.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Apply random cutout to a 3D image-label pair using a raw Python
        dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomCutOut

            transform = RandomCutOut(
                keys=["image", "label"],
                mask_size=(16, 16),
                num_cuts=2,
                prob=0.5,
            )

            image = tf.random.normal((32, 64, 64, 1))
            label = tf.cast(image > 0, tf.int32)
            result = transform({"image": image, "label": label})
            output = result["image"]
            print(output.shape)

        Apply random cutout to a 3D image-label pair stored in a
        ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomCutOut, TensorBundle

            transform = RandomCutOut(
                keys=["image", "label"],
                mask_size=(16, 16),
                num_cuts=2,
                prob=0.5,
            )

            image = tf.random.normal((32, 64, 64, 1))
            label = tf.cast(image > 0, tf.int32)
            bundle = TensorBundle({"image": image, "label": label})
            result = transform(bundle)
            output = result["image"]
            print(output.shape)
    """

    def __init__(
        self,
        keys: Sequence[str],
        mask_size: Sequence[int],
        num_cuts: int,
        prob: float = 0.5,
        fill_mode: str = "constant",
        fill_value: float = 0.0,
        gaussian_std: float = 0.1,
        invalid_label=None,
        cutout_mode: str = "volume",
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob)
        if len(keys) != 2:
            raise ValueError(
                "`keys` must have length 2 and should contain image and label keys. "
                f"Got length {len(keys)}."
            )
        if not isinstance(mask_size, (list, tuple)) or len(mask_size) != 2:
            raise ValueError("`mask_size` must be a sequence of two integers: (height, width).")
        if not all(isinstance(m, int) and m > 0 for m in mask_size):
            raise ValueError("All values in `mask_size` must be positive integers.")
        if num_cuts <= 0:
            raise ValueError("`num_cuts` must be a positive integer.")
        if fill_mode not in {"gaussian", "constant"}:
            raise ValueError(f'`fill_mode` must be either "gaussian" or "constant". Got {fill_mode}.')
        if cutout_mode not in {"slice", "volume"}:
            raise ValueError(f'`cutout_mode` must be one of {{"slice", "volume"}}. Got {cutout_mode}.')

        self.image_key = keys[0]
        self.label_key = keys[1]
        self.mask_size = tuple(mask_size)
        self.num_cuts = num_cuts
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.gaussian_std = gaussian_std
        self.invalid_label = invalid_label
        self.cutout_mode = cutout_mode
        self.allow_missing_keys = allow_missing_keys

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        if self.image_key not in bundle.data or self.label_key not in bundle.data:
            if self.allow_missing_keys:
                return bundle
            missing = self.image_key if self.image_key not in bundle.data else self.label_key
            raise KeyError(f"Key '{missing}' not found in input data.")

        image = bundle.data[self.image_key]
        label = bundle.data[self.label_key]
        spatial_rank = get_spatial_rank(image if image.shape.rank == 4 else image[..., None])
        if spatial_rank != 3:
            raise ValueError(
                f"RandCutOut currently supports only 3D inputs; got spatial rank {spatial_rank} "
                f"for shape {image.shape}."
            )

        should_apply = self.sample_should_apply()
        mask = self.generate_cutout_mask(image, label)

        bundle.data[self.image_key] = tf.cond(
            should_apply,
            lambda: self.apply_cutout(image, mask),
            lambda: image,
        )
        self.record_random_transform(
            bundle,
            params={
                "keys": [self.image_key, self.label_key],
                "mask_size": self.mask_size,
                "num_cuts": self.num_cuts,
                "fill_mode": self.fill_mode,
                "cutout_mode": self.cutout_mode,
            },
            applied=should_apply,
            kernel="cutout_mask",
        )
        return bundle

    def apply_cutout(self, image: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Apply a generated cutout mask to the image tensor."""
        mask_bool = tf.cast(mask, tf.bool)
        if self.fill_mode == "gaussian":
            noise = tf.random.normal(tf.shape(image), stddev=self.gaussian_std, dtype=image.dtype)
            im_min = tf.reduce_min(image)
            im_max = tf.reduce_max(image)
            nz_min = tf.reduce_min(noise)
            nz_max = tf.reduce_max(noise)
            fill = (im_max - im_min) * (noise - nz_min) / (nz_max - nz_min + 1e-8) + im_min
        else:
            fill = tf.fill(tf.shape(image), tf.cast(self.fill_value, image.dtype))
        return tf.where(mask_bool, image, fill)

    def generate_cutout_mask(self, volume: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        """Generate a cutout mask for a 3D sample tensor."""
        if label.shape.rank == 4:
            label = label[..., 0]
        if volume.shape.rank == 3:
            volume = volume[..., None]

        if self.cutout_mode == "slice":
            return self._cutout_mask_slice_wise(volume, label)
        return self._cutout_mask_volume_wise(volume, label)

    def _cutout_mask_slice_wise(self, volume: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(volume)
        depth, height, width = shape[0], shape[1], shape[2]
        mask_h, mask_w = self.mask_size
        cutout_mask = tf.ones((depth, height, width), tf.float32)
        valid_mask = (
            tf.ones((depth, height, width), tf.float32)
            if self.invalid_label is None
            else tf.cast(label != self.invalid_label, tf.float32)
        )

        for _ in range(self.num_cuts):
            cy = tf.random.uniform([depth], 0, height, tf.int32)
            cx = tf.random.uniform([depth], 0, width, tf.int32)

            y = tf.range(height)[None, :]
            x = tf.range(width)[None, :]
            y_mask = (y >= cy[:, None] - mask_h // 2) & (y < cy[:, None] + mask_h // 2)
            x_mask = (x >= cx[:, None] - mask_w // 2) & (x < cx[:, None] + mask_w // 2)
            rect = tf.cast(y_mask[:, :, None] & x_mask[:, None, :], tf.float32) * valid_mask
            cutout_mask *= 1.0 - rect

        return cutout_mask[..., None]

    def _cutout_mask_volume_wise(self, volume: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(volume)
        depth, height, width = shape[0], shape[1], shape[2]
        mask_h, mask_w = self.mask_size
        cutout_mask = tf.ones((depth, height, width), tf.float32)
        valid_mask = (
            tf.ones((depth, height, width), tf.float32)
            if self.invalid_label is None
            else tf.cast(label != self.invalid_label, tf.float32)
        )

        for _ in range(self.num_cuts):
            cy = tf.random.uniform([], 0, height, tf.int32)
            cx = tf.random.uniform([], 0, width, tf.int32)
            y = tf.range(height)
            x = tf.range(width)
            y_mask = (y >= cy - mask_h // 2) & (y < cy + mask_h // 2)
            x_mask = (x >= cx - mask_w // 2) & (x < cx + mask_w // 2)
            rect_hw = tf.cast(y_mask[:, None] & x_mask[None, :], tf.float32)
            rect = tf.broadcast_to(rect_hw[None, ...], (depth, height, width)) * valid_mask
            cutout_mask *= 1.0 - rect

        return cutout_mask[..., None]

from typing import Sequence

import keras
import tensorflow as tf

from ..base import RandomTransform
from ..tensor_bundle import TensorBundle
from ..utils import validate_input_mode, validate_layout


class RandomCutOut(RandomTransform):
    """Apply random CutOut augmentation to 2D or 3D image tensors.

    ``RandomCutOut`` samples one or more rectangular masks and replaces the
    corresponding image regions with either a constant value or Gaussian
    noise.

    Depending on ``input_mode``, it supports:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

    The paired label tensor can optionally be used to avoid masking invalid
    regions.

    Args:
        keys: Two keys containing the image tensor and label tensor.
        mask_size: Height-width mask size for each cutout window.
        num_cuts: Number of cutout windows to sample.
        prob: Probability of applying cutout.
        fill_mode: Either ``"constant"`` or ``"gaussian"``.
        fill_value: Constant fill value used when ``fill_mode="constant"``.
        gaussian_std: Standard deviation for Gaussian fill noise.
        input_mode: Either ``"sample"`` for ``(H, W, C)`` / ``(D, H, W, C)``
            tensors, or ``"batch"`` for ``(B, H, W, C)`` / ``(B, D, H, W, C)``
            tensors. In batch mode, one Bernoulli apply decision is sampled
            for the full batch, while cutout masks are generated per sample.
        seed: Optional random seed. Supports ``None``, an integer seed, or a
            ``keras.random.SeedGenerator``.
        invalid_label: Optional label value marking invalid regions.
        cutout_mode: Either ``"slice"`` for slice-wise masks or ``"volume"``
            for the same mask across all depth slices. For 2D inputs, both
            modes behave identically because there is no depth axis.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Apply random cutout to a 2D image-label pair using a raw Python
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

            image = tf.random.normal((64, 64, 1))
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
        input_mode: str = "sample",
        seed: int | keras.random.SeedGenerator | None = None,
        invalid_label=None,
        cutout_mode: str = "volume",
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob, seed=seed)
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
            raise ValueError(
                f'`fill_mode` must be either "gaussian" or "constant". Got {fill_mode}.'
            )
        if cutout_mode not in {"slice", "volume"}:
            raise ValueError(
                f'`cutout_mode` must be one of {{"slice", "volume"}}. Got {cutout_mode}.'
            )

        self.image_key = keys[0]
        self.label_key = keys[1]
        self.mask_size = tuple(mask_size)
        self.num_cuts = num_cuts
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.gaussian_std = gaussian_std
        self.input_mode = validate_input_mode(input_mode, transform_name=type(self).__name__)
        self.invalid_label = invalid_label
        self.cutout_mode = cutout_mode
        self.allow_missing_keys = allow_missing_keys

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        params = self.get_random_params(bundle)
        if params["skip"]:
            return bundle
        return self.apply_with_params(bundle, params)

    def get_random_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Sample one Bernoulli decision shared across the selected keys."""
        if self.image_key not in bundle.data or self.label_key not in bundle.data:
            if self.allow_missing_keys:
                return {"skip": True}
            missing = self.image_key if self.image_key not in bundle.data else self.label_key
            raise KeyError(f"Key '{missing}' not found in input data.")

        image = bundle.data[self.image_key]
        label = bundle.data[self.label_key]
        layout = validate_layout(
            image,
            input_mode=self.input_mode,
            allowed_spatial_ranks=(2, 3),
            transform_name=type(self).__name__,
        )
        validate_layout(
            label,
            input_mode=self.input_mode,
            allowed_spatial_ranks=(layout.spatial_rank,),
            transform_name=type(self).__name__,
        )
        spatial_rank = layout.spatial_rank

        should_apply = self.sample_should_apply()
        return {
            "skip": False,
            "image": image,
            "label": label,
            "spatial_rank": spatial_rank,
            "should_apply": should_apply,
            "input_mode": self.input_mode,
        }

    def apply_with_params(
        self,
        bundle: TensorBundle,
        params: dict[str, object],
    ) -> TensorBundle:
        """Apply the sampled cutout configuration to the selected image key."""
        if self.input_mode == "batch":
            bundle.data[self.image_key] = tf.cond(
                params["should_apply"],
                lambda: self.apply_batch_cutout(
                    params["image"],
                    params["label"],
                    params["spatial_rank"],
                ),
                lambda: params["image"],
            )
        else:
            bundle.data[self.image_key] = tf.cond(
                params["should_apply"],
                lambda: self.apply_sample_cutout(
                    params["image"],
                    params["label"],
                    params["spatial_rank"],
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
            "keys": [self.image_key, self.label_key],
            "mask_size": self.mask_size,
            "num_cuts": self.num_cuts,
            "fill_mode": self.fill_mode,
            "cutout_mode": self.cutout_mode,
            "input_mode": params["input_mode"],
        }

    def apply_sample_cutout(
        self,
        image: tf.Tensor,
        label: tf.Tensor,
        spatial_rank: int,
    ) -> tf.Tensor:
        """Apply cutout to one sample tensor using a freshly generated mask."""
        mask = self.generate_cutout_mask(image, label, spatial_rank)
        return self.apply_cutout(image, mask)

    def apply_batch_cutout(
        self,
        images: tf.Tensor,
        labels: tf.Tensor,
        spatial_rank: int,
    ) -> tf.Tensor:
        """Apply cutout independently to each sample of a batch."""
        return tf.map_fn(
            lambda elems: self.apply_sample_cutout(elems[0], elems[1], spatial_rank),
            (images, labels),
            fn_output_signature=tf.TensorSpec(shape=images.shape[1:], dtype=images.dtype),
        )

    def apply_cutout(self, image: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Apply a generated cutout mask to the image tensor."""
        mask_bool = tf.cast(mask, tf.bool)
        if self.fill_mode == "gaussian":
            noise = self.random_normal(
                shape=tf.shape(image),
                stddev=self.gaussian_std,
                dtype=image.dtype,
            )
            im_min = tf.reduce_min(image)
            im_max = tf.reduce_max(image)
            nz_min = tf.reduce_min(noise)
            nz_max = tf.reduce_max(noise)
            fill = (im_max - im_min) * (noise - nz_min) / (nz_max - nz_min + 1e-8) + im_min
        else:
            fill = tf.fill(tf.shape(image), tf.cast(self.fill_value, image.dtype))
        return tf.where(mask_bool, image, fill)

    def generate_cutout_mask(
        self, volume: tf.Tensor, label: tf.Tensor, spatial_rank: int
    ) -> tf.Tensor:
        """Generate a cutout mask for a 2D or 3D sample tensor."""
        if spatial_rank == 2:
            if label.shape.rank == 3:
                label = label[..., 0]
            return self._cutout_mask_2d(volume, label)

        if label.shape.rank == 4:
            label = label[..., 0]
        if volume.shape.rank == 3:
            volume = volume[..., None]

        if self.cutout_mode == "slice":
            return self._cutout_mask_slice_wise(volume, label)
        return self._cutout_mask_volume_wise(volume, label)

    def _cutout_mask_2d(self, image: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        mask_h, mask_w = self.mask_size
        y_lo = mask_h // 2
        y_hi = mask_h - y_lo
        x_lo = mask_w // 2
        x_hi = mask_w - x_lo
        cutout_mask = tf.ones((height, width), tf.float32)
        valid_mask = (
            tf.ones((height, width), tf.float32)
            if self.invalid_label is None
            else tf.cast(label != self.invalid_label, tf.float32)
        )
        y = tf.range(height)
        x = tf.range(width)

        for _ in range(self.num_cuts):
            cy = self.random_integers(shape=(), minval=0, maxval=height, dtype=tf.int32)
            cx = self.random_integers(shape=(), minval=0, maxval=width, dtype=tf.int32)
            y_mask = (y >= cy - y_lo) & (y < cy + y_hi)
            x_mask = (x >= cx - x_lo) & (x < cx + x_hi)
            rect = tf.cast(y_mask[:, None] & x_mask[None, :], tf.float32) * valid_mask
            cutout_mask *= 1.0 - rect

        return cutout_mask[..., None]

    def _cutout_mask_slice_wise(self, volume: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(volume)
        depth, height, width = shape[0], shape[1], shape[2]
        mask_h, mask_w = self.mask_size
        y_lo = mask_h // 2
        y_hi = mask_h - y_lo
        x_lo = mask_w // 2
        x_hi = mask_w - x_lo
        cutout_mask = tf.ones((depth, height, width), tf.float32)
        valid_mask = (
            tf.ones((depth, height, width), tf.float32)
            if self.invalid_label is None
            else tf.cast(label != self.invalid_label, tf.float32)
        )
        y = tf.range(height)[None, :]
        x = tf.range(width)[None, :]

        for _ in range(self.num_cuts):
            cy = self.random_integers(shape=[depth], minval=0, maxval=height, dtype=tf.int32)
            cx = self.random_integers(shape=[depth], minval=0, maxval=width, dtype=tf.int32)
            y_mask = (y >= cy[:, None] - y_lo) & (y < cy[:, None] + y_hi)
            x_mask = (x >= cx[:, None] - x_lo) & (x < cx[:, None] + x_hi)
            rect = tf.cast(y_mask[:, :, None] & x_mask[:, None, :], tf.float32) * valid_mask
            cutout_mask *= 1.0 - rect

        return cutout_mask[..., None]

    def _cutout_mask_volume_wise(self, volume: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(volume)
        depth, height, width = shape[0], shape[1], shape[2]
        mask_h, mask_w = self.mask_size
        y_lo = mask_h // 2
        y_hi = mask_h - y_lo
        x_lo = mask_w // 2
        x_hi = mask_w - x_lo
        cutout_mask = tf.ones((depth, height, width), tf.float32)
        valid_mask = (
            tf.ones((depth, height, width), tf.float32)
            if self.invalid_label is None
            else tf.cast(label != self.invalid_label, tf.float32)
        )
        y = tf.range(height)
        x = tf.range(width)

        for _ in range(self.num_cuts):
            cy = self.random_integers(shape=(), minval=0, maxval=height, dtype=tf.int32)
            cx = self.random_integers(shape=(), minval=0, maxval=width, dtype=tf.int32)
            y_mask = (y >= cy - y_lo) & (y < cy + y_hi)
            x_mask = (x >= cx - x_lo) & (x < cx + x_hi)
            rect_hw = tf.cast(y_mask[:, None] & x_mask[None, :], tf.float32)
            rect = rect_hw[None, ...] * valid_mask
            cutout_mask *= 1.0 - rect

        return cutout_mask[..., None]

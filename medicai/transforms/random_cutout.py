from typing import Dict, Sequence, Union

import tensorflow as tf

from .tensor_bundle import TensorBundle


class RandCutOut:
    """
    Random CutOut augmentation for 2D slices stacked as a 3D volume.

    This transform randomly masks rectangular regions (cutouts) independently
    for each slice along the depth dimension. The masked regions can be filled
    with either a constant value or Gaussian noise scaled to the image range.

    Expected image shape:
        - (D, H, W, C) or
        - (D, H, W)

    Example:
        >>> import tensorflow as tf
        >>> from medicai.transforms import RandCutOut
        >>> # Create a dummy 3D volume with channels
        >>> image = tf.random.uniform((128, 128, 128, 4))
        >>> label = tf.random.randint(0, 2, (128, 128, 128, 1))
        >>> rand_cutout = RandCutOut(
        ...     keys=["image", "label"],
        ...     mask_size=[
        ...         image.shape[1] // 4,
        ...         image.shape[2] // 4,
        ...     ],
        ...     num_cuts=5,
        ...     prob=0.8,
        ...     fill_mode="constant",
        ...     fill_value=0.0,
        ... )
        >>>
        >>> output = rand_cutout({"image": image})
        >>> augmented_image, label = output["image"], output["label"]
    """

    def __init__(
        self,
        keys,
        mask_size,
        num_cuts,
        prob=0.5,
        fill_mode="constant",
        fill_value=0.0,
        gaussian_std=0.1,
        invalid_label=None,
        cutout_mode="volume",
    ):
        """
        Args:
            keys:
                Key(s) in the input dictionary to which CutOut is applied.
                Can be a single string ("image") or a sequence like ["image"].
            mask_size:
                Size of the cutout mask as (height, width).
            num_cuts:
                Number of cutout regions applied per volume.
            prob:
                Probability of applying CutOut.
            fill_mode:
                How masked regions are filled. One of {"constant", "gaussian"}.
            fill_value:
                Constant fill value used when fill_mode="constant".
            gaussian_std:
                Standard deviation for Gaussian noise when fill_mode="gaussian".
            invalid_label (Optional[int]): An optional label index (e.g., background
                or ignore index) where applying a cutout is considered technically
                meaningless or should be avoided.
            cutout_mode (str): Controls how cutout regions are applied across the depth
                dimension.
                - "slice": Applies cutout independently for each slice along the depth
                dimension. This behaves like 2D CutOut applied slice-wise and results
                in different masked regions per slice.
                - "volume": Applies the same cutout region across all slices, producing
                a slice-consistent (true 3D) cutout. This preserves volumetric
                continuity and is generally recommended for 3D segmentation tasks.
                Default is "volume".
        """

        if isinstance(keys, (list, tuple)):
            if len(keys) != 2:
                raise ValueError(
                    "`keys` must have length 2 when provided as a sequence. "
                    f"Got length {len(keys)}."
                )

        if not isinstance(mask_size, (list, tuple)) or len(mask_size) != 2:
            raise ValueError("`mask_size` must be a sequence of two integers: (height, width).")

        if not all(isinstance(m, int) and m > 0 for m in mask_size):
            raise ValueError("All values in `mask_size` must be positive integers.")

        if num_cuts <= 0:
            raise ValueError("`num_cuts` must be a positive integer.")

        if not (0.0 <= prob <= 1.0):
            raise ValueError("`prob` must be in the range [0, 1].")

        if fill_mode not in {"gaussian", "constant"}:
            raise ValueError(
                '`fill_mode` must be either "gaussian" or "constant". ' f"Got {fill_mode}."
            )

        if cutout_mode not in {"slice", "volume"}:
            raise ValueError(
                '`cutout_mode` must be one of {"slice", "volume"}. ' f"Got {cutout_mode}."
            )

        self.image_key = keys[0]
        self.label_key = keys[1]

        self.cutout_mode = cutout_mode
        self.mask_size = mask_size
        self.num_cuts = num_cuts
        self.prob = prob
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.gaussian_std = gaussian_std
        self.invalid_label = invalid_label

    def __call__(self, inputs):
        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        image = inputs.data[self.image_key]
        label = inputs.data[self.label_key]

        def apply_cutout():
            out = inputs.data.copy()

            mask = self._generate_cutout_mask(image, label)
            mask_bool = tf.cast(mask, tf.bool)

            if self.fill_mode == "gaussian":
                noise = tf.random.normal(
                    tf.shape(image), stddev=self.gaussian_std, dtype=image.dtype
                )

                im_min = tf.reduce_min(image)
                im_max = tf.reduce_max(image)
                nz_min = tf.reduce_min(noise)
                nz_max = tf.reduce_max(noise)

                fill = (im_max - im_min) * (noise - nz_min) / (nz_max - nz_min + 1e-8) + im_min
            else:
                fill = tf.fill(tf.shape(image), self.fill_value)

            out[self.image_key] = tf.where(mask_bool, image, fill)
            return out

        def skip():
            return inputs.data.copy()

        data = tf.cond(
            tf.random.uniform([]) <= self.prob,
            apply_cutout,
            skip,
        )

        return TensorBundle(data, inputs.meta)

    def _generate_cutout_mask(self, volume, label):
        """
        volume: (D, H, W, C) or (D, H, W)
        label:  (D, H, W) or (D, H, W, 1)
        """

        # Remove label channel if present
        if label.shape.rank == 4:
            label = label[..., 0]

        # Ensure channel-last volume
        if volume.shape.rank == 3:
            volume = volume[..., None]

        if self.cutout_mode == "slice":
            return self._cutout_mask_slice_wise(volume, label)
        else:
            return self._cutout_mask_volume_wise(volume, label)

    def _cutout_mask_slice_wise(self, volume, label):
        """
        volume: (D, H, W[, C])
        label:  (D, H, W)
        """
        shape = tf.shape(volume)
        D, H, W = shape[0], shape[1], shape[2]
        mh, mw = self.mask_size

        cutout_mask = tf.ones((D, H, W), tf.float32)

        if self.invalid_label is None:
            valid_mask = tf.ones((D, H, W), tf.float32)
        else:
            valid_mask = tf.cast(label != self.invalid_label, tf.float32)

        for _ in range(self.num_cuts):
            cy = tf.random.uniform([D], 0, H, tf.int32)
            cx = tf.random.uniform([D], 0, W, tf.int32)

            y = tf.range(H)[None, :]
            x = tf.range(W)[None, :]

            y_mask = (y >= cy[:, None] - mh // 2) & (y < cy[:, None] + mh // 2)
            x_mask = (x >= cx[:, None] - mw // 2) & (x < cx[:, None] + mw // 2)

            rect = tf.cast(y_mask[:, :, None] & x_mask[:, None, :], tf.float32)
            rect = rect * valid_mask

            cutout_mask *= 1.0 - rect

        return cutout_mask[..., None]

    def _cutout_mask_volume_wise(self, volume, label):
        shape = tf.shape(volume)
        D, H, W = shape[0], shape[1], shape[2]
        mh, mw = self.mask_size

        cutout_mask = tf.ones((D, H, W), tf.float32)

        if self.invalid_label is None:
            valid_mask = tf.ones((D, H, W), tf.float32)
        else:
            valid_mask = tf.cast(label != self.invalid_label, tf.float32)

        for _ in range(self.num_cuts):
            cy = tf.random.uniform([], 0, H, tf.int32)
            cx = tf.random.uniform([], 0, W, tf.int32)

            y = tf.range(H)
            x = tf.range(W)

            y_mask = (y >= cy - mh // 2) & (y < cy + mh // 2)
            x_mask = (x >= cx - mw // 2) & (x < cx + mw // 2)

            rect_hw = tf.cast(y_mask[:, None] & x_mask[None, :], tf.float32)
            rect = tf.broadcast_to(rect_hw[None, ...], (D, H, W))

            rect *= valid_mask
            cutout_mask *= 1.0 - rect

        return cutout_mask[..., None]

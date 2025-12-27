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
        >>>
        >>> # Create a dummy 3D volume with channels
        >>> image = tf.random.uniform((128, 128, 128, 4))
        >>>
        >>> rand_cutout = RandCutOut(
        ...     keys=["image"],
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
        >>> augmented_image = output["image"]
    """

    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        mask_size: Sequence[int],
        num_cuts: int,
        prob: float = 0.5,
        fill_mode: str = "constant",  # "constant" or "gaussian"
        fill_value: float = 0.0,
        gaussian_std: float = 0.1,
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
        """
        if isinstance(keys, (list, tuple)):
            if len(keys) != 1:
                raise ValueError(
                    "`keys` must have length 1 when provided as a sequence. "
                    f"Got length {len(keys)}."
                )
            keys = keys[0]
        elif not isinstance(keys, str):
            raise TypeError(
                "`keys` must be a string or a sequence of length 1. " f"Got type {type(keys)}."
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

        self.keys = keys
        self.mask_size = mask_size  # (mh, mw)
        self.num_cuts = num_cuts
        self.prob = prob
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.gaussian_std = gaussian_std

    def __call__(self, inputs: Union[Dict[str, tf.Tensor], any]) -> any:
        """
        Apply random CutOut to the input TensorBundle or dict.

        Args:
            inputs:
                Either a dictionary mapping keys to tensors or a TensorBundle.

        Returns:
            TensorBundle with CutOut applied probabilistically.
        """

        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        if self.keys not in inputs.data:
            raise KeyError(f"Key '{self.keys}' not found in input data.")

        rand_val = tf.random.uniform(())

        def apply_cutout():
            cutout_data = inputs.data.copy()
            image = cutout_data[self.keys]

            # Generate slice-independent mask
            mask = self._generate_slice_independent_mask(image)
            mask_bool = tf.cast(mask, tf.bool)

            # Determine fill value
            if self.fill_mode == "gaussian":
                noise = tf.random.normal(
                    tf.shape(image),
                    stddev=self.gaussian_std,
                    dtype=image.dtype,
                )

                image_max = tf.reduce_max(image)
                image_min = tf.reduce_min(image)
                noise_max = tf.reduce_max(noise)
                noise_min = tf.reduce_min(noise)

                fill_value = (image_max - image_min) * (noise - noise_min) / (
                    noise_max - noise_min + 1e-8
                ) + image_min
            else:
                fill_value = tf.fill(tf.shape(image), self.fill_value)

            # Apply mask
            cutout_data[self.keys] = tf.where(mask_bool, image, fill_value)
            return cutout_data

        def skip_cutout():
            return inputs.data.copy()

        applied_ops = tf.cond(rand_val <= self.prob, apply_cutout, skip_cutout)
        return TensorBundle(applied_ops, inputs.meta)

    def _generate_slice_independent_mask(self, volume: tf.Tensor) -> tf.Tensor:
        """
        Generate a slice-wise independent CutOut mask.

        Each depth slice receives cutouts at different spatial locations.

        Args:
            volume:
                Input tensor of shape (D, H, W, C) or (D, H, W).

        Returns:
            Mask tensor of shape (D, H, W, 1) with values in {0, 1}.
        """

        shape = tf.shape(volume)
        D, H, W = shape[0], shape[1], shape[2]
        mh, mw = self.mask_size

        # Start with an all-ones mask
        mask = tf.ones((D, H, W), dtype=tf.float32)

        for _ in range(self.num_cuts):
            # Random centers per slice
            cy = tf.random.uniform([D], 0, H, dtype=tf.int32)
            cx = tf.random.uniform([D], 0, W, dtype=tf.int32)

            y_grid = tf.range(H)
            x_grid = tf.range(W)

            y_mask = (y_grid[None, :] >= (cy[:, None] - mh // 2)) & (
                y_grid[None, :] < (cy[:, None] + mh // 2)
            )

            x_mask = (x_grid[None, :] >= (cx[:, None] - mw // 2)) & (
                x_grid[None, :] < (cx[:, None] + mw // 2)
            )

            current_cut = y_mask[:, :, None] & x_mask[:, None, :]
            mask = mask * (1.0 - tf.cast(current_cut, tf.float32))

        return mask[..., None]

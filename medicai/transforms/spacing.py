import warnings
from typing import Dict, Sequence, Tuple, Union

import tensorflow as tf

from .resize import resize_volumes
from .tensor_bundle import TensorBundle


class Spacing:
    """
    Resample tensors to a target physical spacing.

    This transform uses the ``affine`` matrix stored in metadata to estimate
    the original voxel spacing, then rescales the selected tensors to the
    requested ``pixdim``. If no affine matrix is available, it falls back to
    ``(1.0, 1.0, 1.0)`` spacing.

    Args:
        keys (Sequence[str]): Keys of the tensors to resample.
        pixdim (Tuple[float, float, float]): The desired physical spacing (voxel dimensions)
            in (depth, height, width) or (z, y, x) order. Default is (1.0, 1.0, 1.0).
        mode (Union[str, Sequence[str], Dict[str, str]]): Interpolation mode used for resampling.

            - If a single string is provided, the same mode is used for all keys.
            - If a tuple or list is provided, it must have the same length as
              ``keys`` and each mode is applied to the corresponding key.
            - If a dictionary is provided, each key is mapped to its own mode.

            Supported modes are:

            - 2D resampling: ``"bilinear"`` and ``"nearest"``
            - 3D resampling: ``"trilinear"`` and ``"nearest"``

    Example:
        Resample an image-label pair to isotropic spacing::

            import tensorflow as tf
            from medicai.transforms import Spacing, TensorBundle

            spacing = Spacing(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("trilinear", "nearest"),
            )

            image = tf.random.normal((32, 64, 64, 1))
            label = tf.random.uniform((32, 64, 64, 1), maxval=2, dtype=tf.int32)
            affine = tf.eye(4)

            bundle = TensorBundle(
                {
                    "image": image, 
                    "label": label
                },
                meta={
                    "affine": affine
                },
            )
            result = spacing(bundle)
            resampled_image = result["image"]
            resampled_label = result["label"]

            print(resampled_image.shape) # (32, 64, 64, 1)
            print(resampled_label.shape) # (32, 64, 64, 1)

    Returns:
        TensorBundle: The transformed output. We can retrieve the resampled
        tensors using the same keys as the input.

    Raises:
        ValueError: If ``pixdim`` is neither 2D nor 3D.
        ValueError: If ``mode`` is a tuple or list and its length does not
            match the length of ``keys``.
        ValueError: If an interpolation mode is not valid for the selected
            spatial dimensionality.
        TypeError: If ``mode`` is not a string, tuple, list, or dict.
    """

    def __init__(
        self,
        keys: Sequence[str] = ("image", "label"),
        pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        mode: Tuple[str, str] = ("trilinear", "nearest"),
    ):
        ndim = len(pixdim)
        if ndim not in (2, 3):
            raise ValueError(f"`pixdim` must be 2D or 3D, got {ndim}D.")

        if ndim == 2:
            valid_modes = {"bilinear", "nearest"}
        else:  # ndim == 3
            valid_modes = {"trilinear", "nearest"}

        self.pixdim = pixdim
        self.keys = keys

        if isinstance(mode, str):
            self.mode = {key: mode for key in keys}
        elif isinstance(mode, (tuple, list)):
            if len(mode) != len(keys):
                raise ValueError("Length of 'mode' must match length of 'keys'.")
            self.mode = dict(zip(keys, mode))
        elif isinstance(mode, dict):
            self.mode = mode
        else:
            raise TypeError("'mode' must be a string, tuple, list, or dict.")

        for key, m in self.mode.items():
            if m not in valid_modes:
                raise ValueError(
                    f"Invalid mode '{m}' for {ndim}D input. "
                    f"Allowed: {sorted(valid_modes)} (key='{key}')."
                )

    def get_spacing_from_affine(self, affine):
        """Calculates the voxel spacing from the affine transformation matrix.

        Args:
            affine (tf.Tensor): The affine transformation matrix (shape (4, 4)).

        Returns:
            Tuple[float, float, float]: The voxel spacing in (depth, height, width) order.
        """
        return tuple(tf.norm(affine[:3, i]) for i in range(3))

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:
        """Apply the spacing resampling to the input TensorBundle.

        Args:
            inputs (TensorBundle): A sample dictionary or ``TensorBundle``. If
                available, the metadata should contain an ``affine`` matrix used
                to infer the original voxel spacing.

        Returns:
            TensorBundle: The transformed output. We can retrieve the resampled
            tensors using the same keys as the input.

        Warns:
            UserWarning: If the input metadata does not contain an ``affine``
            matrix, the transform falls back to spacing ``(1.0, 1.0, 1.0)``.
        """

        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        for key in self.keys:
            if key not in inputs.data:
                continue

            image = inputs.data[key]
            affine = inputs.meta.get("affine", None)

            if affine is not None:
                original_spacing = self.get_spacing_from_affine(affine)
            else:
                original_spacing = (1.0, 1.0, 1.0)
                warnings.warn(
                    "Affine matrix is not provided. Using default spacing (1.0, 1.0, 1.0)."
                )

            resample_image = self.spacingd_resample(
                image, original_spacing, self.pixdim, mode=self.mode.get(key)
            )
            inputs.data[key] = resample_image
        return inputs

    def spacingd_resample(
        self,
        image: tf.Tensor,
        original_spacing: Tuple[float, float, float],
        desired_spacing: Tuple[float, float, float],
        mode: str = "bilinear",
    ) -> tf.Tensor:
        """Resamples a single 3D image tensor to the desired spacing.

        Args:
            image (tf.Tensor): The input 3D image tensor (depth, height, width, channels).
            original_spacing (Tuple[float, float, float]): The original voxel spacing
                in (depth, height, width) order.
            desired_spacing (Tuple[float, float, float]): The desired voxel spacing
                in (depth, height, width) order.
            mode (str): The interpolation mode to use for resampling.

        Returns:
            tf.Tensor: The resampled 3D image tensor.
        """
        scale_d = original_spacing[0] / desired_spacing[0]
        scale_h = original_spacing[1] / desired_spacing[1]
        scale_w = original_spacing[2] / desired_spacing[2]

        original_shape = tf.shape(image)
        original_depth = tf.cast(original_shape[0], tf.float32)
        original_height = tf.cast(original_shape[1], tf.float32)
        original_width = tf.cast(original_shape[2], tf.float32)

        new_depth = tf.cast(tf.round(original_depth * scale_d), tf.int32)
        new_height = tf.cast(tf.round(original_height * scale_h), tf.int32)
        new_width = tf.cast(tf.round(original_width * scale_w), tf.int32)
        spatial_shape = (new_depth, new_height, new_width)

        image = image[None, ...]  # Add temp batch axis
        resized_dhw = resize_volumes(image, *spatial_shape, method=mode, align_corners=False)
        resized_dhw = resized_dhw[0]  # Remove temp batch axis
        return resized_dhw

from medicai.utils.general import hide_warnings

hide_warnings()

import warnings
from typing import Dict, Sequence, Tuple, Union

import tensorflow as tf

from medicai.transforms.resize import Resize

from .tensor_bundle import TensorBundle


class Spacing:
    """Resamples tensors to a specified physical spacing (voxel dimensions).

    This transform takes tensors (e.g., image and label) and resamples them
    to a target physical spacing defined by `pixdim`. It determines the
    original spacing from the affine matrix in the metadata. If the affine
    matrix is not provided, it defaults to a spacing of (1.0, 1.0, 1.0).
    The resampling is performed using the `Resize` transform with specified
    interpolation modes.
    """

    def __init__(
        self,
        keys: Sequence[str] = ("image", "label"),
        pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        mode: Tuple[str, str] = ("bilinear", "nearest"),
    ):
        """Initializes the Spacing transform.

        Args:
            keys (Sequence[str]): Keys of the tensors to resample. Default is ("image", "label").
            pixdim (Tuple[float, float, float]): The desired physical spacing (voxel dimensions)
                in (depth, height, width) or (z, y, x) order. Default is (1.0, 1.0, 1.0).
            mode (Union[str, Tuple[str, str]]): The interpolation mode(s) to use for resampling.
                - If a single string is provided, it's used for all keys.
                - If a tuple of two strings is provided, the first mode is used for the first key
                  and the second mode for the second key (assuming the default `keys`).
                  For more than two keys, provide a dictionary in `__call__`.
                  Supported modes are those of `tf.image.resize` (for spatial H, W)
                  and the depth interpolation used within `Resize` (for spatial D).
                  Default is ("bilinear", "nearest").
        """
        self.pixdim = pixdim
        self.keys = keys
        self.image_mode = mode[0]
        self.label_mode = mode[1]
        self.mode = dict(zip(keys, mode))

    def get_spacing_from_affine(self, affine):
        """Calculates the voxel spacing from the affine transformation matrix.

        Args:
            affine (tf.Tensor): The affine transformation matrix (shape (4, 4)).

        Returns:
            Tuple[float, float, float]: The voxel spacing in (depth, height, width) order.
        """
        width_spacing = tf.norm(affine[:3, 0])
        depth_spacing = tf.norm(affine[:3, 1])
        height_spacing = tf.norm(affine[:3, 2])
        return (width_spacing, depth_spacing, height_spacing)

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:
        """Apply the spacing resampling to the input TensorBundle.

        Args:
            inputs (TensorBundle): A dictionary containing tensors and metadata.
                The metadata is expected to contain an 'affine' key representing the
                affine transformation matrix.

        Returns:
            TensorBundle: A dictionary with resampled tensors and the original metadata.
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

        new_depth = tf.cast(original_depth * scale_d, tf.int32)
        new_height = tf.cast(original_height * scale_h, tf.int32)
        new_width = tf.cast(original_width * scale_w, tf.int32)

        spatial_shape = (new_depth, new_height, new_width)
        inputs = TensorBundle({"image": image})
        resized_dhw = Resize(keys=["image"], mode=[mode], spatial_shape=spatial_shape)(inputs).data[
            "image"
        ]
        return resized_dhw

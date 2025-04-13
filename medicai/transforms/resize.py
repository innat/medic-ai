from typing import Dict, Sequence, Tuple, Union

from medicai.utils.general import hide_warnings

hide_warnings()
import tensorflow as tf

from medicai.transforms.depth_interpolate import (
    depth_interpolation_methods,
    SUPPORTED_DEPTH_METHODS,
)

from .tensor_bundle import TensorBundle


class Resize:
    """Resizes the spatial dimensions of tensors to a specified shape.

    This transform resizes the spatial dimensions (height, width for 2D;
    depth, height, width for 3D) of the tensors specified by `keys` to the
    given `spatial_shape` using a specified interpolation `mode`. Different
    interpolation modes can be specified for different keys.
    """

    def __init__(
        self,
        keys: Sequence[str],
        mode: Tuple[str, str] = ("bilinear", "nearest"),
        spatial_shape: Tuple[int, ...] = (96, 96, 96),
    ):
        """Initializes the Resize transform.

        Args:
            keys (Sequence[str]): Keys of the tensors to resize.
            mode (Union[str, Tuple[str, ...]]): The interpolation mode to use for resizing.
                - If a single string is provided, the same mode is used for all keys.
                - If a tuple of strings is provided, it must have the same length as `keys`,
                  and the mode at each index will be used for the tensor with the
                  corresponding key. Supported modes include 'nearest', 'bilinear', etc.
                  for 2D resizing (using `tf.image.resize`) and these along with others
                  for 3D resizing (using a combination of `tf.image.resize` and a
                  depth interpolation function).
                  Default is "bilinear".
            spatial_shape (Tuple[int, ...]): The desired spatial shape after resizing.
                It should be a tuple of two integers (height, width) for 2D resizing
                or a tuple of three integers (depth, height, width) for 3D resizing.
                Default is (96, 96, 96) for 3D.

        Raises:
            ValueError: If `mode` is a tuple and its length does not match the length of `keys`.
            ValueError: If `spatial_shape` is neither 2D nor 3D.
        """
        self.keys = keys
        self.spatial_shape = spatial_shape

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

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:
        """Apply the resizing transformation to the input TensorBundle.

        Args:
            inputs (TensorBundle): A dictionary containing tensors and metadata. The tensors
                specified by `self.keys` will be resized according to `self.spatial_shape`
                and `self.mode`.

        Returns:
            TensorBundle: A dictionary with the resized tensors and the original metadata.
        """

        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        for key in self.keys:
            if key not in inputs.data:
                continue

            image = inputs.data[key]

            if len(self.spatial_shape) == 2:
                inputs.data[key] = self._resize_2d(image, key)
            elif len(self.spatial_shape) == 3:
                inputs.data[key] = self._resize_3d(image, key)
            else:
                raise ValueError("spatial_shape must be either 2D or 3D.")

        return inputs

    def _resize_2d(self, image: tf.Tensor, key: str) -> tf.Tensor:
        """Resizes a 2D image tensor.

        Args:
            image (tf.Tensor): The input 2D image tensor (height, width, channels).
            key (str): The key of the tensor being processed (used to retrieve the mode).

        Returns:
            tf.Tensor: The resized 2D image tensor (new_height, new_width, channels).
        """
        new_height, new_width = self.spatial_shape
        resized_hw = tf.image.resize(image, [new_height, new_width], method=self.mode.get(key))
        return resized_hw

    def _resize_3d(self, image: tf.Tensor, key: str) -> tf.Tensor:
        """Resizes a 3D image tensor.

        Args:
            image (tf.Tensor): The input 3D image tensor (depth, height, width, channels).
            key (str): The key of the tensor being processed (used to retrieve the mode).

        Returns:
            tf.Tensor: The resized 3D image tensor (new_depth, new_height, new_width, channels).
        """
        new_depth, new_height, new_width = self.spatial_shape
        resized_hw = tf.image.resize(image, [new_height, new_width], method=self.mode.get(key))

        mode_for_key = self.mode.get(key)
        if mode_for_key is not None:
            normalized_mode = mode_for_key.lower()
            depth_method_map = {
                "bilinear": "linear",
                "bicubic": "cubic",
                "nearest": "nearest",
            }
            depth_method = depth_method_map.get(normalized_mode)
        else:
            depth_method = None

        if depth_method not in SUPPORTED_DEPTH_METHODS:
            raise ValueError(
                f"Unsupported depth interpolation method: {depth_method}. "
                f"Supported methods are: {SUPPORTED_DEPTH_METHODS}"
            )

        resized_dhw = depth_interpolation_methods[depth_method](resized_hw, new_depth, depth_axis=0)
        return resized_dhw

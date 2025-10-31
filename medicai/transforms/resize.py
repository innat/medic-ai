from typing import Dict, Sequence, Tuple, Union

import tensorflow as tf

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
        mode: Tuple[str, str] = ("trilinear", "nearest"),
        spatial_shape: Tuple[int, ...] = (96, 96, 96),
    ):
        """Initializes the Resize transform.

        Args:
            keys (Sequence[str]): Keys of the tensors to resize.
            mode (Union[str, Tuple[str, ...]]): The interpolation mode to use for resizing.
                - If a single string is provided, the same mode is used for all keys.
                - If a tuple of strings is provided, it must have the same length as `keys`,
                  and the mode at each index will be used for the tensor with the
                  corresponding key. Supported modes include 'nearest', 'bilinear', 'trilinear'.
                  for 2D resizing (using `tf.image.resize`) supports 'bilinear' and 'nearest'.
                  for 3D resizing (using a custom volume resizer) supports 'trilinear' and 'nearest'.
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

        ndim = len(spatial_shape)
        if ndim not in (2, 3):
            raise ValueError(f"`spatial_shape` must be 2D or 3D, got {ndim}D.")

        if ndim == 2:
            valid_modes = {"bilinear", "nearest"}
        else:  # ndim == 3
            valid_modes = {"trilinear", "nearest"}

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
        image = image[None, ...]  # Add temp batch axis
        resized_dhw = resize_volumes(
            image, *self.spatial_shape, method=self.mode.get(key), align_corners=False
        )
        resized_dhw = resized_dhw[0]  # Remove temp batch axis
        return resized_dhw


# This could be temporary: issue: https://github.com/keras-team/keras/issues/21785
def resize_volumes(volumes, depth, height, width, method="trilinear", align_corners=False):
    def trilinear_resize(volumes, depth, height, width, align_corners):
        original_dtype = volumes.dtype
        volumes = tf.cast(volumes, "float32")
        in_d = tf.shape(volumes)[1]
        in_h = tf.shape(volumes)[2]
        in_w = tf.shape(volumes)[3]

        if align_corners:
            # Map corner to corner
            z_coords = tf.linspace(0.0, tf.cast(in_d - 1, "float32"), depth)
            y_coords = tf.linspace(0.0, tf.cast(in_h - 1, "float32"), height)
            x_coords = tf.linspace(0.0, tf.cast(in_w - 1, "float32"), width)
        else:
            # More accurate PyTorch-compatible mapping
            # Ref: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            scale_d = tf.cast(in_d, "float32") / tf.cast(depth, "float32")
            scale_h = tf.cast(in_h, "float32") / tf.cast(height, "float32")
            scale_w = tf.cast(in_w, "float32") / tf.cast(width, "float32")

            # Create grid with proper alignment
            z_coords = (tf.range(depth, dtype="float32") + 0.5) * scale_d - 0.5
            y_coords = (tf.range(height, dtype="float32") + 0.5) * scale_h - 0.5
            x_coords = (tf.range(width, dtype="float32") + 0.5) * scale_w - 0.5

            # Ensure we don't sample outside the volume
            z_coords = tf.clip_by_value(z_coords, 0.0, tf.cast(in_d - 1, "float32"))
            y_coords = tf.clip_by_value(y_coords, 0.0, tf.cast(in_h - 1, "float32"))
            x_coords = tf.clip_by_value(x_coords, 0.0, tf.cast(in_w - 1, "float32"))

        # Helper function for 1D interpolation
        def interpolate_1d(input_vol, coords, axis):
            # Get floor and ceil indices
            idx0 = tf.cast(tf.floor(coords), "int32")
            idx1 = tf.minimum(idx0 + 1, tf.shape(input_vol)[axis] - 1)

            # Get the values at these indices
            values0 = tf.gather(input_vol, idx0, axis=axis)
            values1 = tf.gather(input_vol, idx1, axis=axis)

            # Calculate weights
            weight1 = coords - tf.cast(idx0, "float32")
            weight0 = 1.0 - weight1

            # Reshape for broadcasting
            new_shape = [1] * 5  # bs, d, h, w, c
            new_shape[axis] = tf.shape(coords)[0]
            weight0 = tf.reshape(weight0, new_shape)
            weight1 = tf.reshape(weight1, new_shape)

            return weight0 * values0 + weight1 * values1

        # Apply interpolation along each dimension
        interp_d = interpolate_1d(volumes, z_coords, axis=1)
        interp_h = interpolate_1d(interp_d, y_coords, axis=2)
        interp_w = interpolate_1d(interp_h, x_coords, axis=3)

        return tf.cast(interp_w, original_dtype)

    def nearest(volumes, depth, height, width):
        shape = tf.shape(volumes)
        bs, d, h, w, c = shape[0], shape[1], shape[2], shape[3], shape[4]

        z = tf.linspace(0.0, tf.cast(d - 1, "float32"), depth)
        z = tf.cast(tf.round(z), "int32")
        z = tf.clip_by_value(z, 0, d - 1)

        y = tf.linspace(0.0, tf.cast(h - 1, "float32"), height)
        y = tf.cast(tf.round(y), "int32")
        y = tf.clip_by_value(y, 0, h - 1)

        x = tf.linspace(0.0, tf.cast(w - 1, "float32"), width)
        x = tf.cast(tf.round(x), "int32")
        x = tf.clip_by_value(x, 0, w - 1)

        # Create 3D grid
        Z, Y, X = tf.meshgrid(z, y, x, indexing="ij")

        # indices
        Z = tf.reshape(Z, (-1,))
        Y = tf.reshape(Y, (-1,))
        X = tf.reshape(X, (-1,))

        # Batch replication
        batch_idx = tf.repeat(tf.range(bs), tf.shape(Z)[0])
        Z = tf.tile(Z, [bs])
        Y = tf.tile(Y, [bs])
        X = tf.tile(X, [bs])

        # Flatten input
        flat = tf.reshape(volumes, (bs * d * h * w, c))

        # Compute linear indices
        indices = (batch_idx * d * h * w) + (Z * h * w) + (Y * w) + X
        result = tf.gather(flat, indices, axis=0)

        # Reshape to final size
        result = tf.reshape(result, (bs, depth, height, width, c))
        return result

    if method == "trilinear":
        return trilinear_resize(volumes, depth, height, width, align_corners)

    elif method == "nearest":
        return nearest(volumes, depth, height, width)

    else:
        raise ValueError(f"Unsupported resize method: {method}")

from typing import Callable, Dict, Optional, Sequence, Union

from medicai.utils.general import hide_warnings

hide_warnings()

import tensorflow as tf

from .tensor_bundle import TensorBundle


class CropForeground:
    """Crop an image based on the foreground defined by a selection function.

    This transform identifies the foreground within a specified source tensor
    using a provided selection function. It then determines the minimal bounding
    box encompassing this foreground and crops all specified tensors in the
    input bundle to this bounding box. Optionally, a margin can be added to the
    bounding box, and the resulting dimensions can be made divisible by a
    specified value. The start and end coordinates of the cropped region can
    also be stored in the metadata of the input bundle.
    """

    def __init__(
        self,
        keys: Sequence[str] = ("image", "label"),
        source_key: str = "image",
        select_fn: Callable = lambda x: x > 0,
        channel_indices: Optional[Sequence[int]] = None,
        margin: Union[Sequence[int], int] = 0,
        allow_smaller: bool = True,
        k_divisible: Union[Sequence[int], int] = 1,
        start_coord_key: Optional[str] = "foreground_start_coord",
        end_coord_key: Optional[str] = "foreground_end_coord",
        allow_missing_keys: bool = False,
    ):
        """
        Initializes the CropForeground transform.

        Args:
            keys (Sequence[str]): Keys of the tensors to crop. Default is ("image", "label").
            source_key (str): Key of the tensor to use to determine the foreground. Default is "image".
            select_fn (Callable): A function that takes a tensor and returns a boolean mask
                indicating the foreground. Default is lambda x: x > 0.
            channel_indices (Optional[Sequence[int]]): If specified, only these channels of the
                source tensor will be used to determine the foreground. Default is None.
            margin (Union[Sequence[int], int]): Amount of margin to add to the bounding box.
                If an int, the same margin is applied to all spatial dimensions. If a sequence,
                it should have a length of 3, corresponding to the margin in depth, height, and width.
                Default is 0.
            allow_smaller (bool): If True, the cropped region can be smaller than the margin.
                If False, the cropped region will be at least as large as 2 * margin in each dimension.
                Default is True.
            k_divisible (Union[Sequence[int], int]): Ensure the cropped size is divisible by k in each
                dimension. If an int, the same value is applied to all spatial dimensions. If a
                sequence, it should have a length of 3. Default is 1.
            start_coord_key (Optional[str]): Key to store the start coordinates of the cropped region
                in the metadata. If None, the start coordinates are not stored. Default is "foreground_start_coord".
            end_coord_key (Optional[str]): Key to store the end coordinates of the cropped region
                in the metadata. If None, the end coordinates are not stored. Default is "foreground_end_coord".
            allow_missing_keys (bool): If True, processing will continue even if some of the specified
                keys are not found in the input data. The missing keys will be ignored. Default is False.
        """
        self.keys = keys
        self.source_key = source_key
        self.select_fn = select_fn
        self.channel_indices = channel_indices
        self.margin = margin
        self.allow_smaller = allow_smaller
        self.k_divisible = k_divisible
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:
        """
        Apply the CropForeground transform to the input TensorBundle.

        Args:
            inputs (TensorBundle): A dictionary containing tensors and metadata.

        Returns:
            TensorBundle: A dictionary with cropped tensors and updated metadata.
        """

        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        # Extract the source data (used to determine the foreground)
        if self.source_key not in inputs.data and self.allow_missing_keys:
            return inputs
        source_data = inputs.data[self.source_key]

        # Validate input shapes
        for key in self.keys:
            if key not in inputs.data and self.allow_missing_keys:
                continue
            if len(inputs.data[key].shape) != 4:
                raise ValueError(
                    f"Input tensor '{key}' must have shape (depth, height, width, channels)."
                )

        # Apply channel selection if specified
        if self.channel_indices is not None:
            source_data = tf.gather(source_data, self.channel_indices, axis=-1)

        # Find the bounding box of the foreground
        min_coords, max_coords = self.find_bounding_box(source_data, self.select_fn)

        # Add margin to the bounding box
        min_coords, max_coords = self.add_margin(
            min_coords, max_coords, self.margin, tf.shape(source_data)[:3], self.allow_smaller
        )

        # Ensure the bounding box is divisible by k_divisible
        min_coords, max_coords = self.make_divisible(
            min_coords, max_coords, self.k_divisible, tf.shape(source_data)[:3]
        )

        # Crop the tensors using the bounding box
        for key in self.keys:
            if key not in inputs.data and self.allow_missing_keys:
                continue
            inputs.data[key] = self.crop_tensor(inputs.data[key], min_coords, max_coords)

        # Record the bounding box coordinates if requested
        if self.start_coord_key is not None:
            inputs.meta[self.start_coord_key] = min_coords
        if self.end_coord_key is not None:
            inputs.meta[self.end_coord_key] = max_coords

        return inputs

    def find_bounding_box(self, image, select_fn):
        """
        Find the bounding box of the foreground in the image.

        Args:
            image (tf.Tensor): The input tensor to find the foreground in.
                Shape should be (depth, height, width, channels).
            select_fn (Callable): A function that takes a tensor and returns a boolean mask
                indicating the foreground.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: A tuple containing the minimum and maximum
            coordinates of the foreground with shape (3,).
        """
        # Apply the selection function to create a mask
        mask = select_fn(image)
        # Reduce across channels to get a 3D mask
        mask = tf.reduce_any(mask, axis=-1)
        # Find coordinates of the foreground
        coords = tf.where(mask)
        # Compute min and max coordinates
        min_coords = tf.reduce_min(coords, axis=0)
        max_coords = tf.reduce_max(coords, axis=0)
        return min_coords, max_coords

    def add_margin(self, min_coords, max_coords, margin, image_shape, allow_smaller):
        """
        Add margin to the bounding box, ensuring it stays within the image bounds.

        Args:
            min_coords (tf.Tensor): The minimum coordinates of the bounding box (shape (3,)).
            max_coords (tf.Tensor): The maximum coordinates of the bounding box (shape (3,)).
            margin (Union[Sequence[int], int]): The margin to add to each dimension.
            image_shape (tf.Tensor): The shape of the original image (shape (3,)).
            allow_smaller (bool): Whether to allow the cropped size to be smaller than 2 * margin.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: The adjusted minimum and maximum coordinates.
        """
        if isinstance(margin, int):
            margin = [margin] * 3  # Apply the same margin to all dimensions
        elif len(margin) != 3:
            raise ValueError("Margin must be an int or a sequence of length 3.")

        # Ensure all tensors have the same data type (e.g., int32)
        min_coords = tf.cast(min_coords, tf.int32)
        max_coords = tf.cast(max_coords, tf.int32)
        image_shape = tf.cast(image_shape, tf.int32)
        margin = tf.cast(margin, tf.int32)

        # Subtract margin from min_coords and add to max_coords
        min_coords = tf.maximum(min_coords - margin, 0)
        max_coords = tf.minimum(max_coords + margin + 1, image_shape)

        if not allow_smaller:
            # Ensure the bounding box is at least as large as the margin
            min_coords = tf.minimum(min_coords, image_shape - margin)
            max_coords = tf.maximum(max_coords, margin)

        return min_coords, max_coords

    def make_divisible(self, min_coords, max_coords, k_divisible, image_shape):
        """
        Ensure the bounding box dimensions are divisible by k_divisible.

        Args:
            min_coords (tf.Tensor): The minimum coordinates of the bounding box (shape (3,)).
            max_coords (tf.Tensor): The maximum coordinates of the bounding box (shape (3,)).
            k_divisible (Union[Sequence[int], int]): The divisibility factor for each dimension.
            image_shape (tf.Tensor): The shape of the original image (shape (3,)).

        Returns:
            tuple[tf.Tensor, tf.Tensor]: The adjusted minimum and maximum coordinates.
        """
        if isinstance(k_divisible, int):
            k_divisible = [k_divisible] * 3  # Apply the same value to all dimensions
        elif len(k_divisible) != 3:
            raise ValueError("k_divisible must be an int or a sequence of length 3.")

        # Ensure all tensors have the same data type (e.g., int32)
        min_coords = tf.cast(min_coords, tf.int32)
        max_coords = tf.cast(max_coords, tf.int32)
        image_shape = tf.cast(image_shape, tf.int32)
        k_divisible = tf.cast(k_divisible, tf.int32)

        # Calculate the size of the bounding box
        size = max_coords - min_coords

        # Calculate the remainder when dividing by k_divisible
        remainder = size % k_divisible

        # Calculate the padding needed to make the size divisible by k_divisible
        padding = tf.where(remainder != 0, k_divisible - remainder, 0)

        # Adjust max_coords by adding the padding
        max_coords = max_coords + padding

        # Ensure the bounding box stays within the image bounds
        max_coords = tf.minimum(max_coords, image_shape)

        return min_coords, max_coords

    def crop_tensor(self, tensor, min_coords, max_coords):
        """
        Crop a tensor using the bounding box coordinates.

        Args:
            tensor (tf.Tensor): The tensor to crop (shape (depth, height, width, channels)).
            min_coords (tf.Tensor): The minimum coordinates of the cropping region (shape (3,)).
            max_coords (tf.Tensor): The maximum coordinates of the cropping region (shape (3,)).

        Returns:
            tf.Tensor: The cropped tensor.
        """
        return tensor[
            min_coords[0] : max_coords[0],
            min_coords[1] : max_coords[1],
            min_coords[2] : max_coords[2],
            :,
        ]

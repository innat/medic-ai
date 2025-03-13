import tensorflow as tf
from typing import Tuple, Dict

class Spacingd:
    """
    Resamples 3D image and label tensors to a specified spatial resolution.

    Args:
        pixdim (Tuple[float, float, float]): The desired voxel spacing in (depth, height, width).
        mode (Tuple[str, str], optional): The interpolation modes for image and label resizing.
            Defaults to ("bilinear", "nearest").
    """
    def __init__(self, pixdim: Tuple[float, float, float], mode: Tuple[str, str] = ("bilinear", "nearest")):
        self.pixdim = pixdim
        self.image_mode = mode[0]
        self.label_mode = mode[1]

    def __call__(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Applies resampling to the input image and label tensors.

        Args:
            inputs (Dict[str, tf.Tensor]): A dictionary with 'image' and 'label' tensors.

        Returns:
            Dict[str, tf.Tensor]: A dictionary with resampled 'image' and 'label' tensors.
        """
        image = inputs['image']
        label = inputs['label']
        
        resample_image = self.spacingd_resample(image, (1.0, 1.0, 1.0), self.pixdim, self.image_mode)
        resample_label = self.spacingd_resample(label, (1.0, 1.0, 1.0), self.pixdim, self.label_mode)
        
        return {'image': resample_image[..., None], 'label': resample_label[..., None]}

    def spacingd_resample(self, image: tf.Tensor, original_spacing: Tuple[float, float, float], desired_spacing: Tuple[float, float, float], mode: str = "bilinear") -> tf.Tensor:
        """
        Resizes a 3D tensor to match the desired spatial resolution.

        Args:
            image (tf.Tensor): The input 3D tensor.
            original_spacing (Tuple[float, float, float]): The original voxel spacing.
            desired_spacing (Tuple[float, float, float]): The target voxel spacing.
            mode (str, optional): Interpolation mode for resizing. Defaults to "bilinear".

        Returns:
            tf.Tensor: The resized 3D tensor.
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
        
        resized_image = tf.image.resize(image, [new_height, new_width], method=mode)
        resized_image = tf.transpose(resized_image[..., 0], perm=[1, 0, 2])  # Transpose to resize depth
        resized_image = tf.image.resize(resized_image[..., tf.newaxis], [new_depth, new_height], method=mode)
        resized_image = tf.transpose(resized_image[..., 0], perm=[1, 0, 2])  # Transpose back
        
        return resized_image
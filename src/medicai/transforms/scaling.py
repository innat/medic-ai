import tensorflow as tf
from typing import Optional, Dict

class ScaleIntensityRange:
    """
    A TensorFlow transformation class that scales the intensity of an image within a specified range.

    Attributes:
        a_min (float): Minimum intensity value of the input range.
        a_max (float): Maximum intensity value of the input range.
        b_min (Optional[float]): Minimum intensity value of the output range. Defaults to None.
        b_max (Optional[float]): Maximum intensity value of the output range. Defaults to None.
        clip (bool): Whether to clip the output values to the specified range. Defaults to False.
        dtype (tf.DType): Data type to which the output image will be cast. Defaults to tf.float32.
    """
    
    def __init__(
        self,
        a_min: float,
        a_max: float,
        b_min: Optional[float] = None,
        b_max: Optional[float] = None,
        clip: bool = False,
        dtype: tf.DType = tf.float32,
    ):
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.dtype = dtype

    def __call__(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Applies intensity scaling to the input image while keeping the label unchanged.
        
        Args:
            inputs (Dict[str, tf.Tensor]): A dictionary with 'image' and 'label' tensors.
        
        Returns:
            Dict[str, tf.Tensor]: A dictionary containing the transformed image and unchanged label.
        """
        image = inputs['image']
        label = inputs['label']
        scaled_image = self.scale_intensity_range(image)
        return {'image': scaled_image, 'label': label}

    def scale_intensity_range(self, image: tf.Tensor) -> tf.Tensor:
        """
        Scales the intensity of the image from the given input range [a_min, a_max] 
        to the output range [b_min, b_max] if provided.
        
        Args:
            image (tf.Tensor): Input image tensor.
        
        Returns:
            tf.Tensor: Scaled image tensor.
        """
        image = tf.convert_to_tensor(image, dtype=self.dtype)

        if self.a_max == self.a_min:
            return image - self.a_min if self.b_min is None else image - self.a_min + self.b_min

        # Normalize to [0, 1]
        image = (image - self.a_min) / (self.a_max - self.a_min)

        # Scale to [b_min, b_max] if provided
        if self.b_min is not None and self.b_max is not None:
            image = image * (self.b_max - self.b_min) + self.b_min

        # Clip the values if required
        if self.clip and self.b_min is not None and self.b_max is not None:
            image = tf.clip_by_value(image, self.b_min, self.b_max)

        return tf.cast(image, dtype=self.dtype)
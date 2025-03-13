
import tensorflow as tf

def ScaleIntensityRanged(
    a_min: float,
    a_max: float,
    b_min: Optional[float] = None,
    b_max: Optional[float] = None,
    clip: bool = False,
    dtype: tf.DType = tf.float32,
):
    def wrapper(inputs):
        image = inputs['image']
        label = inputs['label']
        scaled_image = scale_intensity_range(image, a_min, a_max, b_min, b_max, clip, dtype)
        return {'image': scaled_image, 'label': label}
        
    
    def scale_intensity_range(image, a_min, a_max, b_min=None, b_max=None, clip=False, dtype=tf.float32):
        image = tf.convert_to_tensor(image, dtype=dtype)
    
        # Handle potential divide-by-zero
        if a_max == a_min:
            return image - a_min if b_min is None else image - a_min + b_min
    
        # Normalize to [0, 1]
        image = (image - a_min) / (a_max - a_min)
    
        # Scale to [b_min, b_max] if provided
        if b_min is not None and b_max is not None:
            image = image * (b_max - b_min) + b_min
    
        # Clip the values if required
        if clip and b_min is not None and b_max is not None:
            image = tf.clip_by_value(image, b_min, b_max)

        image = tf.cast(image, dtype=dtype)
    
        return image


    return wrapper

import tensorflow as tf

def Spacingd(pixdim, mode=("bilinear", "nearest")):
    def wrapper(inputs):
        image = inputs['image']
        label = inputs['label']

        resample_image = spacingd_resample(image, (1.0, 1.0, 1.0), pixdim, mode[0])
        resample_label = spacingd_resample(label, (1.0, 1.0, 1.0), pixdim, mode[1])

        return {'image': resample_image[..., None], 'label': resample_label[..., None]}
        

    def spacingd_resample(image, original_spacing, desired_spacing, mode="bilinear"):
        # Calculate scaling factors
        scale_d = original_spacing[0] / desired_spacing[0]
        scale_h = original_spacing[1] / desired_spacing[1]
        scale_w = original_spacing[2] / desired_spacing[2]
        
        # Get the original dimensions as float32
        original_depth = tf.cast(tf.shape(image)[0], tf.float32)
        original_height = tf.cast(tf.shape(image)[1], tf.float32)
        original_width = tf.cast(tf.shape(image)[2], tf.float32)
        
        # Calculate new dimensions
        new_depth = tf.cast(original_depth * scale_d, tf.int32)
        new_height = tf.cast(original_height * scale_h, tf.int32)
        new_width = tf.cast(original_width * scale_w, tf.int32)
        
        # Resize the image
        resized_image = tf.image.resize(image, [new_height, new_width], method=mode)
        resized_image = tf.transpose(resized_image[..., 0], perm=[1, 0, 2])  # Transpose to resize depth
        resized_image = tf.image.resize(resized_image[..., tf.newaxis], [new_depth, new_height], method=mode)
        resized_image = tf.transpose(resized_image[..., 0], perm=[1, 0, 2])  # Transpose back
        return resized_image

    return wrapper
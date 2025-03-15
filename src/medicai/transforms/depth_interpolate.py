
import tensorflow as tf

class DepthInterpolation:

    SUPPORTED_METHOD = ('linear', 'nearest', 'cubic')

    def __init__(self, target_depth, depth_axis=0, method='linear'):
        if method not in self.SUPPORTED_METHOD:
            raise ValuerError(
                f'Support interplation methods are {self.SUPPORTED_METHOD} '
                f'But got {method}'
            )
        
        self.target_depth=target_depth
        self.depth_axis=0
        self.method=method

    def __call__(self, inputs):
        ip_methods = {
            self.SUPPORTED_METHOD[0]: self.linear_interpolation,
            self.SUPPORTED_METHOD[1]: self.nearest_interpolation,
            self.SUPPORTED_METHOD[2]: self.cubic_interpolation,
        }
        return ip_methods.get(self.method)(
            inputs, self.target_depth, self.depth_axis
        )

    @staticmethod
    def nearest_interpolation(volume, target_depth, depth_axis=0):
        # Generate floating-point indices for the target depth
        depth_indices = tf.linspace(0.0, tf.cast(tf.shape(volume)[depth_axis] - 1, tf.float32), target_depth)
        # Round the indices to the nearest integer (nearest-neighbor interpolation)
        depth_indices = tf.cast(depth_indices, tf.int32)
        # Gather slices from the original volume using the rounded indices
        resized_volume = tf.gather(volume, depth_indices, axis=depth_axis)
        return resized_volume

    @staticmethod
    def linear_interpolation(volume, target_depth, depth_axis=0):
        # Get the original depth size along the specified axis
        original_depth = tf.shape(volume)[depth_axis]
        
        # Generate floating-point indices for the target depth
        indices = tf.linspace(0.0, tf.cast(original_depth - 1, tf.float32), target_depth)
        
        # Split indices into integer and fractional parts
        lower_indices = tf.cast(tf.floor(indices), tf.int32)
        alpha = indices - tf.cast(lower_indices, tf.float32)  # Fractional part
        
        # Adjust the shape of alpha for broadcasting along the depth axis
        alpha_shape = [1] * len(tf.shape(volume))
        alpha_shape[depth_axis] = target_depth
        alpha = tf.reshape(alpha, alpha_shape)  # Reshape alpha for proper broadcasting
        
        # Gather the lower and upper slices along the specified depth axis
        lower_indices = tf.maximum(lower_indices, 0)
        upper_indices = tf.minimum(lower_indices + 1, original_depth - 1)
        
        lower_slices = tf.gather(volume, lower_indices, axis=depth_axis)
        upper_slices = tf.gather(volume, upper_indices, axis=depth_axis)
        
        # Cast slices to float32 to ensure type compatibility with alpha
        lower_slices = tf.cast(lower_slices, tf.float32)
        upper_slices = tf.cast(upper_slices, tf.float32)
        
        # Perform linear interpolation along the specified depth axis
        interpolated_volume = (1 - alpha) * lower_slices + alpha * upper_slices
        
        return interpolated_volume

    @staticmethod
    def cubic_interpolation(volume, target_depth, depth_axis=0):
        # Get the original depth size along the specified axis
        original_depth = tf.shape(volume)[depth_axis]
        
        # Generate floating-point indices for the target depth
        indices = tf.linspace(0.0, tf.cast(original_depth - 1, tf.float32), target_depth)
        
        # Split indices into integer and fractional parts
        lower_indices = tf.cast(tf.floor(indices), tf.int32)
        alpha = indices - tf.cast(lower_indices, tf.float32)  # Fractional part
        
        # Adjust the shape of alpha for broadcasting along the depth axis
        alpha_shape = [1] * len(tf.shape(volume))
        alpha_shape[depth_axis] = target_depth
        alpha = tf.reshape(alpha, alpha_shape)  # Reshape alpha for proper broadcasting
        
        # Gather the four neighboring slices along the specified depth axis
        indices_0 = tf.maximum(lower_indices - 1, 0)
        indices_1 = lower_indices
        indices_2 = tf.minimum(lower_indices + 1, original_depth - 1)
        indices_3 = tf.minimum(lower_indices + 2, original_depth - 1)
        
        slices_0 = tf.gather(volume, indices_0, axis=depth_axis)
        slices_1 = tf.gather(volume, indices_1, axis=depth_axis)
        slices_2 = tf.gather(volume, indices_2, axis=depth_axis)
        slices_3 = tf.gather(volume, indices_3, axis=depth_axis)
        
        # Cast slices to float32 to ensure type compatibility with alpha
        slices_0 = tf.cast(slices_0, tf.float32)
        slices_1 = tf.cast(slices_1, tf.float32)
        slices_2 = tf.cast(slices_2, tf.float32)
        slices_3 = tf.cast(slices_3, tf.float32)
        
        # Cubic interpolation coefficients
        alpha_sq = alpha ** 2
        alpha_cu = alpha ** 3
        w0 = -0.5 * alpha_cu + 1.0 * alpha_sq - 0.5 * alpha
        w1 = 1.5 * alpha_cu - 2.5 * alpha_sq + 1.0
        w2 = -1.5 * alpha_cu + 2.0 * alpha_sq + 0.5 * alpha
        w3 = 0.5 * alpha_cu - 0.5 * alpha_sq
        
        # Perform cubic interpolation along the specified depth axis
        interpolated_volume = (
            w0 * slices_0 +
            w1 * slices_1 +
            w2 * slices_2 +
            w3 * slices_3
        )
        
        return interpolated_volume

import tensorflow as tf


def nearest_interpolation(volume, target_depth, depth_axis=0):
    """Resize a tensor along one axis with nearest-neighbor interpolation.

    Args:
        volume: Input tensor to resize.
        target_depth: Target size for ``depth_axis``.
        depth_axis: Axis to interpolate along.

    Returns:
        ``tf.Tensor``: The resized tensor.
    """
    depth_indices = tf.linspace(
        0.0, tf.cast(tf.shape(volume)[depth_axis] - 1, tf.float32), target_depth
    )
    depth_indices = tf.cast(depth_indices, tf.int32)
    resized_volume = tf.gather(volume, depth_indices, axis=depth_axis)
    return resized_volume


def linear_interpolation(volume, target_depth, depth_axis=0, align_corners=False):
    """Resize a tensor along one axis with linear interpolation.

    Args:
        volume: Input tensor to resize.
        target_depth: Target size for ``depth_axis``.
        depth_axis: Axis to interpolate along.
        align_corners: Whether corner indices should align exactly.

    Returns:
        ``tf.Tensor``: The resized tensor. Interpolation is performed in
        ``float32`` space.
    """
    original_depth = tf.shape(volume)[depth_axis]

    if original_depth == target_depth:
        return volume

    if align_corners:
        indices = tf.linspace(0.0, tf.cast(original_depth - 1, tf.float32), target_depth)
    else:
        if target_depth > 1:
            scale = tf.cast(original_depth, tf.float32) / tf.cast(target_depth, tf.float32)
            indices = (tf.range(target_depth, dtype=tf.float32) + 0.5) * scale - 0.5
            indices = tf.clip_by_value(indices, 0.0, tf.cast(original_depth - 1, tf.float32))
        else:
            indices = (tf.cast(original_depth, tf.float32) - 1) / 2.0

    lower_indices = tf.cast(tf.floor(indices), tf.int32)
    alpha = indices - tf.cast(lower_indices, tf.float32)

    alpha_shape = [1] * len(tf.shape(volume))
    alpha_shape[depth_axis] = target_depth
    alpha = tf.reshape(alpha, alpha_shape)

    lower_indices = tf.maximum(lower_indices, 0)
    upper_indices = tf.minimum(lower_indices + 1, original_depth - 1)

    lower_slices = tf.gather(volume, lower_indices, axis=depth_axis)
    upper_slices = tf.gather(volume, upper_indices, axis=depth_axis)

    lower_slices = tf.cast(lower_slices, tf.float32)
    upper_slices = tf.cast(upper_slices, tf.float32)

    interpolated_volume = (1 - alpha) * lower_slices + alpha * upper_slices

    return interpolated_volume


def cubic_interpolation(volume, target_depth, depth_axis=0):
    """Resize a tensor along one axis with cubic interpolation.

    Args:
        volume: Input tensor to resize.
        target_depth: Target size for ``depth_axis``.
        depth_axis: Axis to interpolate along.

    Returns:
        ``tf.Tensor``: The resized tensor. Interpolation is performed in
        ``float32`` space.
    """
    original_depth = tf.shape(volume)[depth_axis]

    indices = tf.linspace(0.0, tf.cast(original_depth - 1, tf.float32), target_depth)

    lower_indices = tf.cast(tf.floor(indices), tf.int32)
    alpha = indices - tf.cast(lower_indices, tf.float32)

    alpha_shape = [1] * len(tf.shape(volume))
    alpha_shape[depth_axis] = target_depth
    alpha = tf.reshape(alpha, alpha_shape)

    indices_0 = tf.maximum(lower_indices - 1, 0)
    indices_1 = lower_indices
    indices_2 = tf.minimum(lower_indices + 1, original_depth - 1)
    indices_3 = tf.minimum(lower_indices + 2, original_depth - 1)

    slices_0 = tf.gather(volume, indices_0, axis=depth_axis)
    slices_1 = tf.gather(volume, indices_1, axis=depth_axis)
    slices_2 = tf.gather(volume, indices_2, axis=depth_axis)
    slices_3 = tf.gather(volume, indices_3, axis=depth_axis)

    slices_0 = tf.cast(slices_0, tf.float32)
    slices_1 = tf.cast(slices_1, tf.float32)
    slices_2 = tf.cast(slices_2, tf.float32)
    slices_3 = tf.cast(slices_3, tf.float32)

    alpha_sq = alpha**2
    alpha_cu = alpha**3
    w0 = -0.5 * alpha_cu + 1.0 * alpha_sq - 0.5 * alpha
    w1 = 1.5 * alpha_cu - 2.5 * alpha_sq + 1.0
    w2 = -1.5 * alpha_cu + 2.0 * alpha_sq + 0.5 * alpha
    w3 = 0.5 * alpha_cu - 0.5 * alpha_sq

    interpolated_volume = w0 * slices_0 + w1 * slices_1 + w2 * slices_2 + w3 * slices_3

    return interpolated_volume


SUPPORTED_DEPTH_METHODS = ("linear", "nearest", "cubic")

depth_interpolation_methods = {
    SUPPORTED_DEPTH_METHODS[0]: linear_interpolation,
    SUPPORTED_DEPTH_METHODS[1]: nearest_interpolation,
    SUPPORTED_DEPTH_METHODS[2]: cubic_interpolation,
}

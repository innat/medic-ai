from medicai.utils.general import hide_warnings

hide_warnings()

import tensorflow as tf

from .tensor_bundle import TensorBundle


class RandSpatialCrop:
    def __init__(self, keys, roi_size, max_roi_size=None, random_center=True, random_size=False):
        self.keys = keys
        self.roi_size = tf.convert_to_tensor(roi_size, dtype=tf.int32)
        self.max_roi_size = (
            tf.convert_to_tensor(max_roi_size, dtype=tf.int32) if max_roi_size is not None else None
        )
        self.random_center = random_center
        self.random_size = random_size

    def __call__(self, inputs: TensorBundle) -> TensorBundle:
        sample_key = self.keys[0]
        img = inputs.data[sample_key]
        input_shape = tf.shape(img)  # shape = (D, H, W, C)
        spatial_shape = input_shape[:3]  # D, H, W

        roi_size = self._get_roi_size(spatial_shape)
        center = self._get_center(spatial_shape, roi_size)

        # Get start and end slices
        starts = tf.maximum(center - roi_size // 2, 0)
        ends = tf.minimum(starts + roi_size, spatial_shape)
        starts = tf.maximum(ends - roi_size, 0)

        d_start, h_start, w_start = tf.unstack(starts)
        d_end, h_end, w_end = tf.unstack(ends)

        for key in self.keys:
            if key in inputs.data:
                volume = inputs[key]
                inputs.data[key] = volume[d_start:d_end, h_start:h_end, w_start:w_end, :]
        return inputs

    def _get_roi_size(self, spatial_shape):
        roi_size = self.roi_size
        if self.random_size:
            max_roi_size = self.max_roi_size if self.max_roi_size is not None else spatial_shape
            max_roi_size = tf.where(max_roi_size <= 0, spatial_shape, max_roi_size)

            def sample_dim(min_s, max_s, img_s):
                min_s = tf.where(min_s <= 0, img_s, min_s)
                max_s = tf.where(max_s <= 0, img_s, max_s)
                max_s = tf.minimum(max_s, img_s)  # Ensure max_s doesn't exceed image size
                min_s = tf.minimum(min_s, max_s)  # Ensure min_s is not greater than max_s
                return tf.random.uniform([], minval=min_s, maxval=max_s + 1, dtype=tf.int32)

            roi_size = tf.stack(
                [sample_dim(roi_size[i], max_roi_size[i], spatial_shape[i]) for i in range(3)]
            )
        else:
            roi_size = tf.where(roi_size > 0, roi_size, spatial_shape)
            roi_size = tf.minimum(roi_size, spatial_shape)
        return roi_size

    def _get_center(self, spatial_shape, roi_size):
        if self.random_center:
            max_start = tf.maximum(spatial_shape - roi_size, 0)  # Corrected max_start

            # Sample each spatial center coordinate independently
            random_start = tf.stack(
                [tf.random.uniform([], maxval=max_start[i] + 1, dtype=tf.int32) for i in range(3)]
            )
            center = random_start + roi_size // 2
        else:
            center = spatial_shape // 2
        return center

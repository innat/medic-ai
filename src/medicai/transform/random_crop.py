
import tensorflow as tf

def RandCropByPosNegLabel(spatial_size, pos, neg, num_samples=1):
    if pos < 0 or neg < 0:
        raise ValueError("pos and neg must be non-negative.")
    if pos == 0 and neg == 0:
        raise ValueError("pos and neg cannot both be zero.")

    pos_ratio = pos / (pos + neg)


    def _sample_patch(image, label, positive=True):
        shape = tf.shape(image, out_type=tf.int32)
        depth, height, width = shape[0], shape[1], shape[2]

        coords = tf.where(label > 0) if positive else tf.where(label == 0)

        if tf.equal(tf.shape(coords)[0], 0):
            coords = tf.where(tf.ones_like(label) > 0)

        idx = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(coords)[0], dtype=tf.int32)
        center = tf.cast(coords[idx], tf.int32)

        start = [
            tf.maximum(center[i] - spatial_size[i] // 2, 0) for i in range(3)
        ]
        end = [
            tf.minimum(start[i] + spatial_size[i], shape[i]) for i in range(3)
        ]
        start = [end[i] - spatial_size[i] for i in range(3)]

        patch_image = image[start[0]:end[0], start[1]:end[1], start[2]:end[2], :]
        patch_label = label[start[0]:end[0], start[1]:end[1], start[2]:end[2], :]

        return patch_image, patch_label


    def _process_sample(image, label):
        rand_val = tf.random.uniform(shape=[], minval=0, maxval=1)
        return _sample_patch(image, label, positive=rand_val < pos_ratio)

    def wrapper(inputs):
        image = inputs['image']
        label = inputs['label']

        if len(image.shape) != 4 or len(label.shape) != 4:
            raise ValueError("Input tensors must have shape (depth, height, width, channels).")

        image_patches, label_patches = tf.map_fn(
            lambda _: _process_sample(image, label),
            tf.range(num_samples, dtype=tf.int32),
            dtype=(tf.float32, tf.float32)
        )

        return {'image': image_patches, 'label': label_patches}
 
    return wrapper
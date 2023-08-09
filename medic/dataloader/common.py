import tensorflow as tf


def data_reader(config):
    image_size = config.dataset.image_size
    class_label = config.dataset.num_classes

    def image_reader(path):
        file_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(file_bytes, channels=3)
        img = tf.image.resize(img, (image_size, image_size))
        return img

    def labels_reader(path, label):
        return (image_reader(path), tf.one_hot(label, depth=class_label))

    return labels_reader


def read_files(image_path, size, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.io.decode_png(image, channels=3)
        image = tf.squeeze(image)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.divide(image, 128)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[size, size])
        image = tf.cast(image, tf.int32)
    else:
        image = tf.io.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[size, size])

    return tf.cast(image, dtype=tf.float32)


def load_data(image_list, mask_list, size):
    image = read_files(image_list, size=size)
    mask = read_files(mask_list, size=size, mask=True)
    return image, mask

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
        return (
            image_reader(path), tf.one_hot(label, depth=class_label)
        )

    return labels_reader


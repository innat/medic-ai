import tensorflow as tf
from .common import augment

IMG_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE


def data_reader():
    def image_reader(path):
        file_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(file_bytes, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return img

    def labels_reader(path, label):
        return image_reader(path), tf.one_hot(label, depth=5)

    return labels_reader


def get_dataloader(df, batch_size, augment, shuffle):
    dataset = tf.data.Dataset.from_tensor_slices(df["id_code"].values, df["diagnosis"].values)
    decode_fn = data_reader()
    dataset = dataset.map(decode_fn, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat() if shuffle else dataset
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True) if shuffle else dataset
    dataset = dataset.batch(batch_size, drop_remainder=shuffle)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


import os
import pandas as pd
import tensorflow as tf
from .common import augment


def data_reader(image_size):
    def image_reader(path):
        file_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(file_bytes, channels=3)
        img = tf.image.resize(img, (image_size, image_size))
        return img

    def labels_reader(path, label):
        return image_reader(path), tf.one_hot(label, depth=5)

    return labels_reader


def get_dataloader(config):
    df = pd.read_csv(
        os.path.join(config.dataset.path, 'df.csv')
    )
    df = df.sample(frac=1).reset_index(drop=True)
    df['id_code'] = df['id_code'].apply(lambda x: f'{config.dataset.path}{x}.png')

    reader_method = data_reader(config.dataset.image_size)
    dataset = tf.data.Dataset.from_tensor_slices(
        df["id_code"].values, df["diagnosis"].values
    )
    dataset = dataset.map(reader_method, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(8 * config.dataset.batch_size)
    dataset = dataset.batch(config.dataset.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

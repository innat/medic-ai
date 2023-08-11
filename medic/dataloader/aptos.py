import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset
from medic.dataloader.common import data_reader

from typing import Union
from omegaconf import DictConfig, ListConfig


class APTOSDataloader:
    def __init__(self, config: Union[DictConfig, ListConfig]) -> None:
        super().__init__()
        self.config = config
        self.train_df, self.valid_df = self.prepare_dataframe(self.config)
        self.dataset = Dataset.from_tensor_slices(
            (
                self.train_df[config.dataset.meta_columns.x].values,
                self.train_df[config.dataset.meta_columns.y].values,
            )
        )
        self.reader_method = data_reader(self.config)


    def prepare_dataframe(self, config: Union[DictConfig, ListConfig]):
        df = pd.read_csv(
            os.path.join(config.dataset.path, config.dataset.name, config.dataset.meta_file)
        )
        df = df.sample(frac=1).reset_index(drop=True)
        df[config.dataset.meta_columns.x] = df[config.dataset.meta_columns.x].apply(
            lambda x: (
                f"{config.dataset.path}/"
                f"{config.dataset.name}/"
                f"{config.dataset.sub_folder}/"
                f"{x}.{config.dataset.image_extention}"
            )
        )
        return df

    def preprocess(self):
        dataset = self.dataset.map(self.reader_method, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def augment(self, augment_config: Union[DictConfig, ListConfig]) -> keras.Sequential:

        augmentation = keras.Sequential(
            name="augmentation"
        )

        if 'random_resize_and_crop' in augment_config:
            width = augment_config['random_resize_and_crop']['width']
            height = augment_config['random_resize_and_crop']['height']
            augmentation.add(
                keras.layers.Resizing(
                    height,
                    width,
                )
            )
            augmentation.add(
                keras.layers.RandomCrop(
                    self.config.dataset.image_size, 
                    self.config.dataset.image_size
                )
            )

        if augment_config.get('random_flip', False):
            augmentation.add(keras.layers.RandomFlip("horizontal"))

        return augmentation

    def generate(self, shuffle: bool, drop_reminder: bool) -> tf.data.Dataset:
        dataset = self.preprocess()
        dataset = dataset.shuffle(8 * self.config.dataset.batch_size) if shuffle else dataset
        dataset = dataset.batch(self.config.dataset.batch_size, drop_remainder=drop_reminder)

        if shuffle:
            augment_config = self.config.get('augment', {}).get('training', {})
            dataset = self.augment(augment_config)(dataset)

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def data_reader(config: Union[DictConfig, ListConfig]):
        image_size = config.dataset.image_size
        num_classes = config.dataset.num_classes
        class_activation = config.dataset.class_activation

        def image_reader(path):
            file_bytes = tf.io.read_file(path)
            img = tf.image.decode_jpeg(file_bytes, channels=3)
            img = tf.image.resize(img, (image_size, image_size))
            return img

        def labels_reader(path, label):
            if class_activation == "softmax":
                target_array = tf.one_hot(label, depth=num_classes)
            elif class_activation is None:
                target_array = label
            return (image_reader(path), target_array)

        return labels_reader

import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset

from typing import Union
from omegaconf import DictConfig, ListConfig


class APTOSDataloader:
    def __init__(self, config: Union[DictConfig, ListConfig]) -> None:
        self.config = config
        self.df = self.prepare_dataframe(self.config)
        self.dataset = Dataset.from_tensor_slices(
            (
                self.df[self.config.dataset.meta_columns.x].values,
                self.df[self.config.dataset.meta_columns.y].values,
            )
        )
        self.reader_method = self.data_reader(self.config)
        self._preprocessed = False


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

    def preprocess(self) -> tf.data.Dataset:
        if self._preprocessed:
            return self.dataset
        
        self.dataset = self.dataset.map(self.reader_method, num_parallel_calls=tf.data.AUTOTUNE)
        self._preprocessed = True

        return self.dataset

    def prepare_batches(self) -> tf.data.Dataset:

        if not self._preprocessed:
            self.preprocess()

        dataset = self.dataset
        dataset = dataset.shuffle(8 * self.config.dataset.batch_size) if self.config.dataset.shuffle else dataset
        dataset = dataset.batch(
            self.config.dataset.batch_size, 
            drop_remainder=True
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    @staticmethod
    def data_reader(config: DictConfig):
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
                target_array = tf.cast(label, dtype=tf.flaot32)
            return (image_reader(path), target_array)

        return labels_reader

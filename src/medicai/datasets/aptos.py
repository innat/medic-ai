import os
from typing import List, Union

import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig, ListConfig
from tensorflow import keras
from tensorflow.data import Dataset


class APTOSDataloader:
    def __init__(
        self,
        dataset_path: str,
        subfolder: str,
        meta_file: str,
        meta_columns: List[str, str],
        num_classes: int,
        shuffle: bool = True,
        batch_size: int = 32,
        image_size: int = 224,
        label_mode: str = "int",
        image_extention: str = "png",
        **kwargs,
    ):
        self.dataset_path = dataset_path
        self.dataframe = meta_file
        self.subfolder = subfolder
        self.x, self.y = meta_columns
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.image_size = image_size
        self.label_mode = label_mode
        self.image_extention = image_extention

        self.df = self.prepare_dataframe()
        self.dataset = Dataset.from_tensor_slices(
            (
                self.df[self.x].values,
                self.df[self.y].values,
            )
        )
        self.reader_method = self.data_reader()
        self._preprocessed = False

    def prepare_dataframe(self):
        df = pd.read_csv(os.path.join(self.dataset_path, self.dataframe))
        df = df.sample(frac=1).reset_index(drop=True)
        df[self.x] = df[self.x].apply(
            lambda x: (f"{self.dataset_path}/" f"{self.subfolder}/" f"{x}.{self.image_extention}")
        )
        return df

    def prepare_sample(self) -> tf.data.Dataset:
        if self._preprocessed:
            return self.dataset

        self.dataset = self.dataset.map(self.reader_method, num_parallel_calls=tf.data.AUTOTUNE)
        self._preprocessed = True

        return self.dataset

    def prepare_samples(self) -> tf.data.Dataset:
        if not self._preprocessed:
            self.prepare_sample()

        dataset = self.dataset
        dataset = dataset.shuffle(8 * self.batch_size) if self.shuffle else dataset
        dataset = dataset.batch(self.batch_size, drop_remainder=self.shuffle)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def data_reader(self):
        image_size = self.image_size
        num_classes = self.num_classes

        def image_reader(path):
            file_bytes = tf.io.read_file(path)
            img = tf.image.decode_jpeg(file_bytes, channels=3)
            img = tf.image.resize(img, (image_size, image_size))
            return img

        def labels_reader(path, label):
            if self.label_mode == "categorical":
                target_array = tf.one_hot(label, depth=num_classes)
            else:
                target_array = tf.cast(label, dtype=tf.flaot32)
            return (image_reader(path), target_array)

        return labels_reader

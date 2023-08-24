import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset

from typing import Union, List
from omegaconf import DictConfig, ListConfig


class APTOSDataloader:
    def __init__(
            self, 
            dataset_path:str,
            data_directory:str,
            meta_file:str,
            meta_columns:List[str, str],
            batch_size:int=32,
            image_size:int=224,
            label_mode:str='int',
            image_extention:str='png',
            **kwargs,
        ):
        self.dataset_path=dataset_path
        self.dataframe=meta_file
        self.data_directory=data_directory
        self.x, self.y = meta_columns
        self.batch_size=batch_size
        self.image_size=image_size
        self.label_mode=label_mode
        self.image_extention=image_extention

        self.df = self.prepare_dataframe(
            self.dataset_path,
        )
        self.dataset = Dataset.from_tensor_slices(
            (
                self.df[self.config.dataset.meta_columns.x].values,
                self.df[self.config.dataset.meta_columns.y].values,
            )
        )
        self.reader_method = self.data_reader(self.config)
        self._preprocessed = False

    def prepare_dataframe(self):
        df = pd.read_csv(
            os.path.join(self.dataset_path, self.dataframe)
        )
        df = df.sample(frac=1).reset_index(drop=True)
        df[self.x] = df[self.x].apply(
            lambda x: (
                f"{self.dataset_path}/"
                f"{self.data_directory}/"
                f"{x}.{self.image_extention}"
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

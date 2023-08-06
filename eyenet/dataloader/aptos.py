import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset
from eyenet.dataloader.common import data_reader


class APTOSDataloader:
    def __init__(self, config: str) -> None:
        super().__init__()
        self.train_df = self.prepare_dataframe(config)
        self.dataset = Dataset.from_tensor_slices(
            (self.train_df["id_code"].values, self.train_df["diagnosis"].values)
        )
        self.reader_method = data_reader(config.dataset.image_size)
        self.config = config

    def prepare_dataframe(self, config):
        train_df = pd.read_csv(
            os.path.join(config.dataset.path, config.dataset.name, "train_images", "df.csv")
        )
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        train_df["id_code"] = train_df["id_code"].apply(
            lambda x: f"{config.dataset.path}/{config.dataset.name}/train_images/{x}.png"
        )

    def process(self):
        dataset = dataset.map(self.reader_method, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(8 * self.config.dataset.batch_size)
        return dataset

    def augment(self, config):
        augmentation = keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal"),
            ]
        )
        return augmentation

    def load(self):
        dataset = self.process(self.config)
        dataset = dataset.batch(self.config.dataset.batch_size, drop_remainder=True)
        
        # TODO : Make it customizable.
        if 'augment' in self.config.dataset:
            dataset = self.augment(self.config)(dataset)
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

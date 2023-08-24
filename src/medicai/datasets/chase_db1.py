import os

import tensorflow as tf
from tensorflow.data import Dataset

from medicai.dataloader.common import load_data

exts = ("jpg", "JPG", "png", "PNG", "tif", "gif", "ppm")


class CHASE_DB1:
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        images, masks = self.prepare_dataset(config)
        self.dataset = Dataset.from_tensor_slices((images, masks))

    def prepare_dataset(self, config):
        input_data = os.path.join(config.dataset.path, config.dataset.name, "images")
        target_data = os.path.join(config.dataset.path, config.dataset.name, "masks")
        images = sorted(
            [
                os.path.join(input_data, fname)
                for fname in os.listdir(input_data)
                if fname.endswith(exts) and not fname.startswith(".")
            ]
        )
        masks = sorted(
            [
                os.path.join(target_data, fname)
                for fname in os.listdir(target_data)
                if fname.endswith(exts) and not fname.startswith(".")
            ]
        )
        return images, masks

    def process(self):
        dataset = self.dataset.map(
            lambda x, y: load_data(x, y, size=self.config.dataset.image_size),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.shuffle(8 * self.config.dataset.batch_size)
        return dataset

    def augment(self):
        pass

    def load(self):
        dataset = self.process()
        dataset = dataset.batch(self.config.dataset.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

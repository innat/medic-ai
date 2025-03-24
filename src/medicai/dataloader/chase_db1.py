import os

import tensorflow as tf
from tensorflow.data import Dataset

from medicai.datasets.common import IMAGE_EXTENTION


class CHASE_DB1:
    def __init__(
        self,
        dataset_path: str,
        image_path: str,
        mask_path: str,
        num_classes: int,
        shuffle: bool = True,
        batch_size: int = 32,
        image_size: int = 224,
        label_mode: str = "int",
        image_extention: str = "png",
        **kwargs,
    ):
        self.dataset_path = dataset_path
        self.image_path = image_path
        self.mask_path = mask_path
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.image_size = image_size
        self.label_mode = label_mode
        self.image_extention = image_extention

        super().__init__()
        images, masks = self.prepare_dataset()
        self.dataset = Dataset.from_tensor_slices((images, masks))
        self._preprocessed = False

    def prepare_dataset(self):
        input_data = os.path.join(self.dataset_path, self.image_path)
        target_data = os.path.join(self.dataset_path, self.mask_path)
        images = sorted(
            [
                os.path.join(input_data, fname)
                for fname in os.listdir(input_data)
                if fname.endswith(IMAGE_EXTENTION) and not fname.startswith(".")
            ]
        )
        masks = sorted(
            [
                os.path.join(target_data, fname)
                for fname in os.listdir(target_data)
                if fname.endswith(IMAGE_EXTENTION) and not fname.startswith(".")
            ]
        )
        return images, masks

    def prepare_sample(self):
        if self._preprocessed:
            return self.dataset

        dataset = self.dataset.map(
            lambda x, y: self.data_reader(x, y, size=self.image_size),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        self._preprocessed = True

        return dataset

    def prepare_samples(self):
        if not self._preprocessed:
            self.prepare_sample()

        dataset = self.dataset
        dataset = dataset.shuffle(8 * self.batch_size) if self.shuffle else dataset
        dataset = dataset.batch(self.batch_size, drop_remainder=self.shuffle)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def data_reader(self, image_list, mask_list, size):
        image = self.read_files(image_list, size=size)
        mask = self.read_files(mask_list, size=size, mask=True)
        return image, mask

    def read_files(self, file, size, mask=False):
        image = tf.io.read_file(file)
        if mask:
            image = tf.io.decode_png(image, channels=3)
            image = tf.squeeze(image)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.divide(image, 128)
            image.set_shape([None, None, 1])
            image = tf.image.resize(
                images=image, size=[size, size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            image = tf.cast(image, tf.int32)
            if self.label_mode == "categorical":
                image = tf.one_hot(image, depth=self.num_classes)
        else:
            image = tf.io.decode_png(image, channels=3)
            image.set_shape([None, None, 3])
            image = tf.image.resize(images=image, size=[size, size])

        return tf.cast(image, dtype=tf.float32)

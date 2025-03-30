from medicai.utils.general import hide_warnings

hide_warnings()

import json

import keras
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


class NiftiDataLoader(keras.utils.PyDataset):
    def __init__(
        self,
        dataset_path,
        batch_size=1,
        dim=(128, 128, 128),
        shuffle=True,
        input_channels=1,
        num_classes=None,
        training=True,
    ):
        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)

        self.image_paths = [item["image"] for item in self.dataset["training"]]
        self.label_paths = [item["label"] for item in self.dataset["training"]]

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.training = training
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        image_paths_batch = [self.image_paths[k] for k in indices]
        label_paths_batch = [self.label_paths[k] for k in indices]

        if self.training:
            X, y = self.train_data_generator(image_paths_batch, label_paths_batch)
            return X, y

        return self.test_data_generator(image_paths_batch, label_paths_batch)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def train_data_generator(self, image_paths_batch, label_paths_batch):
        X = np.zeros((self.batch_size, *self.dim, self.input_channels), dtype=np.float32)
        y_shape = (*self.dim, self.num_classes) if self.num_classes else (*self.dim, 1)
        y = np.zeros((self.batch_size, *y_shape), dtype=np.float32)

        for i, (img_path, lbl_path) in enumerate(zip(image_paths_batch, label_paths_batch)):
            img = nib.load(img_path).get_fdata()
            lbl = nib.load(lbl_path).get_fdata()

            img_zoom = zoom(img, np.array(self.dim) / np.array(img.shape), order=1)
            lbl_zoom = zoom(lbl, np.array(self.dim) / np.array(lbl.shape), order=0)

            if img_zoom.ndim == 3:
                img_zoom = np.expand_dims(img_zoom, axis=-1)
            if img_zoom.shape[-1] != self.input_channels:
                img_zoom = np.repeat(img_zoom, self.input_channels, axis=-1)

            X[i] = img_zoom

            if self.num_classes:
                y[i] = keras.utils.to_categorical(lbl_zoom, num_classes=self.num_classes)
            else:
                y[i, ..., 0] = lbl_zoom

        return X, y

    def test_data_generator(self, image_paths_batch, label_paths_batch):
        X = []
        y = []

        for img_path, lbl_path in zip(image_paths_batch, label_paths_batch):
            img = nib.load(img_path).get_fdata().transpose(2, 1, 0)
            lbl = nib.load(lbl_path).get_fdata().transpose(2, 1, 0)
            X.append(np.expand_dims(img, axis=-1))
            y.append(np.expand_dims(lbl, axis=-1))

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

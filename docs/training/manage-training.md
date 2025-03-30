## Classification

A high-level overview of the 3D classification process using `medicai`.

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow" # tensorflow, torch

import keras

from medicai.models import SwinTransformer
from medicai.transforms import (
    Compose,
    ScaleIntensityRange,
    RandRotate90,
    Resize
)
```

**Transformation**

Import processing and augmentation operations for training, while using only processing operations for validation.

```python
def train_transformation(image, label):
    data = {"image": image, "label": label}
    pipeline = Compose(
        [
            ScaleIntensityRange(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            Resize(keys=["image"], mode=['bilinear'], spatial_shape=(96,96,96)),
            RandRotate90(keys=["image"], prob=0.1, max_k=3, spatial_axes=(1, 2))
        ]
    )
    result = pipeline(data)
    return result.data["image"], result.data["label"]

def val_transformation(image, label):
    data = {"image": image, "label": label}
    pipeline = Compose(
        [
            ScaleIntensityRange(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            Resize(keys=["image"], mode=['bilinear'], spatial_shape=(96,96,96)),
        ]
    )
    result = pipeline(data)
    return result.data["image"], result.data["label"]
```

**Dataloader**

Let's build the dataloader using [`keras.utils.PyDataset`](https://keras.io/api/utils/python_utils/#pydataset-class).

```python
import numpy as np
import nibabel as nib

class NiftiDataLoader(keras.utils.PyDataset):
    def __init__(
        self, 
        image_paths, 
        labels,
        batch_size=1, 
        dim=(128, 128, 128), 
        shuffle=True, 
        training=True
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.training = training
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        image_paths_batch = [self.image_paths[k] for k in indices]
        labels_batch = [self.labels[k] for k in indices]
        
        X = np.zeros((self.batch_size, *self.dim, 1), dtype=np.float32)
        y = np.zeros((self.batch_size), dtype=np.float32)

        for i, (img_path, label) in enumerate(zip(image_paths_batch, labels_batch)):
            # Load and preprocess image
            img = nib.load(img_path).get_fdata()
            
            # Add channel dimension if needed
            if img.ndim == 3:
                img = np.expand_dims(img, axis=-1)

            if self.training:
                img, label = train_transformation(img, label)
            else:
                img, label = val_transformation(img, label)
            
            X[i] = img
            y[i] = label
        
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)


train_loader = NiftiDataLoader(
    image_paths=X_train,
    labels=y_train,
    batch_size=3,
    dim=(96, 96, 96),
    shuffle=True,
    training=True
)

val_loader = NiftiDataLoader(
    image_paths=X_test,
    labels=y_test,
    batch_size=3,
    dim=(96, 96, 96),
    shuffle=False,
    training=False
)
```

**Model**

Create the model and compile it with the necessary loss function and metrics.

```python
model = SwinTransformer(
    input_shape=(96, 96, 96, 1),
    num_classes=1,
    classifier_activation='sigmoid',
)
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-4,
    ),
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=["acc"],
    jit_compile=False,
)
```

**Training**

```python
history = model.fit(
    train_loader, 
    epochs=10,
    validation_data=val_loader
)
```

## Segmentation

A high-level overview of the 3D segmentation process using `medicai`.

```python
import keras
import tensorflow as tf
from medicai.metrics import DiceMetric
from medicai.losses import SparseDiceCELoss
from medicai.models import SwinUNETR
from medicai.transforms import (
    Compose,
    ScaleIntensityRange,
    CropForeground,
    RandCropByPosNegLabel,
    Spacing,
    Orientation,
    RandShiftIntensity,
    RandRotate90,
    RandFlip
)
from medicai.callbacks import SlidingWindowInferenceCallback
```

**Transformation**

Import processing and augmentation operations for training, while using only processing operations for validation.

```python
def train_transformation(sample):
    meta = {"affine": sample["image_affine"]} # Since image and label affine are the same
    data = {"image": sample["image"], "label": sample["label"]}
    pipeline = Compose([
        ScaleIntensityRange(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        CropForeground(
            keys=("image", "label"),
            source_key="image"
        ),
        Orientation(keys = ("image", "label"), axcodes = "RAS"),
        Spacing(pixdim=(2.0, 1.5, 1.5), keys=["image", "label"]),
        RandCropByPosNegLabel(
            keys=("image", "label"),
            spatial_size=[96, 96, 96], 
            pos=1, 
            neg=1, 
            num_samples=1
        ),
        RandFlip(keys=["image", "label"], spatial_axis=[0], prob=0.1),
        RandFlip(keys=["image", "label"], spatial_axis=[1], prob=0.1),
        RandFlip(keys=["image", "label"], spatial_axis=[2], prob=0.1),
        RandRotate90(keys=["image", "label"], prob=0.1, max_k=3, spatial_axes=(0, 1)),
        RandShiftIntensity(keys=["image"], offsets=0.10, prob=0.50)
    ])
    result = pipeline(data, meta)
    return result.data["image"], result.data["label"]


def val_transformation(sample):
    meta = {"affine": sample["image_affine"]} # Since image and label affine are the same
    data = {"image": sample["image"], "label": sample["label"]}
    pipeline = Compose([
        ScaleIntensityRange(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        CropForeground(
            keys=("image", "label"),
            source_key="image"
        ),
        Orientation(keys = ("image", "label"), axcodes = "RAS"),
        Spacing(pixdim=(2.0, 1.5, 1.5), keys=["image", "label"])
    ])
    result = pipeline(data, meta)
    return result.data["image"], result.data["label"]
```

**Dataloader**

Create the dataloader using `tf.data.TFRecordDataset` API. Generating `tfrecod` is shown [here](https://www.kaggle.com/code/ipythonx/generate-3d-nii-to-tfrecord-dataset)

```python
def load_tfrecord_dataset(tfrecord_pattern, batch_size=1, shuffle=True):
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(tfrecord_pattern))
    dataset = dataset.shuffle(buffer_size=50) if shuffle else dataset
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(rearrange_shape, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.map(train_transformation, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(val_transformation, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


tfrecord_pattern = "1/tfrecords/{}_shard_*.tfrec"

train_ds = load_tfrecord_dataset(
    tfrecord_pattern.format("training"), batch_size=1, shuffle=True
)
val_ds = load_tfrecord_dataset(
    tfrecord_pattern.format("validation"), batch_size=1, shuffle=False
)
```

**Model**

Build the `SwinUNETR` model with the specified input shape and number of classes.

```python
num_classes=4
model=SwinUNETR(
    input_shape=(96, 96, 96, 1),
    out_channels=num_classes,
    classifier_activation=None,
)

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-5,
    ),
    loss=SparseDiceCELoss(from_logits=True),
    metrics=[DiceMetric(
        num_classes=num_classes,
        include_background=True,
        reduction="mean",
        ignore_empty=True,
        smooth=1e-6,
        name='dice_score'
    )],
    jit_compile=False,
)
```

**Sliding Window Inference Callback**

The Sliding Window Inference callback provides a convenient method for processing large volumetric samples efficiently. Instead of processing the entire volume at once (which may exceed memory limits), the input is divided into smaller overlapping windows. Each window is inferred separately, and the outputs are stitched together to form the final prediction. This approach helps in handling large 3D medical images while optimizing memory usage and ensuring accurate predictions.

```python
swi_callback = SlidingWindowInferenceCallback(
    model,
    dataset=val_ds, 
    num_classes=num_classes,
    overlap=0.8,
    roi_size=(96, 96, 96),
    sw_batch_size=4,
    interval=100, 
    mode="constant",
    padding_mode="constant",
    sigma_scale=0.125,
    cval=0.0,
    roi_weight_map=0.8,
    save_path="model.weights.h5"
)
```
```python
history = model.fit(
    train_ds, 
    epochs=500,
    callbacks=[
        swi_callback
    ]
)
```
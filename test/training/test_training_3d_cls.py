import keras
import tensorflow as tf

from src.medicai.models import SwinTransformer
from src.medicai.transforms import Compose, RandRotate90, Resize, ScaleIntensityRange


def create_sample_dict(image, label):
    return {"image": image, "label": label}


def transformation(sample):
    data = {"image": sample["image"], "label": sample["label"]}
    pipeline = Compose(
        [
            ScaleIntensityRange(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            Resize(keys=["image"], mode=["bilinear"], spatial_shape=(96, 96, 96), only_image=True),
            RandRotate90(keys=["image"], prob=0.1, max_k=3, spatial_axes=(1, 2)),
        ]
    )
    result = pipeline(data)
    return result.data["image"], result.data["label"]


def train_and_assert(model, dataset):
    history = model.fit(dataset, epochs=5, verbose=0)
    assert "loss" in history.history
    assert "binary_accuracy" in history.history
    assert len(history.history["loss"]) == 5
    assert len(history.history["binary_accuracy"]) == 5


def create_model_and_compile(num_classes):
    model = SwinTransformer(
        input_shape=(96, 96, 96, 1),
        num_classes=num_classes,
        classifier_activation=None,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=1e-4,
            weight_decay=1e-5,
        ),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)],
        jit_compile=False,
    )
    return model


def create_dummy_dataset(batch_size, num_classes):
    input_shape = (batch_size, 96, 96, 96, 1)
    label_shape = (batch_size, 1)
    dummy_input = tf.random.normal(input_shape)
    dummy_labels = tf.random.uniform(label_shape, minval=0, maxval=num_classes, dtype=tf.int32)
    dummy_labels = tf.cast(dummy_labels, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((dummy_input, dummy_labels))
    return dataset


def test_training_with_meta():
    num_classes = 1
    model = create_model_and_compile(num_classes)
    dataset = create_dummy_dataset(1, num_classes)
    dataset = dataset.map(create_sample_dict, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(transformation, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(1).prefetch(tf.data.AUTOTUNE)
    train_and_assert(model, dataset)

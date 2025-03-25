import keras
import tensorflow as tf

from src.medicai.losses import SparseDiceCELoss
from src.medicai.metrics import DiceMetric
from src.medicai.models import SwinUNETR
from src.medicai.transforms import (
    Compose,
    ScaleIntensityRange,
)


def transformation(sample):
    meta = {"affine": tf.eye(4)}
    data = {"image": sample["image"], "label": sample["label"]}
    pipeline = Compose(
        [
            ScaleIntensityRange(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
        ]
    )
    result = pipeline(data, meta)
    return result.data["image"], result.data["label"]


def create_sample_dict(image, label):
    return {"image": image, "label": label}


def train_and_assert(model, dataset):
    history = model.fit(dataset, epochs=5, verbose=0)
    assert "loss" in history.history
    assert "dice_score" in history.history
    assert len(history.history["loss"]) == 5
    assert len(history.history["dice_score"]) == 5


def create_model_and_compile(num_classes):
    model = SwinUNETR(
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
        metrics=[
            DiceMetric(
                num_classes=num_classes,
                include_background=True,
                reduction="mean",
                ignore_empty=True,
                smooth=1e-6,
                name="dice_score",
            )
        ],
        jit_compile=False,
    )
    return model


def create_dummy_dataset(batch_size, num_classes):
    input_shape = (batch_size, 96, 96, 96, 1)
    label_shape = (batch_size, 96, 96, 96, 1)
    dummy_input = tf.random.normal(input_shape)
    dummy_labels = tf.random.uniform(label_shape, minval=0, maxval=num_classes, dtype=tf.int32)
    dummy_labels = tf.cast(dummy_labels, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((dummy_input, dummy_labels))
    return dataset


def test_training_with_meta():
    num_classes = 4
    model = create_model_and_compile(num_classes)
    dataset = create_dummy_dataset(1, num_classes)
    dataset = dataset.map(create_sample_dict, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(transformation, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(1).prefetch(tf.data.AUTOTUNE)
    train_and_assert(model, dataset)


def test_training():
    num_classes = 4
    model = create_model_and_compile(num_classes)
    dataset = create_dummy_dataset(1, num_classes)
    dataset = dataset.batch(1).prefetch(tf.data.AUTOTUNE)
    train_and_assert(model, dataset)

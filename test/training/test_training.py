import keras
import numpy as np
import pytest
from keras import ops

from medicai.transforms import Compose, CropForeground, Flip, ScaleIntensityRange, SpatialCrop


def _backend() -> str:
    return keras.backend.backend()


def _require_backend(name: str) -> None:
    if _backend() != name:
        pytest.skip(f"Training pipeline test is only relevant for the {name!r} backend.")


def _build_binary_classifier(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(4, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(1)(x)
    return keras.Model(inputs, outputs)


def _compile_binary_classifier(model):
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(threshold=0.0)],
        jit_compile=False,
    )
    return model


def _make_phase1_samples(num_samples=4):
    images = np.linspace(0.0, 4.0, num=num_samples * 12 * 12, dtype=np.float32).reshape(
        num_samples, 12, 12, 1
    )
    labels = np.asarray([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)[:num_samples]
    return images, labels


def _make_phase2_samples(num_samples=4):
    images = np.zeros((num_samples, 12, 12, 1), dtype=np.float32)
    for index in range(num_samples):
        value = float(index + 1)
        images[index, 2:10, 2:10, 0] = value
    labels = np.asarray([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)[:num_samples]
    return images, labels


def _phase1_pipeline():
    return Compose(
        [
            ScaleIntensityRange(
                keys=["image"],
                input_min=0.0,
                input_max=4.0,
                output_min=0.0,
                output_max=1.0,
                clip=True,
                input_layout="HWC",
            ),
            Flip(keys=["image"], spatial_axis=1, input_layout="HWC"),
        ]
    )


def _phase2_pipeline_tensorflow():
    return Compose(
        [
            CropForeground(keys=["image"], source_key="image", input_layout="HWC"),
            SpatialCrop(
                keys=["image"],
                crop_size=(6, 6),
                crop_start=(1, 1),
                input_layout="HWC",
            ),
            ScaleIntensityRange(
                keys=["image"],
                input_min=0.0,
                input_max=4.0,
                output_min=0.0,
                output_max=1.0,
                clip=True,
                input_layout="HWC",
            ),
        ]
    )


def _phase2_pipeline_torch():
    return Compose(
        [
            SpatialCrop(
                keys=["image"],
                crop_size=(6, 6),
                crop_start=(3, 3),
                input_layout="HWC",
            ),
            Flip(keys=["image"], spatial_axis=1, input_layout="HWC"),
            ScaleIntensityRange(
                keys=["image"],
                input_min=0.0,
                input_max=4.0,
                output_min=0.0,
                output_max=1.0,
                clip=True,
                input_layout="HWC",
            ),
        ]
    )


@pytest.mark.integration
def test_tensorflow_training_pipeline_accepts_phase1_migrated_transforms():
    _require_backend("tensorflow")
    import tensorflow as tf

    images, labels = _make_phase1_samples()
    pipeline = _phase1_pipeline()
    model = _compile_binary_classifier(_build_binary_classifier((12, 12, 1)))

    def map_sample(image, label):
        transformed = pipeline({"image": image})
        return transformed["image"], label

    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).map(map_sample).batch(2)

    history = model.fit(dataset, epochs=1, verbose=0)

    assert "loss" in history.history
    assert len(history.history["loss"]) == 1


@pytest.mark.integration
def test_tensorflow_training_pipeline_accepts_phase2_migrated_transforms():
    _require_backend("tensorflow")
    import tensorflow as tf

    images, labels = _make_phase2_samples()
    pipeline = _phase2_pipeline_tensorflow()
    model = _compile_binary_classifier(_build_binary_classifier((6, 6, 1)))

    def map_sample(image, label):
        transformed = pipeline({"image": image})
        return transformed["image"], label

    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).map(map_sample).batch(2)

    history = model.fit(dataset, epochs=1, verbose=0)

    assert "loss" in history.history
    assert len(history.history["loss"]) == 1


@pytest.mark.integration
def test_torch_training_pipeline_accepts_phase1_migrated_transforms():
    _require_backend("torch")
    from torch.utils.data import DataLoader, Dataset

    images, labels = _make_phase1_samples()
    pipeline = _phase1_pipeline()
    model = _compile_binary_classifier(_build_binary_classifier((12, 12, 1)))

    class Phase1Dataset(Dataset):
        def __len__(self):
            return len(images)

        def __getitem__(self, index):
            transformed = pipeline({"image": images[index]})
            image = transformed["image"]
            label = ops.convert_to_tensor(labels[index], dtype="float32")
            return image, label

    loader = DataLoader(Phase1Dataset(), batch_size=2, shuffle=False)
    history = model.fit(loader, epochs=1, verbose=0)

    assert "loss" in history.history
    assert len(history.history["loss"]) == 1


@pytest.mark.integration
def test_torch_training_pipeline_accepts_phase2_migrated_transforms():
    _require_backend("torch")
    from torch.utils.data import DataLoader, Dataset

    images, labels = _make_phase2_samples()
    pipeline = _phase2_pipeline_torch()
    model = _compile_binary_classifier(_build_binary_classifier((6, 6, 1)))

    class Phase2Dataset(Dataset):
        def __len__(self):
            return len(images)

        def __getitem__(self, index):
            transformed = pipeline({"image": images[index]})
            image = transformed["image"]
            label = ops.convert_to_tensor(labels[index], dtype="float32")
            return image, label

    loader = DataLoader(Phase2Dataset(), batch_size=2, shuffle=False)
    history = model.fit(loader, epochs=1, verbose=0)

    assert "loss" in history.history
    assert len(history.history["loss"]) == 1

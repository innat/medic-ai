"""End-to-end training coverage for dictionary-structured inputs and outputs."""

from test.training.common import DatasetBuilder as make_dataset
from test.training.common import (
    build_multi_input_classification_model,
    build_multi_input_output_classification_model,
    build_multi_output_classification_model,
)

import keras
import numpy as np
import pytest

from medicai.transforms import Flip


def _require_backend():
    """Skip this integration module when no supported Keras backend is active."""
    if keras.config.backend() not in {"tensorflow", "torch", "jax"}:
        pytest.skip("Multi-input/output training requires a supported Keras backend.")


def _transformed_multi_input_data():
    """Return two keyed images after one synchronized batch transform."""
    images, labels = make_dataset().classification_2d(spatial_shape=(32, 32))
    image_2 = images * 0.5
    pipeline = Flip(
        keys=["image_1", "image_2"],
        spatial_axis=1,
        input_layout="BHWC",
    )
    result = pipeline({"image_1": images, "image_2": image_2})
    inputs = {
        "image_1": keras.ops.convert_to_numpy(result["image_1"]),
        "image_2": keras.ops.convert_to_numpy(result["image_2"]),
    }
    return inputs, labels


def _assert_one_epoch(history):
    assert len(history.history["loss"]) == 1
    assert np.isfinite(history.history["loss"][0])


@pytest.mark.integration
def test_training_supports_usual_input_with_multi_output_model():
    """Train one image input against two named classification targets."""
    _require_backend()
    images, labels = make_dataset().classification_2d(spatial_shape=(32, 32))
    targets = {
        "class_output": labels,
        "intensity_output": images.mean(axis=(1, 2, 3))[:, None],
    }
    model = build_multi_output_classification_model((32, 32, 1))

    history = model.fit(images, targets, batch_size=2, epochs=1, verbose=0)

    _assert_one_epoch(history)


@pytest.mark.integration
def test_training_supports_multi_input_with_usual_output_model():
    """Train two named image inputs against one classification target."""
    _require_backend()
    inputs, labels = _transformed_multi_input_data()
    model = build_multi_input_classification_model((32, 32, 1))

    history = model.fit(inputs, labels, batch_size=2, epochs=1, verbose=0)

    _assert_one_epoch(history)


@pytest.mark.integration
def test_training_supports_multi_input_with_multi_output_model():
    """Train two transformed image inputs against two named targets."""
    _require_backend()
    inputs, labels = _transformed_multi_input_data()
    targets = {
        "class_output": labels,
        "intensity_output": inputs["image_1"].mean(axis=(1, 2, 3))[:, None],
    }
    model = build_multi_input_output_classification_model((32, 32, 1))

    history = model.fit(inputs, targets, batch_size=2, epochs=1, verbose=0)

    _assert_one_epoch(history)

import numpy as np
import pytest
from keras import ops

from medicai.transforms import RandomTranslate, TensorBundle


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_random_translate_preserves_shape_and_aligns_selected_keys():
    image = as_tensor(np.arange(2 * 8 * 9, dtype=np.float32).reshape(2, 8, 9, 1))
    label = image * 2.0
    transform = RandomTranslate(
        keys=["image", "label"],
        factor=0.0,
        prob=1.0,
        input_layout="BHWC",
        interpolation={"image": "nearest", "label": "nearest"},
    )

    output = transform(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(output["image"])) == (2, 8, 9, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(output["label"]),
        ops.convert_to_numpy(output["image"]) * 2.0,
    )


@pytest.mark.unit
def test_random_translate_accepts_3d_axis_factors():
    image = as_tensor(np.zeros((3, 4, 5, 1), dtype=np.float32))
    transform = RandomTranslate(
        keys=["image"],
        factor={"z": 0.1, "x": 0.2, "y": 0.3},
        prob=1.0,
        input_layout="DHWC",
    )

    output = transform(TensorBundle({"image": image}))

    assert tuple(ops.shape(output["image"])) == (3, 4, 5, 1)


@pytest.mark.unit
def test_random_translate_samples_independent_offsets_per_batch_item(monkeypatch):
    image = as_tensor(np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6, 1))
    transform = RandomTranslate(
        keys=["image"],
        factor={"y": 0.2, "x": 0.0},
        prob=1.0,
        input_layout="BHWC",
        seed=3,
    )
    calls = 0

    def sample_uniform(*, shape, minval=0.0, maxval=1.0, dtype="float32"):
        del minval, maxval
        nonlocal calls
        calls += 1
        values = [0.0, 0.0] if calls == 1 else [0.0, 1.0]
        return as_tensor(values, dtype=dtype)

    monkeypatch.setattr(transform, "random_uniform", sample_uniform)
    output = transform(TensorBundle({"image": image}))
    offsets = output.get_applied_transforms()[-1]["params"]["offsets"]["y"]

    assert not np.isclose(ops.convert_to_numpy(offsets)[0], ops.convert_to_numpy(offsets)[1])

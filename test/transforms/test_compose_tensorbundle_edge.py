import numpy as np
import pytest
from keras import ops

from medicai.transforms import Compose, TensorBundle


@pytest.mark.unit
def test_compose_converts_numpy_for_data_and_meta():
    image = np.ones((4, 4, 1), dtype=np.float32)
    label = np.zeros((4, 4, 1), dtype=np.int32)
    meta = {"affine": np.eye(4, dtype=np.float32)}

    pipeline = Compose([])
    output = pipeline({"image": image, "label": label}, meta)

    assert isinstance(output, TensorBundle)
    assert hasattr(output["image"], "shape")
    assert hasattr(output["label"], "shape")
    assert hasattr(output["affine"], "shape")


@pytest.mark.unit
def test_compose_applies_transforms_in_order():
    def add_one(bundle):
        bundle["image"] = bundle["image"] + 1.0
        return bundle

    def multiply_by_two(bundle):
        bundle["image"] = bundle["image"] * 2.0
        return bundle

    pipeline = Compose([add_one, multiply_by_two])
    output = pipeline({"image": ops.convert_to_tensor(np.array([[[1.0]]], dtype=np.float32))})

    # (1 + 1) * 2 = 4
    assert float(output["image"].numpy().item()) == 4.0


@pytest.mark.unit
def test_tensorbundle_missing_key_raises():
    bundle = TensorBundle({"image": ops.zeros((2, 2, 1), dtype="float32")}, {"meta_key": "value"})
    with pytest.raises(KeyError):
        _ = bundle["unknown"]


@pytest.mark.unit
def test_tensorbundle_repr_contains_data_and_meta():
    bundle = TensorBundle({"image": ops.zeros((2, 2, 1), dtype="float32")}, {"spacing": [1.0, 1.0]})
    value = repr(bundle)
    assert "MetaTensor" in value
    assert "spacing" in value

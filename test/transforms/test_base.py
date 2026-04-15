import numpy as np
import pytest
from keras import ops

from medicai.transforms import TensorBundle


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_tensorbundle_creation_and_meta_access():
    data = {
        "image": as_tensor(np.zeros((4, 4, 1), dtype=np.float32)),
        "label": as_tensor(np.ones((4, 4, 1), dtype=np.int32)),
    }
    meta = {"pixdim": [1.0, 1.0, 1.0], "id": "case-a"}
    bundle = TensorBundle(data, meta)

    assert bundle.data["image"] is data["image"]
    assert bundle.meta["id"] == "case-a"
    assert bundle["pixdim"] == [1.0, 1.0, 1.0]


@pytest.mark.unit
def test_tensorbundle_setitem_routes_to_data_or_meta():
    bundle = TensorBundle({"image": as_tensor(np.zeros((2, 2, 1), dtype=np.float32))})
    new_image = as_tensor(np.ones((2, 2, 1), dtype=np.float32))
    bundle["image"] = new_image
    bundle["affine"] = np.eye(4, dtype=np.float32).tolist()

    np.testing.assert_allclose(
        ops.convert_to_numpy(bundle["image"]),
        ops.convert_to_numpy(new_image),
    )
    assert "affine" in bundle.meta


@pytest.mark.unit
def test_tensorbundle_missing_key_raises_keyerror():
    bundle = TensorBundle({"image": as_tensor(np.zeros((2, 2, 1), dtype=np.float32))})
    with pytest.raises(KeyError):
        _ = bundle["unknown"]

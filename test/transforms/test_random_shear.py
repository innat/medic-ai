import numpy as np
import pytest
from keras import ops

from medicai.transforms import RandomShear, TensorBundle


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_random_shear_preserves_shape_and_aligns_selected_keys():
    image = as_tensor(np.arange(2 * 8 * 9, dtype=np.float32).reshape(2, 8, 9, 1))
    label = image * 2.0
    transform = RandomShear(
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
def test_random_shear_accepts_3d_axis_pair_factors():
    image = as_tensor(np.zeros((2, 3, 4, 5, 1), dtype=np.float32))
    transform = RandomShear(
        keys=["image"],
        factor={axis: 0.1 for axis in ("zy", "zx", "yz", "yx", "xz", "xy")},
        prob=1.0,
        input_layout="BDHWC",
    )

    output = transform(TensorBundle({"image": image}))

    assert tuple(ops.shape(output["image"])) == (2, 3, 4, 5, 1)

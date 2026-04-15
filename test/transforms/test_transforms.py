import numpy as np
import pytest
from keras import ops

from medicai.transforms import (
    Compose,
    Orientation,
    RandRotate90,
    RandShiftIntensity,
    Resize,
    ScaleIntensityRange,
    Spacing,
    TensorBundle,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_resize_transform_for_2d_and_3d():
    inputs_2d = TensorBundle(
        {
            "image": as_tensor(np.random.randn(1, 32, 32, 1).astype(np.float32)),
            "label": as_tensor(np.random.randint(0, 2, (1, 32, 32, 1)).astype(np.float32)),
        }
    )
    out_2d = Resize(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        spatial_shape=(24, 20),
    )(inputs_2d)
    assert tuple(ops.shape(out_2d["image"])) == (1, 24, 20, 1)
    assert tuple(ops.shape(out_2d["label"])) == (1, 24, 20, 1)

    inputs_3d = TensorBundle(
        {
            "image": as_tensor(np.random.randn(16, 16, 16, 1).astype(np.float32)),
            "label": as_tensor(np.random.randint(0, 2, (16, 16, 16, 1)).astype(np.float32)),
        }
    )
    out_3d = Resize(keys=["image", "label"], spatial_shape=(8, 10, 12))(inputs_3d)
    assert tuple(ops.shape(out_3d["image"])) == (8, 10, 12, 1)
    assert tuple(ops.shape(out_3d["label"])) == (8, 10, 12, 1)


@pytest.mark.unit
def test_scale_intensity_range_handles_flat_input():
    image = as_tensor(np.full((1, 2, 2), 5.0, dtype=np.float32))
    out = ScaleIntensityRange(keys=["image"], a_min=5.0, a_max=5.0, b_min=0.0, b_max=1.0)(
        TensorBundle({"image": image})
    )
    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), 0.0, rtol=1e-6)


@pytest.mark.unit
def test_rand_shift_intensity_preserves_shape_and_range():
    image = as_tensor(np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32))
    out = RandShiftIntensity(keys=["image"], offsets=(-0.2, 0.8), prob=1.0)(TensorBundle({"image": image}))
    shifted = ops.convert_to_numpy(out["image"])
    original = ops.convert_to_numpy(image)

    assert shifted.shape == (1, 2, 2, 1)
    assert np.all(shifted >= original - 0.8)
    assert np.all(shifted <= original + 0.8)


@pytest.mark.unit
def test_rand_rotate90_preserves_shape():
    image = as_tensor(np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2))
    out = RandRotate90(keys=["image"], prob=1.0, max_k=3)(TensorBundle({"image": image}))
    assert tuple(ops.shape(out["image"])) == (1, 2, 2, 2)


@pytest.mark.unit
def test_compose_spacing_orientation_pipeline():
    image = as_tensor(np.random.randn(8, 8, 8, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (8, 8, 8, 1)).astype(np.float32))
    affine = as_tensor(np.eye(4, dtype=np.float32))

    output = Compose(
        [
            Spacing(keys=["image", "label"], pixdim=[0.5, 0.5, 0.5]),
            Orientation(keys=["image", "label"], axcodes="RAS"),
        ]
    )({"image": image, "label": label}, {"affine": affine, "pixdim": [1.0, 1.0, 1.0]})

    assert tuple(ops.shape(output["image"])) == (16, 16, 16, 1)
    assert tuple(ops.shape(output["label"])) == (16, 16, 16, 1)

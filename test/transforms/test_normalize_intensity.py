import numpy as np
import pytest
from keras import ops
from medicai.transforms.random import random_elastic_transform as elastic_module

from medicai.transforms import (
    Compose,
    CropForeground,
    Flip,
    NormalizeIntensity,
    Orientation,
    RandomCropByPosNegLabel,
    RandomCutOut,
    RandomElasticTransform,
    RandomFlip,
    RandomRotate,
    RandomRotate90,
    RandomShiftIntensity,
    RandomSpatialCrop,
    Resize,
    Rotate90,
    ScaleIntensityRange,
    ShiftIntensity,
    SignalFillEmpty,
    Spacing,
    SpatialCrop,
    TensorBundle,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_normalize_intensity_records_trace():
    image = as_tensor(np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32))
    out = NormalizeIntensity(keys=["image"], input_layout="HWC")(TensorBundle({"image": image}))

    assert tuple(ops.shape(out["image"])) == (2, 2, 1)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "NormalizeIntensity"
    assert trace["random"] is False
    assert trace["params"]["input_layout"] == "HWC"


@pytest.mark.unit
def test_normalize_intensity_nonzero_preserves_zero_background():
    image = as_tensor(np.array([[[0.0], [1.0]], [[3.0], [0.0]]], dtype=np.float32))
    out = NormalizeIntensity(keys=["image"], nonzero=True, input_layout="HWC")(
        TensorBundle({"image": image})
    )

    normalized = ops.convert_to_numpy(out["image"])
    assert normalized[0, 0, 0] == 0.0
    assert normalized[1, 1, 0] == 0.0


@pytest.mark.unit
def test_normalize_intensity_channel_wise_with_fixed_stats():
    image = as_tensor(np.array([[[1.0, 5.0], [3.0, 9.0]]], dtype=np.float32))
    out = NormalizeIntensity(
        keys=["image"],
        offset=1.0,
        scale=2.0,
        channel_wise=True,
        input_layout="HWC",
    )(TensorBundle({"image": image}))

    np.testing.assert_allclose(
        ops.convert_to_numpy(out["image"]),
        np.array([[[0.0, 2.0], [1.0, 4.0]]], dtype=np.float32),
    )


@pytest.mark.unit
def test_normalize_intensity_channel_wise_nonzero_leaves_empty_channel_unchanged():
    image = as_tensor(np.array([[[0.0, 1.0], [0.0, 3.0]]], dtype=np.float32))
    out = NormalizeIntensity(
        keys=["image"],
        nonzero=True,
        channel_wise=True,
        input_layout="HWC",
    )(TensorBundle({"image": image}))

    normalized = ops.convert_to_numpy(out["image"])
    np.testing.assert_allclose(normalized[..., 0], np.array([[0.0, 0.0]], dtype=np.float32))
    assert np.isfinite(normalized).all()


@pytest.mark.unit
def test_normalize_intensity_channel_wise_nonzero_preserves_zero_background():
    image = as_tensor(np.array([[[0.0], [1.0]], [[3.0], [0.0]]], dtype=np.float32))
    out = NormalizeIntensity(
        keys=["image"],
        nonzero=True,
        channel_wise=True,
        input_layout="HWC",
    )(TensorBundle({"image": image}))

    normalized = ops.convert_to_numpy(out["image"])
    assert normalized[0, 0, 0] == 0.0
    assert normalized[1, 1, 0] == 0.0


@pytest.mark.unit
def test_normalize_intensity_supports_batch_mode():
    image_2d = as_tensor(np.ones((2, 3, 4, 1), dtype=np.float32))
    image_3d = as_tensor(np.ones((2, 3, 4, 5, 1), dtype=np.float32))

    out_2d = NormalizeIntensity(keys=["image"], input_layout="BHWC")(
        TensorBundle({"image": image_2d})
    )
    out_3d = NormalizeIntensity(keys=["image"], input_layout="BDHWC")(
        TensorBundle({"image": image_3d})
    )

    assert tuple(ops.shape(out_2d["image"])) == (2, 3, 4, 1)
    assert tuple(ops.shape(out_3d["image"])) == (2, 3, 4, 5, 1)
    assert np.isfinite(ops.convert_to_numpy(out_2d["image"])).all()
    assert np.isfinite(ops.convert_to_numpy(out_3d["image"])).all()


@pytest.mark.unit
def test_normalize_intensity_accepts_input_layout():
    image = as_tensor(np.ones((2, 3, 4, 1), dtype=np.float32))

    out = NormalizeIntensity(keys=["image"], input_layout="bhwc")(TensorBundle({"image": image}))

    assert tuple(ops.shape(out["image"])) == (2, 3, 4, 1)
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_normalize_intensity_accepts_plain_numpy_inputs():
    image = np.arange(12, dtype=np.float32).reshape(3, 4, 1)

    out = NormalizeIntensity(keys=["image"], input_layout="HWC")({"image": image})

    assert tuple(ops.shape(out["image"])) == (3, 4, 1)
    assert ops.is_tensor(out["image"])



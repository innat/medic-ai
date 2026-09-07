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
def test_spatial_crop_accepts_input_layout():
    image = as_tensor(np.random.randn(2, 8, 8, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (2, 8, 8, 1)).astype(np.float32))

    out = SpatialCrop(
        keys=["image", "label"],
        crop_size=(3, 4),
        crop_start=(1, 1),
        input_layout="BHWC",
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (2, 3, 4, 1)
    assert tuple(ops.shape(out["label"])) == (2, 3, 4, 1)
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_spatial_crop_supports_2d_and_3d_channel_last_tensors():
    image_2d = as_tensor(np.arange(30, dtype=np.float32).reshape(5, 6, 1))
    image_3d = as_tensor(np.arange(120, dtype=np.float32).reshape(4, 5, 6, 1))

    out_2d = SpatialCrop(keys=["image"], crop_size=(3, 4), crop_start=(1, 1), input_layout="HWC")(
        TensorBundle({"image": image_2d})
    )
    out_3d = SpatialCrop(
        keys=["image"],
        crop_size=(2, 3, 4),
        crop_center=(2, 2, 3),
        input_layout="DHWC",
    )(TensorBundle({"image": image_3d}))

    assert tuple(ops.shape(out_2d["image"])) == (3, 4, 1)
    assert tuple(ops.shape(out_3d["image"])) == (2, 3, 4, 1)
    assert out_2d.get_applied_transforms()[-1]["name"] == "SpatialCrop"


@pytest.mark.unit
def test_spatial_crop_supports_batch_mode_with_shared_crop():
    image_2d = as_tensor(np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6, 1))
    image_3d = as_tensor(np.arange(2 * 4 * 5 * 6, dtype=np.float32).reshape(2, 4, 5, 6, 1))

    out_2d = SpatialCrop(
        keys=["image"],
        crop_size=(3, 4),
        crop_start=(1, 1),
        input_layout="BHWC",
    )(TensorBundle({"image": image_2d}))
    out_3d = SpatialCrop(
        keys=["image"],
        crop_size=(2, 3, 4),
        crop_start=(1, 1, 1),
        input_layout="BDHWC",
    )(TensorBundle({"image": image_3d}))

    assert tuple(ops.shape(out_2d["image"])) == (2, 3, 4, 1)
    assert tuple(ops.shape(out_3d["image"])) == (2, 2, 3, 4, 1)
    assert out_2d.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_spatial_crop_inverse_restores_original_canvas_for_2d():
    image = as_tensor(np.zeros((5, 6, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[1:4, 1:5, 0] = np.arange(12, dtype=np.float32).reshape(3, 4)
    image = as_tensor(image_np)

    transform = SpatialCrop(keys=["image"], crop_size=(3, 4), crop_start=(1, 1), input_layout="HWC")
    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(restored["image"])) == (5, 6, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_spatial_crop_inverse_restores_original_canvas_for_batch_mode():
    image = as_tensor(np.zeros((2, 5, 6, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[:, 1:4, 1:5, 0] = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    image = as_tensor(image_np)

    transform = SpatialCrop(
        keys=["image"],
        crop_size=(3, 4),
        crop_start=(1, 1),
        input_layout="BHWC",
    )
    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(restored["image"])) == (2, 5, 6, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_spatial_crop_inverse_restores_original_canvas_for_3d():
    image = as_tensor(np.zeros((4, 5, 6, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[1:3, 1:4, 1:5, 0] = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    image = as_tensor(image_np)

    transform = SpatialCrop(
        keys=["image"],
        crop_size=(2, 3, 4),
        crop_start=(1, 1, 1),
        input_layout="DHWC",
    )
    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(restored["image"])) == (4, 5, 6, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_spatial_crop_inverse_places_prediction_back_on_original_canvas():
    image = as_tensor(np.arange(30, dtype=np.float32).reshape(5, 6, 1))
    label = as_tensor(np.zeros((5, 6, 1), dtype=np.float32))
    transform = SpatialCrop(
        keys=["image", "label"],
        crop_size=(3, 4),
        crop_start=(1, 1),
        input_layout="HWC",
    )

    forward = transform(TensorBundle({"image": image, "label": label}))
    prediction = ops.ones_like(forward["label"])
    prediction_bundle = TensorBundle(
        {"image": forward["image"], "label": prediction},
        dict(forward.meta),
    )
    prediction_bundle.meta["applied_transforms"] = list(forward.get_applied_transforms())

    restored = transform.inverse(prediction_bundle)

    expected = np.zeros((5, 6, 1), dtype=np.float32)
    expected[1:4, 1:5, 0] = 1.0
    np.testing.assert_allclose(ops.convert_to_numpy(restored["label"]), expected)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"])[1:4, 1:5, :],
        ops.convert_to_numpy(forward["image"]),
    )


@pytest.mark.unit
def test_spatial_crop_inverse_without_trace_is_noop():
    bundle = TensorBundle({"image": as_tensor(np.ones((4, 5, 1), dtype=np.float32))})
    transform = SpatialCrop(keys=["image"], crop_size=(2, 2), input_layout="HWC")

    restored = transform.inverse(bundle)

    assert restored is bundle


@pytest.mark.unit
def test_spatial_crop_validates_exclusive_start_and_center():
    with pytest.raises(
        ValueError, match="Only one of `crop_start` or `crop_center` may be provided"
    ):
        SpatialCrop(
            keys=["image"],
            crop_size=(2, 2),
            crop_start=(0, 0),
            crop_center=(1, 1),
            input_layout="HWC",
        )


@pytest.mark.unit
def test_spatial_crop_nonpositive_roi_uses_full_extent():
    image = as_tensor(np.arange(30, dtype=np.float32).reshape(5, 6, 1))
    out = SpatialCrop(keys=["image"], crop_size=(0, -1), input_layout="HWC")(
        TensorBundle({"image": image})
    )

    assert tuple(ops.shape(out["image"])) == (5, 6, 1)


@pytest.mark.unit
def test_spatial_crop_records_per_key_crop_bounds_for_mixed_shapes():
    image = as_tensor(np.arange(36, dtype=np.float32).reshape(6, 6, 1))
    label = as_tensor(np.arange(16, dtype=np.float32).reshape(4, 4, 1))
    transform = SpatialCrop(
        keys=["image", "label"],
        crop_size=(4, 4),
        crop_center=(4, 4),
        input_layout="HWC",
    )

    forward = transform(TensorBundle({"image": image, "label": label}))
    trace = forward.get_applied_transforms()[-1]

    image_start = ops.convert_to_numpy(trace["params"]["crop_start"]["image"])
    label_start = ops.convert_to_numpy(trace["params"]["crop_start"]["label"])
    assert image_start.tolist() == [2, 2]
    assert label_start.tolist() == [0, 0]



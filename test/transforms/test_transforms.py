import numpy as np
import pytest
from keras import ops

from medicai.transforms import (
    Compose,
    CropForeground,
    Flip,
    NormalizeIntensity,
    Orientation,
    RandCropByPosNegLabel,
    RandCutOut,
    RandFlip,
    RandRotate,
    RandRotate90,
    RandShiftIntensity,
    RandSpatialCrop,
    Rotate90,
    Resize,
    ScaleIntensityRange,
    SignalFillEmpty,
    ShiftIntensity,
    SpatialCrop,
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
    trace = out_3d.get_applied_transforms()[-1]
    assert trace["name"] == "Resize"
    assert trace["invertible"] is True


@pytest.mark.unit
def test_resize_inverse_restores_original_spatial_shape():
    image = as_tensor(np.random.randn(6, 8, 1).astype(np.float32))
    resize = Resize(keys=["image"], mode="bilinear", spatial_shape=(3, 4))

    forward = resize(TensorBundle({"image": image}))
    restored = resize.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(forward["image"])) == (3, 4, 1)
    assert tuple(ops.shape(restored["image"])) == (6, 8, 1)


@pytest.mark.unit
def test_spacing_records_trace_and_inverse_restores_original_shape():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))
    affine = as_tensor(np.eye(4, dtype=np.float32))

    spacing = Spacing(keys=["image", "label"], pixdim=(0.5, 0.5, 0.5))
    forward = spacing(TensorBundle({"image": image, "label": label}, {"affine": affine}))
    trace = forward.get_applied_transforms()[-1]

    assert trace["name"] == "Spacing"
    assert trace["invertible"] is True
    restored = spacing.inverse(TensorBundle({"image": forward["image"], "label": forward["label"]}, forward.meta))
    assert tuple(ops.shape(restored["image"])) == (4, 5, 6, 1)
    assert tuple(ops.shape(restored["label"])) == (4, 5, 6, 1)


@pytest.mark.unit
def test_orientation_records_trace_and_inverse_restores_shape():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))
    affine = as_tensor(np.eye(4, dtype=np.float32))

    orientation = Orientation(keys=["image", "label"], axcodes="RAS")
    forward = orientation(TensorBundle({"image": image, "label": label}, {"affine": affine}))
    trace = forward.get_applied_transforms()[-1]

    assert trace["name"] == "Orientation"
    assert trace["invertible"] is True
    restored = orientation.inverse(
        TensorBundle({"image": forward["image"], "label": forward["label"]}, forward.meta)
    )
    assert tuple(ops.shape(restored["image"])) == (4, 5, 6, 1)
    assert tuple(ops.shape(restored["label"])) == (4, 5, 6, 1)


@pytest.mark.unit
def test_scale_intensity_range_handles_flat_input():
    image = as_tensor(np.full((1, 2, 2), 5.0, dtype=np.float32))
    out = ScaleIntensityRange(keys=["image"], a_min=5.0, a_max=5.0, b_min=0.0, b_max=1.0)(
        TensorBundle({"image": image})
    )
    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), 0.0, rtol=1e-6)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "ScaleIntensityRange"
    assert trace["random"] is False


@pytest.mark.unit
def test_normalize_intensity_records_trace():
    image = as_tensor(np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32))
    out = NormalizeIntensity(keys=["image"])(TensorBundle({"image": image}))

    assert tuple(ops.shape(out["image"])) == (2, 2, 1)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "NormalizeIntensity"
    assert trace["random"] is False


@pytest.mark.unit
def test_signal_fill_empty_replaces_invalid_values_and_records_trace():
    image = as_tensor(np.array([[[np.nan], [np.inf]], [[-np.inf], [1.0]]], dtype=np.float32))
    out = SignalFillEmpty(keys=["image"], replacement=0.0)(TensorBundle({"image": image}))

    filled = ops.convert_to_numpy(out["image"])
    assert np.isfinite(filled).all()
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "SignalFillEmpty"
    assert trace["random"] is False


@pytest.mark.unit
def test_rand_shift_intensity_preserves_shape_and_range():
    image = as_tensor(np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32))
    out = RandShiftIntensity(keys=["image"], offsets=(-0.2, 0.8), prob=1.0)(
        TensorBundle({"image": image})
    )
    shifted = ops.convert_to_numpy(out["image"])
    original = ops.convert_to_numpy(image)

    assert shifted.shape == (1, 2, 2, 1)
    assert np.all(shifted >= original - 0.8)
    assert np.all(shifted <= original + 0.8)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandShiftIntensity"
    assert bool(ops.convert_to_numpy(trace["applied"]))
    assert trace["random"] is True
    assert trace["invertible"] is False
    assert trace["kernel"] == "ShiftIntensity"


@pytest.mark.unit
def test_shift_intensity_supports_2d_and_3d_channel_last_tensors():
    image_2d = as_tensor(np.ones((6, 5, 1), dtype=np.float32))
    image_3d = as_tensor(np.ones((4, 6, 5, 1), dtype=np.float32))

    out_2d = ShiftIntensity(keys=["image"], offsets=0.5)(TensorBundle({"image": image_2d}))
    out_3d = ShiftIntensity(keys=["image"], offsets=-0.25)(TensorBundle({"image": image_3d}))

    np.testing.assert_allclose(ops.convert_to_numpy(out_2d["image"]), 1.5)
    np.testing.assert_allclose(ops.convert_to_numpy(out_3d["image"]), 0.75)


@pytest.mark.unit
def test_spatial_crop_supports_2d_and_3d_channel_last_tensors():
    image_2d = as_tensor(np.arange(30, dtype=np.float32).reshape(5, 6, 1))
    image_3d = as_tensor(np.arange(120, dtype=np.float32).reshape(4, 5, 6, 1))

    out_2d = SpatialCrop(keys=["image"], roi_size=(3, 4), roi_start=(1, 1))(
        TensorBundle({"image": image_2d})
    )
    out_3d = SpatialCrop(keys=["image"], roi_size=(2, 3, 4), roi_center=(2, 2, 3))(
        TensorBundle({"image": image_3d})
    )

    assert tuple(ops.shape(out_2d["image"])) == (3, 4, 1)
    assert tuple(ops.shape(out_3d["image"])) == (2, 3, 4, 1)
    assert out_2d.get_applied_transforms()[-1]["name"] == "SpatialCrop"


@pytest.mark.unit
def test_rand_spatial_crop_supports_2d_and_3d_channel_last_tensors():
    image_2d = as_tensor(np.arange(30, dtype=np.float32).reshape(5, 6, 1))
    image_3d = as_tensor(np.arange(120, dtype=np.float32).reshape(4, 5, 6, 1))

    out_2d = RandSpatialCrop(keys=["image"], roi_size=(3, 4), random_center=False)(
        TensorBundle({"image": image_2d})
    )
    out_3d = RandSpatialCrop(keys=["image"], roi_size=(2, 3, 4), random_center=False)(
        TensorBundle({"image": image_3d})
    )

    assert tuple(ops.shape(out_2d["image"])) == (3, 4, 1)
    assert tuple(ops.shape(out_3d["image"])) == (2, 3, 4, 1)
    assert out_3d.get_applied_transforms()[-1]["kernel"] == "SpatialCrop"


@pytest.mark.unit
def test_crop_foreground_supports_2d_and_3d_channel_last_tensors():
    image_2d = as_tensor(
        np.array(
            [
                [[0.0], [0.0], [0.0], [0.0]],
                [[0.0], [1.0], [1.0], [0.0]],
                [[0.0], [1.0], [1.0], [0.0]],
                [[0.0], [0.0], [0.0], [0.0]],
            ],
            dtype=np.float32,
        )
    )
    image_3d = as_tensor(np.pad(np.ones((2, 2, 2, 1), dtype=np.float32), ((1, 1), (1, 1), (1, 1), (0, 0))))

    out_2d = CropForeground(keys=["image"], source_key="image")(TensorBundle({"image": image_2d}))
    out_3d = CropForeground(keys=["image"], source_key="image")(TensorBundle({"image": image_3d}))

    assert tuple(ops.shape(out_2d["image"])) == (2, 2, 1)
    assert tuple(ops.shape(out_3d["image"])) == (2, 2, 2, 1)
    assert out_2d.get_applied_transforms()[-1]["name"] == "CropForeground"


@pytest.mark.unit
def test_rand_crop_by_pos_neg_label_uses_spatial_crop_kernel():
    image = as_tensor(np.random.randn(6, 6, 6, 1).astype(np.float32))
    label = as_tensor(np.zeros((6, 6, 6, 1), dtype=np.float32))
    label = ops.convert_to_tensor(
        np.pad(np.ones((2, 2, 2, 1), dtype=np.float32), ((2, 2), (2, 2), (2, 2), (0, 0)))
    )

    out = RandCropByPosNegLabel(
        keys=["image", "label"],
        spatial_size=(3, 3, 3),
        pos=1,
        neg=1,
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (3, 3, 3, 1)
    assert tuple(ops.shape(out["label"])) == (3, 3, 3, 1)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandCropByPosNegLabel"
    assert trace["kernel"] == "SpatialCrop"
    assert trace["random"] is True


@pytest.mark.unit
def test_flip_supports_2d_and_3d_and_records_inverse_trace():
    image_2d = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    image_3d = as_tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1))

    flip_2d = Flip(keys=["image"], spatial_axis=1)
    flip_3d = Flip(keys=["image"], spatial_axis=(0, 2))

    out_2d = flip_2d(TensorBundle({"image": image_2d}))
    out_3d = flip_3d(TensorBundle({"image": image_3d}))

    np.testing.assert_allclose(
        ops.convert_to_numpy(out_2d["image"]),
        ops.convert_to_numpy(image_2d)[:, ::-1, :],
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(out_3d["image"]),
        ops.convert_to_numpy(image_3d)[::-1, :, ::-1, :],
    )
    trace = out_2d.get_applied_transforms()[-1]
    assert trace["name"] == "Flip"
    assert trace["applied"] is True
    assert trace["random"] is False
    assert trace["invertible"] is True
    np.testing.assert_allclose(
        ops.convert_to_numpy(flip_2d.inverse(TensorBundle({"image": out_2d["image"]}))["image"]),
        ops.convert_to_numpy(image_2d),
    )


@pytest.mark.unit
def test_rotate90_supports_2d_and_3d_and_records_inverse_trace():
    image_2d = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    image_3d = as_tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1))

    rotate_2d = Rotate90(keys=["image"], k=1)
    rotate_3d = Rotate90(keys=["image"], k=3, spatial_axes=(1, 2))

    out_2d = rotate_2d(TensorBundle({"image": image_2d}))
    out_3d = rotate_3d(TensorBundle({"image": image_3d}))

    expected_2d = np.rot90(ops.convert_to_numpy(image_2d), k=1, axes=(0, 1))
    expected_3d = np.rot90(ops.convert_to_numpy(image_3d), k=3, axes=(1, 2))

    np.testing.assert_allclose(ops.convert_to_numpy(out_2d["image"]), expected_2d)
    np.testing.assert_allclose(ops.convert_to_numpy(out_3d["image"]), expected_3d)
    trace = out_3d.get_applied_transforms()[-1]
    assert trace["params"]["k"] == 3
    assert trace["applied"] is True
    assert trace["random"] is False
    assert trace["invertible"] is True
    np.testing.assert_allclose(
        ops.convert_to_numpy(rotate_2d.inverse(TensorBundle({"image": out_2d["image"]}))["image"]),
        ops.convert_to_numpy(image_2d),
    )


@pytest.mark.unit
def test_rand_rotate90_preserves_shape():
    image = as_tensor(np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2))
    out = RandRotate90(keys=["image"], prob=1.0, max_k=3)(TensorBundle({"image": image}))
    assert tuple(ops.shape(out["image"])) == (1, 2, 2, 2)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandRotate90"
    assert bool(ops.convert_to_numpy(trace["applied"]))
    assert trace["random"] is True
    assert trace["invertible"] is False
    assert trace["kernel"] == "Rotate90"


@pytest.mark.unit
def test_rand_rotate_preserves_shape_and_records_trace():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))

    out = RandRotate(keys=["image", "label"], factor=0.2, prob=1.0)(
        TensorBundle({"image": image, "label": label})
    )

    assert tuple(ops.shape(out["image"])) == (4, 5, 6, 1)
    assert tuple(ops.shape(out["label"])) == (4, 5, 6, 1)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandRotate"
    assert bool(ops.convert_to_numpy(trace["applied"]))
    assert trace["random"] is True
    assert trace["invertible"] is False
    assert trace["kernel"] == "rotate_volume"


@pytest.mark.unit
def test_rand_cutout_preserves_shape_and_records_trace():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))

    out = RandCutOut(
        keys=["image", "label"],
        mask_size=(2, 2),
        num_cuts=2,
        prob=1.0,
        fill_mode="constant",
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (4, 5, 6, 1)
    assert tuple(ops.shape(out["label"])) == (4, 5, 6, 1)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandCutOut"
    assert bool(ops.convert_to_numpy(trace["applied"]))
    assert trace["random"] is True
    assert trace["invertible"] is False
    assert trace["kernel"] == "cutout_mask"


@pytest.mark.unit
def test_rand_flip_records_random_trace():
    image = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    out = RandFlip(keys=["image"], prob=1.0, spatial_axis=1)(TensorBundle({"image": image}))

    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandFlip"
    assert bool(ops.convert_to_numpy(trace["applied"]))
    assert trace["random"] is True
    assert trace["invertible"] is False
    assert trace["kernel"] == "Flip"


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

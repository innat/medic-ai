import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from medicai.transforms import (
    Compose,
    CropForeground,
    Flip,
    NormalizeIntensity,
    Orientation,
    RandomCropByPosNegLabel,
    RandomCutOut,
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
@pytest.mark.parametrize(
    ("interpolation", "target_shape", "error"),
    [
        (("bilinear", "nearest"), (4,), "`target_shape` must be 2D or 3D"),
        ("trilinear", (4, 5), "Invalid interpolation 'trilinear' for 2D input"),
        ("bilinear", (4, 5, 6), "Invalid interpolation 'bilinear' for 3D input"),
    ],
)
def test_resize_validates_rank_and_interpolation(interpolation, target_shape, error):
    with pytest.raises(ValueError, match=error):
        Resize(keys=["image", "label"], interpolation=interpolation, target_shape=target_shape)


@pytest.mark.unit
def test_resize_accepts_mapping_mode_and_allow_missing_keys():
    image = as_tensor(np.random.randn(5, 6, 1).astype(np.float32))
    transform = Resize(
        keys=["image", "label"],
        interpolation={"image": "bilinear", "label": "nearest"},
        target_shape=(3, 4),
        allow_missing_keys=True,
    )

    out = transform(TensorBundle({"image": image}))

    assert tuple(ops.shape(out["image"])) == (3, 4, 1)
    assert "label" not in out.data


@pytest.mark.unit
def test_resize_rejects_mapping_without_all_requested_keys():
    with pytest.raises(ValueError, match="Missing interpolation mode for keys"):
        Resize(
            keys=["image", "label"],
            interpolation={"image": "bilinear"},
            target_shape=(3, 4),
        )


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
        interpolation=("bilinear", "nearest"),
        target_shape=(24, 20),
    )(inputs_2d)
    assert tuple(ops.shape(out_2d["image"])) == (1, 24, 20, 1)
    assert tuple(ops.shape(out_2d["label"])) == (1, 24, 20, 1)

    inputs_3d = TensorBundle(
        {
            "image": as_tensor(np.random.randn(16, 16, 16, 1).astype(np.float32)),
            "label": as_tensor(np.random.randint(0, 2, (16, 16, 16, 1)).astype(np.float32)),
        }
    )
    out_3d = Resize(
        keys=["image", "label"],
        interpolation=("trilinear", "nearest"),
        target_shape=(8, 10, 12),
    )(inputs_3d)
    assert tuple(ops.shape(out_3d["image"])) == (8, 10, 12, 1)
    assert tuple(ops.shape(out_3d["label"])) == (8, 10, 12, 1)
    trace = out_3d.get_applied_transforms()[-1]
    assert trace["name"] == "Resize"
    assert trace["invertible"] is True


@pytest.mark.unit
def test_resize_inverse_restores_original_spatial_shape():
    image = as_tensor(np.random.randn(6, 8, 1).astype(np.float32))
    resize = Resize(keys=["image"], interpolation="bilinear", target_shape=(3, 4))

    forward = resize(TensorBundle({"image": image}))
    restored = resize.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(forward["image"])) == (3, 4, 1)
    assert tuple(ops.shape(restored["image"])) == (6, 8, 1)


@pytest.mark.unit
def test_resize_inverse_without_trace_is_noop():
    bundle = TensorBundle({"image": as_tensor(np.ones((4, 4, 1), dtype=np.float32))})
    resize = Resize(keys=["image"], interpolation="bilinear", target_shape=(2, 2))

    restored = resize.inverse(bundle)

    assert restored is bundle


@pytest.mark.unit
def test_spacing_rejects_2d_inputs_with_clear_error():
    image = as_tensor(np.random.randn(8, 8, 1).astype(np.float32))
    spacing = Spacing(keys=["image"], pixdim=(1.0, 1.0, 1.0))

    with pytest.raises(
        ValueError, match="supports only 3D channel-last tensors shaped \\(D, H, W, C\\)"
    ):
        spacing(TensorBundle({"image": image}))


@pytest.mark.unit
def test_spacing_uses_default_spacing_when_affine_missing():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    spacing = Spacing(keys=["image"], pixdim=(2.0, 2.0, 2.0))

    with pytest.warns(UserWarning, match="Affine matrix is not provided"):
        out = spacing(TensorBundle({"image": image}))

    np.testing.assert_allclose(ops.convert_to_numpy(out.meta["pixdim"]), np.array([2.0, 2.0, 2.0]))


@pytest.mark.unit
def test_spacing_validates_pixdim_and_mode():
    with pytest.raises(ValueError, match="`pixdim` must be 3D"):
        Spacing(keys=["image"], pixdim=(1.0, 1.0))

    with pytest.raises(ValueError, match="Invalid mode 'bilinear' for 3D input"):
        Spacing(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear")


@pytest.mark.unit
def test_spacing_rejects_mapping_without_all_requested_keys():
    with pytest.raises(ValueError, match="Missing resampling mode for keys"):
        Spacing(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode={"image": "trilinear"},
        )


@pytest.mark.unit
def test_orientation_rejects_2d_inputs_with_clear_error():
    image = as_tensor(np.random.randn(8, 8, 1).astype(np.float32))
    affine = as_tensor(np.eye(4, dtype=np.float32))
    orientation = Orientation(keys=["image"], axcodes="RAS")

    with pytest.raises(
        ValueError,
        match="supports only 3D channel-last tensors shaped \\(D, H, W, C\\)",
    ):
        orientation(TensorBundle({"image": image}, {"affine": affine}))


@pytest.mark.unit
def test_orientation_requires_affine_and_valid_axcodes():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))

    with pytest.raises(ValueError, match="axcodes must be a 3-character string"):
        Orientation(keys=["image"], axcodes="RA")

    with pytest.raises(ValueError, match="Affine matrix is required"):
        Orientation(keys=["image"], axcodes="RAS")(TensorBundle({"image": image}))


@pytest.mark.unit
def test_orientation_allow_missing_keys_returns_bundle_unchanged():
    affine = as_tensor(np.eye(4, dtype=np.float32))
    orientation = Orientation(keys=["image"], axcodes="RAS", allow_missing_keys=True)
    bundle = TensorBundle(
        {"other": as_tensor(np.ones((4, 5, 6, 1), dtype=np.float32))}, {"affine": affine}
    )

    out = orientation(bundle)

    assert out is bundle


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
    restored = spacing.inverse(
        TensorBundle({"image": forward["image"], "label": forward["label"]}, forward.meta)
    )
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
    assert trace["params"]["target_tensor_axcodes"] == "SAR"
    restored = orientation.inverse(
        TensorBundle({"image": forward["image"], "label": forward["label"]}, forward.meta)
    )
    assert tuple(ops.shape(restored["image"])) == (4, 5, 6, 1)
    assert tuple(ops.shape(restored["label"])) == (4, 5, 6, 1)


@pytest.mark.unit
def test_orientation_flip_only_restores_original_layout_and_affine():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    affine = as_tensor(np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float32))

    orientation = Orientation(keys=["image"], axcodes="RAS")
    forward = orientation(TensorBundle({"image": image}, {"affine": affine}))

    trace = forward.get_applied_transforms()[-1]
    assert trace["params"]["target_tensor_axcodes"] == "SAR"

    restored = orientation.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(restored["image"])) == (4, 5, 6, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["affine"]),
        np.diag([-1.0, 1.0, 1.0, 1.0]),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_orientation_permutation_changes_spatial_order_and_inverse_restores():
    image = as_tensor(np.random.randn(2, 3, 4, 1).astype(np.float32))
    affine = as_tensor(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )

    orientation = Orientation(keys=["image"], axcodes="RAS")
    forward = orientation(TensorBundle({"image": image}, {"affine": affine}))

    assert tuple(ops.shape(forward["image"])) == (2, 4, 3, 1)

    restored = orientation.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(restored["image"])) == (2, 3, 4, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["affine"]),
        ops.convert_to_numpy(affine),
        rtol=1e-6,
    )


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
def test_scale_intensity_range_clips_and_preserves_dtype():
    image = as_tensor(np.array([[[-1.0], [0.5], [2.0]]], dtype=np.float32))
    out = ScaleIntensityRange(
        keys=["image"],
        a_min=0.0,
        a_max=1.0,
        b_min=0.0,
        b_max=10.0,
        clip=True,
        dtype=np.float32,
    )(TensorBundle({"image": image}))

    np.testing.assert_allclose(
        ops.convert_to_numpy(out["image"]), np.array([[[0.0], [5.0], [10.0]]])
    )


@pytest.mark.unit
def test_scale_intensity_range_rejects_partial_target_range():
    with pytest.raises(ValueError, match="must be provided together"):
        ScaleIntensityRange(keys=["image"], a_min=0.0, a_max=1.0, b_min=0.0)


@pytest.mark.unit
def test_normalize_intensity_records_trace():
    image = as_tensor(np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32))
    out = NormalizeIntensity(keys=["image"])(TensorBundle({"image": image}))

    assert tuple(ops.shape(out["image"])) == (2, 2, 1)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "NormalizeIntensity"
    assert trace["random"] is False


@pytest.mark.unit
def test_normalize_intensity_nonzero_preserves_zero_background():
    image = as_tensor(np.array([[[0.0], [1.0]], [[3.0], [0.0]]], dtype=np.float32))
    out = NormalizeIntensity(keys=["image"], nonzero=True)(TensorBundle({"image": image}))

    normalized = ops.convert_to_numpy(out["image"])
    assert normalized[0, 0, 0] == 0.0
    assert normalized[1, 1, 0] == 0.0


@pytest.mark.unit
def test_normalize_intensity_channel_wise_with_fixed_stats():
    image = as_tensor(np.array([[[1.0, 5.0], [3.0, 9.0]]], dtype=np.float32))
    out = NormalizeIntensity(
        keys=["image"],
        subtrahend=1.0,
        divisor=2.0,
        channel_wise=True,
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
    )(TensorBundle({"image": image}))

    normalized = ops.convert_to_numpy(out["image"])
    assert normalized[0, 0, 0] == 0.0
    assert normalized[1, 1, 0] == 0.0


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
def test_signal_fill_empty_outputs_float32_tensor():
    image = as_tensor(np.array([[[np.nan], [1.0]]], dtype=np.float64))
    out = SignalFillEmpty(keys=["image"], replacement=2.0)(TensorBundle({"image": image}))

    assert out["image"].dtype == tf.float32
    np.testing.assert_allclose(
        ops.convert_to_numpy(out["image"]), np.array([[[2.0], [1.0]]], dtype=np.float32)
    )


@pytest.mark.unit
def test_shift_intensity_records_trace():
    image = as_tensor(np.ones((4, 4, 1), dtype=np.float32))
    out = ShiftIntensity(keys=["image"], offsets=2.0)(TensorBundle({"image": image}))

    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "ShiftIntensity"
    assert trace["params"]["keys"] == ["image"]


@pytest.mark.unit
def test_random_shift_intensity_preserves_shape_and_range():
    image = as_tensor(np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32))
    out = RandomShiftIntensity(keys=["image"], offsets=(-0.2, 0.8), prob=1.0)(
        TensorBundle({"image": image})
    )
    shifted = ops.convert_to_numpy(out["image"])
    original = ops.convert_to_numpy(image)

    assert shifted.shape == (1, 2, 2, 1)
    assert np.all(shifted >= original - 0.8)
    assert np.all(shifted <= original + 0.8)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandomShiftIntensity"
    assert bool(ops.convert_to_numpy(trace["applied"]))
    assert trace["random"] is True
    assert trace["invertible"] is False
    assert trace["kernel"] == "ShiftIntensity"


@pytest.mark.unit
def test_random_shift_intensity_channel_wise_records_per_channel_offsets():
    image = as_tensor(np.ones((4, 4, 2), dtype=np.float32))
    out = RandomShiftIntensity(keys=["image"], offsets=0.5, prob=1.0, channel_wise=True)(
        TensorBundle({"image": image})
    )

    trace = out.get_applied_transforms()[-1]
    offsets = trace["params"]["offsets"]["image"]
    assert tuple(ops.shape(offsets)) == (1, 1, 2)


@pytest.mark.unit
def test_random_shift_intensity_prob_zero_is_noop():
    image = as_tensor(np.ones((4, 4, 1), dtype=np.float32))
    out = RandomShiftIntensity(keys=["image"], offsets=0.5, prob=0.0)(
        TensorBundle({"image": image})
    )

    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), ops.convert_to_numpy(image))
    assert not bool(ops.convert_to_numpy(out.get_applied_transforms()[-1]["applied"]))


@pytest.mark.unit
def test_random_shift_intensity_supports_2d_and_3d_channel_last_tensors():
    image_2d = as_tensor(np.ones((6, 5, 1), dtype=np.float32))
    image_3d = as_tensor(np.ones((4, 6, 5, 1), dtype=np.float32))

    out_2d = RandomShiftIntensity(keys=["image"], offsets=0.5, prob=1.0)(
        TensorBundle({"image": image_2d})
    )
    out_3d = RandomShiftIntensity(keys=["image"], offsets=0.25, prob=1.0)(
        TensorBundle({"image": image_3d})
    )

    shifted_2d = ops.convert_to_numpy(out_2d["image"])
    shifted_3d = ops.convert_to_numpy(out_3d["image"])

    assert shifted_2d.shape == (6, 5, 1)
    assert shifted_3d.shape == (4, 6, 5, 1)
    assert np.all(shifted_2d >= 0.5)
    assert np.all(shifted_2d <= 1.5)
    assert np.all(shifted_3d >= 0.75)
    assert np.all(shifted_3d <= 1.25)


@pytest.mark.unit
def test_random_shift_intensity_channel_wise_samples_per_channel_values():
    image = as_tensor(np.ones((3, 4, 2), dtype=np.float32))

    out = RandomShiftIntensity(keys=["image"], offsets=0.5, prob=1.0, channel_wise=True)(
        TensorBundle({"image": image})
    )

    shifted = ops.convert_to_numpy(out["image"])
    trace = out.get_applied_transforms()[-1]
    offsets = ops.convert_to_numpy(trace["params"]["offsets"]["image"])

    assert shifted.shape == (3, 4, 2)
    assert offsets.shape == (1, 1, 2)
    assert np.all(shifted >= 0.5)
    assert np.all(shifted <= 1.5)


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
def test_spatial_crop_validates_exclusive_start_and_center():
    with pytest.raises(ValueError, match="Only one of `roi_start` or `roi_center` may be provided"):
        SpatialCrop(keys=["image"], roi_size=(2, 2), roi_start=(0, 0), roi_center=(1, 1))


@pytest.mark.unit
def test_spatial_crop_nonpositive_roi_uses_full_extent():
    image = as_tensor(np.arange(30, dtype=np.float32).reshape(5, 6, 1))
    out = SpatialCrop(keys=["image"], roi_size=(0, -1))(TensorBundle({"image": image}))

    assert tuple(ops.shape(out["image"])) == (5, 6, 1)


@pytest.mark.unit
def test_random_spatial_crop_supports_2d_and_3d_channel_last_tensors():
    image_2d = as_tensor(np.arange(30, dtype=np.float32).reshape(5, 6, 1))
    image_3d = as_tensor(np.arange(120, dtype=np.float32).reshape(4, 5, 6, 1))

    out_2d = RandomSpatialCrop(keys=["image"], roi_size=(3, 4), random_center=False)(
        TensorBundle({"image": image_2d})
    )
    out_3d = RandomSpatialCrop(keys=["image"], roi_size=(2, 3, 4), random_center=False)(
        TensorBundle({"image": image_3d})
    )

    assert tuple(ops.shape(out_2d["image"])) == (3, 4, 1)
    assert tuple(ops.shape(out_3d["image"])) == (2, 3, 4, 1)
    assert out_3d.get_applied_transforms()[-1]["kernel"] == "SpatialCrop"


@pytest.mark.unit
def test_random_spatial_crop_validates_configuration():
    with pytest.raises(ValueError, match="must contain at least one key"):
        RandomSpatialCrop(keys=[], roi_size=(2, 2))

    with pytest.raises(ValueError, match="min_valid_ratio must be in range"):
        RandomSpatialCrop(keys=["image"], roi_size=(2, 2), min_valid_ratio=1.5)

    with pytest.raises(ValueError, match="max_attempts must be a positive integer"):
        RandomSpatialCrop(keys=["image"], roi_size=(2, 2), max_attempts=0)

    with pytest.raises(ValueError, match="must provide an invalid_label"):
        RandomSpatialCrop(keys=["image"], roi_size=(2, 2), min_valid_ratio=0.2)


@pytest.mark.unit
def test_random_spatial_crop_random_size_and_label_aware_modes():
    image = as_tensor(np.arange(120, dtype=np.float32).reshape(4, 5, 6, 1))
    label = as_tensor(np.zeros((4, 5, 6, 1), dtype=np.int32))
    label = ops.convert_to_tensor(
        np.pad(np.ones((2, 2, 2, 1), dtype=np.int32), ((1, 1), (1, 2), (2, 2), (0, 0)))
    )

    out = RandomSpatialCrop(
        keys=["image", "label"],
        roi_size=(1, 2, 2),
        max_roi_size=(2, 4, 4),
        random_size=True,
        invalid_label=0,
        min_valid_ratio=0.0,
    )(TensorBundle({"image": image, "label": label}))

    trace = out.get_applied_transforms()[-1]
    roi_size = ops.convert_to_numpy(trace["params"]["roi_size"])
    assert np.all(roi_size >= np.array([1, 2, 2]))
    assert np.all(roi_size <= np.array([2, 4, 4]))


@pytest.mark.unit
def test_random_spatial_crop_requires_label_for_label_aware_mode():
    transform = RandomSpatialCrop(keys=["image"], roi_size=(2, 2), invalid_label=0)

    with pytest.raises(KeyError, match="`label` key is required"):
        transform(TensorBundle({"image": as_tensor(np.ones((4, 4, 1), dtype=np.float32))}))


@pytest.mark.unit
def test_random_spatial_crop_uses_second_key_for_label_aware_mode():
    image = as_tensor(np.arange(16, dtype=np.float32).reshape(4, 4, 1))
    mask = as_tensor(np.pad(np.ones((2, 2, 1), dtype=np.int32), ((1, 1), (1, 1), (0, 0))))

    out = RandomSpatialCrop(
        keys=["image", "mask"],
        roi_size=(2, 2),
        invalid_label=0,
        random_center=False,
    )(TensorBundle({"image": image, "mask": mask}))

    assert tuple(ops.shape(out["image"])) == (2, 2, 1)
    assert tuple(ops.shape(out["mask"])) == (2, 2, 1)


@pytest.mark.unit
def test_random_spatial_crop_rejects_unsupported_spatial_rank():
    image_4d_spatial = as_tensor(np.ones((2, 3, 4, 5, 1), dtype=np.float32))
    transform = RandomSpatialCrop(keys=["image"], roi_size=(2, 2, 2, 2))

    with pytest.raises(ValueError, match="currently supports only 2D or 3D inputs"):
        transform(TensorBundle({"image": image_4d_spatial}))


@pytest.mark.unit
def test_random_spatial_crop_label_aware_mode_keeps_thin_spatial_dimensions():
    image = as_tensor(np.arange(12, dtype=np.float32).reshape(1, 4, 3, 1))
    label = as_tensor(np.ones((1, 4, 3, 1), dtype=np.int32))

    out = RandomSpatialCrop(
        keys=["image", "label"],
        roi_size=(1, 2, 2),
        invalid_label=0,
        random_center=False,
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (1, 2, 2, 1)


@pytest.mark.unit
def test_random_spatial_crop_label_aware_mode_supports_multi_channel_labels():
    image = as_tensor(np.arange(16, dtype=np.float32).reshape(4, 4, 1))
    label = as_tensor(
        np.stack(
            [
                np.pad(np.ones((2, 2), dtype=np.int32), ((1, 1), (1, 1))),
                np.zeros((4, 4), dtype=np.int32),
            ],
            axis=-1,
        )
    )

    out = RandomSpatialCrop(
        keys=["image", "label"],
        roi_size=(2, 2),
        invalid_label=0,
        random_center=False,
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (2, 2, 1)
    assert tuple(ops.shape(out["label"])) == (2, 2, 2)


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
    image_3d = as_tensor(
        np.pad(np.ones((2, 2, 2, 1), dtype=np.float32), ((1, 1), (1, 1), (1, 1), (0, 0)))
    )

    out_2d = CropForeground(keys=["image"], source_key="image")(TensorBundle({"image": image_2d}))
    out_3d = CropForeground(keys=["image"], source_key="image")(TensorBundle({"image": image_3d}))

    assert tuple(ops.shape(out_2d["image"])) == (2, 2, 1)
    assert tuple(ops.shape(out_3d["image"])) == (2, 2, 2, 1)
    assert out_2d.get_applied_transforms()[-1]["name"] == "CropForeground"


@pytest.mark.unit
def test_crop_foreground_empty_mask_returns_full_image_and_can_disable_metadata():
    image = as_tensor(np.zeros((4, 5, 1), dtype=np.float32))
    out = CropForeground(
        keys=["image"],
        source_key="image",
        select_fn=lambda x: x > 10,
        start_coord_key=None,
        end_coord_key=None,
    )(TensorBundle({"image": image}))

    assert tuple(ops.shape(out["image"])) == (4, 5, 1)
    assert "foreground_start_coord" not in out.meta
    assert "foreground_end_coord" not in out.meta


@pytest.mark.unit
def test_crop_foreground_channel_indices_and_k_divisible():
    image = as_tensor(np.zeros((6, 6, 2), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[1:4, 2:5, 1] = 1.0
    out = CropForeground(
        keys=["image"],
        source_key="image",
        channel_indices=[1],
        k_divisible=2,
        margin=0,
    )(TensorBundle({"image": as_tensor(image_np)}))

    shape = tuple(ops.shape(out["image"]))
    assert shape[0] % 2 == 0
    assert shape[1] % 2 == 0


@pytest.mark.unit
def test_crop_foreground_runs_under_tf_function_graph_mode():
    transform = CropForeground(keys=["image"], source_key="image")
    image = as_tensor(
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

    @tf.function
    def apply_transform(x):
        return transform({"image": x})["image"]

    output = apply_transform(image)

    assert tuple(ops.shape(output)) == (2, 2, 1)


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_uses_spatial_crop_kernel():
    image = as_tensor(np.random.randn(6, 6, 6, 1).astype(np.float32))
    label = as_tensor(np.zeros((6, 6, 6, 1), dtype=np.float32))
    label = ops.convert_to_tensor(
        np.pad(np.ones((2, 2, 2, 1), dtype=np.float32), ((2, 2), (2, 2), (2, 2), (0, 0)))
    )

    out = RandomCropByPosNegLabel(
        keys=["image", "label"],
        spatial_size=(3, 3, 3),
        pos=1,
        neg=1,
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (3, 3, 3, 1)
    assert tuple(ops.shape(out["label"])) == (3, 3, 3, 1)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandomCropByPosNegLabel"
    assert trace["kernel"] == "SpatialCrop"
    assert trace["random"] is True


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_supports_2d_and_3d():
    image_2d = as_tensor(np.random.randn(8, 8, 1).astype(np.float32))
    label_2d = as_tensor(np.pad(np.ones((2, 2, 1), dtype=np.float32), ((3, 3), (3, 3), (0, 0))))
    out_2d = RandomCropByPosNegLabel(
        keys=["image", "label"],
        spatial_size=(4, 4),
        pos=1,
        neg=1,
    )(TensorBundle({"image": image_2d, "label": label_2d}))

    image_3d = as_tensor(np.random.randn(6, 6, 6, 1).astype(np.float32))
    label_3d = ops.convert_to_tensor(
        np.pad(np.ones((2, 2, 2, 1), dtype=np.float32), ((2, 2), (2, 2), (2, 2), (0, 0)))
    )
    out_3d = RandomCropByPosNegLabel(
        keys=["image", "label"],
        spatial_size=(3, 3, 3),
        pos=1,
        neg=1,
    )(TensorBundle({"image": image_3d, "label": label_3d}))

    assert tuple(ops.shape(out_2d["image"])) == (4, 4, 1)
    assert tuple(ops.shape(out_2d["label"])) == (4, 4, 1)
    assert tuple(ops.shape(out_3d["image"])) == (3, 3, 3, 1)
    assert tuple(ops.shape(out_3d["label"])) == (3, 3, 3, 1)


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_validates_arguments():
    with pytest.raises(ValueError, match="pos and neg must be non-negative"):
        RandomCropByPosNegLabel(keys=["image", "label"], spatial_size=(2, 2, 2), pos=-1, neg=1)

    with pytest.raises(ValueError, match="pos and neg cannot both be zero"):
        RandomCropByPosNegLabel(keys=["image", "label"], spatial_size=(2, 2, 2), pos=0, neg=0)

    with pytest.raises(ValueError, match="requires a pair of image and label as keys"):
        RandomCropByPosNegLabel(keys=["image"], spatial_size=(2, 2, 2), pos=1, neg=1)

    with pytest.raises(ValueError, match="currently supports only num_samples=1"):
        RandomCropByPosNegLabel(
            keys=["image", "label"], spatial_size=(2, 2, 2), pos=1, neg=1, num_samples=2
        )


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_rejects_2d_and_supports_allow_missing_keys():
    image_1d_like = as_tensor(np.ones((6, 1), dtype=np.float32))
    label_1d_like = as_tensor(np.ones((6, 1), dtype=np.float32))
    transform = RandomCropByPosNegLabel(keys=["image", "label"], spatial_size=(2, 2), pos=1, neg=1)

    with pytest.raises(ValueError, match="currently supports only 2D or 3D inputs"):
        transform(TensorBundle({"image": image_1d_like, "label": label_1d_like}))

    image_2d = as_tensor(np.ones((6, 6, 1), dtype=np.float32))
    label_2d = as_tensor(np.ones((6, 6, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="`spatial_size` must contain exactly 2 values"):
        RandomCropByPosNegLabel(keys=["image", "label"], spatial_size=(2, 2, 2), pos=1, neg=1)(
            TensorBundle({"image": image_2d, "label": label_2d})
        )

    skip_transform = RandomCropByPosNegLabel(
        keys=["image", "label"],
        spatial_size=(2, 2, 2),
        pos=1,
        neg=1,
        allow_missing_keys=True,
    )
    bundle = TensorBundle({"image": as_tensor(np.ones((4, 4, 4, 1), dtype=np.float32))})
    assert skip_transform(bundle) is bundle


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_validates_image_reference_key():
    image = as_tensor(np.ones((4, 4, 4, 1), dtype=np.float32))
    label = as_tensor(np.ones((4, 4, 4, 1), dtype=np.float32))

    transform = RandomCropByPosNegLabel(
        keys=["image", "label"],
        spatial_size=(2, 2, 2),
        pos=1,
        neg=1,
        image_reference_key="reference",
    )

    with pytest.raises(KeyError, match="reference"):
        transform(TensorBundle({"image": image, "label": label}))


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
def test_flip_none_axis_is_noop_and_invalid_axis_raises():
    image = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    no_op = Flip(keys=["image"], spatial_axis=None)(TensorBundle({"image": image}))
    np.testing.assert_allclose(ops.convert_to_numpy(no_op["image"]), ops.convert_to_numpy(image))

    with pytest.raises(ValueError):
        Flip(keys=["image"], spatial_axis=5)(TensorBundle({"image": image}))


@pytest.mark.unit
def test_flip_negative_axis_resolves_against_spatial_rank_only():
    image = as_tensor(np.arange(12, dtype=np.float32).reshape(3, 4, 1))
    out = Flip(keys=["image"], spatial_axis=-1)(TensorBundle({"image": image}))

    np.testing.assert_allclose(
        ops.convert_to_numpy(out["image"]),
        ops.convert_to_numpy(image)[:, ::-1, :],
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
def test_rotate90_k_zero_is_noop_and_invalid_axes_raise():
    image = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    out = Rotate90(keys=["image"], k=4)(TensorBundle({"image": image}))
    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), ops.convert_to_numpy(image))
    assert out.get_applied_transforms() == []

    with pytest.raises(ValueError, match="must contain exactly two axes"):
        Rotate90(keys=["image"], k=1, spatial_axes=(0,))(TensorBundle({"image": image}))


@pytest.mark.unit
def test_rotate90_negative_axes_resolve_against_spatial_rank_only():
    image = as_tensor(np.arange(12, dtype=np.float32).reshape(3, 4, 1))
    out = Rotate90(keys=["image"], k=1, spatial_axes=(0, -1))(TensorBundle({"image": image}))

    expected = np.rot90(ops.convert_to_numpy(image), k=1, axes=(0, 1))
    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), expected)


@pytest.mark.unit
def test_random_rotate90_preserves_shape():
    image = as_tensor(np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2))
    out = RandomRotate90(keys=["image"], prob=1.0, max_k=3)(TensorBundle({"image": image}))
    assert tuple(ops.shape(out["image"])) == (1, 2, 2, 2)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandomRotate90"
    assert bool(ops.convert_to_numpy(trace["applied"]))
    assert trace["random"] is True
    assert trace["invertible"] is False
    assert trace["kernel"] == "Rotate90"


@pytest.mark.unit
def test_random_rotate90_prob_zero_records_no_application():
    image = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    out = RandomRotate90(keys=["image"], prob=0.0, max_k=3)(TensorBundle({"image": image}))

    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), ops.convert_to_numpy(image))
    assert not bool(ops.convert_to_numpy(out.get_applied_transforms()[-1]["applied"]))


@pytest.mark.unit
def test_random_rotate90_validates_max_k():
    with pytest.raises(ValueError, match="must be >= 1"):
        RandomRotate90(keys=["image"], max_k=0)


@pytest.mark.unit
def test_random_rotate_preserves_shape_and_records_trace():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))

    out = RandomRotate(keys=["image", "label"], factor=0.2, prob=1.0)(
        TensorBundle({"image": image, "label": label})
    )

    assert tuple(ops.shape(out["image"])) == (4, 5, 6, 1)
    assert tuple(ops.shape(out["label"])) == (4, 5, 6, 1)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandomRotate"
    assert bool(ops.convert_to_numpy(trace["applied"]))
    assert trace["random"] is True
    assert trace["invertible"] is False
    assert trace["kernel"] == "rotate_volume"


@pytest.mark.unit
def test_random_rotate_supports_integer_label_tensors():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 3, (4, 5, 6, 1)).astype(np.int32))

    out = RandomRotate(keys=["image", "label"], factor=0.2, prob=1.0)(
        TensorBundle({"image": image, "label": label})
    )

    assert out["label"].dtype == label.dtype
    assert tuple(ops.shape(out["label"])) == (4, 5, 6, 1)


@pytest.mark.unit
def test_random_rotate_validates_arguments_and_fill_crop_mode():
    with pytest.raises(TypeError, match="`keys` must be a list or tuple"):
        RandomRotate(keys="image", factor=0.1)

    with pytest.raises(ValueError, match="`keys` must have length 1 or 2"):
        RandomRotate(keys=["image", "label", "mask"], factor=0.1)

    with pytest.raises(ValueError, match="fill_mode must be either 'crop' or 'constant'"):
        RandomRotate(keys=["image"], fill_mode="reflect")

    with pytest.raises(ValueError, match="must be non-negative"):
        RandomRotate(keys=["image"], factor=-0.1)

    image = as_tensor(np.random.randn(4, 8, 8, 1).astype(np.float32))
    out = RandomRotate(keys=["image"], factor=0.2, prob=1.0, fill_mode="crop")(
        TensorBundle({"image": image})
    )
    assert tuple(ops.shape(out["image"])) == (4, 8, 8, 1)


@pytest.mark.unit
def test_random_rotate_allow_missing_keys_and_prob_zero():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    transform = RandomRotate(keys=["image", "label"], prob=0.0, allow_missing_keys=True)
    bundle = TensorBundle({"image": image})

    out = transform(bundle)

    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), ops.convert_to_numpy(image))
    assert not bool(ops.convert_to_numpy(out.get_applied_transforms()[-1]["applied"]))


@pytest.mark.unit
def test_rand_cutout_preserves_shape_and_records_trace():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))

    out = RandomCutOut(
        keys=["image", "label"],
        mask_size=(2, 2),
        num_cuts=2,
        prob=1.0,
        fill_mode="constant",
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (4, 5, 6, 1)
    assert tuple(ops.shape(out["label"])) == (4, 5, 6, 1)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandomCutOut"
    assert bool(ops.convert_to_numpy(trace["applied"]))
    assert trace["random"] is True
    assert trace["invertible"] is False
    assert trace["kernel"] == "cutout_mask"


@pytest.mark.unit
def test_random_cutout_supports_2d_and_3d():
    image_2d = as_tensor(np.random.randn(8, 8, 1).astype(np.float32))
    label_2d = as_tensor(np.ones((8, 8, 1), dtype=np.float32))
    out_2d = RandomCutOut(
        keys=["image", "label"],
        mask_size=(2, 2),
        num_cuts=1,
        prob=1.0,
    )(TensorBundle({"image": image_2d, "label": label_2d}))

    image_3d = as_tensor(np.random.randn(4, 8, 8, 1).astype(np.float32))
    label_3d = as_tensor(np.ones((4, 8, 8, 1), dtype=np.float32))
    out_3d = RandomCutOut(
        keys=["image", "label"],
        mask_size=(2, 2),
        num_cuts=1,
        prob=1.0,
    )(TensorBundle({"image": image_3d, "label": label_3d}))

    assert tuple(ops.shape(out_2d["image"])) == (8, 8, 1)
    assert tuple(ops.shape(out_3d["image"])) == (4, 8, 8, 1)


@pytest.mark.unit
def test_random_cutout_validates_arguments():
    with pytest.raises(ValueError, match="`keys` must have length 2"):
        RandomCutOut(keys=["image"], mask_size=(2, 2), num_cuts=1)

    with pytest.raises(ValueError, match="`mask_size` must be a sequence of two integers"):
        RandomCutOut(keys=["image", "label"], mask_size=(2,), num_cuts=1)

    with pytest.raises(ValueError, match="All values in `mask_size` must be positive integers"):
        RandomCutOut(keys=["image", "label"], mask_size=(2, 0), num_cuts=1)

    with pytest.raises(ValueError, match="`num_cuts` must be a positive integer"):
        RandomCutOut(keys=["image", "label"], mask_size=(2, 2), num_cuts=0)

    with pytest.raises(ValueError, match='`fill_mode` must be either "gaussian" or "constant"'):
        RandomCutOut(keys=["image", "label"], mask_size=(2, 2), num_cuts=1, fill_mode="reflect")

    with pytest.raises(ValueError, match="`cutout_mode` must be one of"):
        RandomCutOut(keys=["image", "label"], mask_size=(2, 2), num_cuts=1, cutout_mode="plane")


@pytest.mark.unit
def test_random_cutout_supports_slice_mode_gaussian_mode_and_allow_missing_keys():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))

    out = RandomCutOut(
        keys=["image", "label"],
        mask_size=(2, 2),
        num_cuts=1,
        prob=1.0,
        fill_mode="gaussian",
        cutout_mode="slice",
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (4, 5, 6, 1)

    skip = RandomCutOut(
        keys=["image", "label"],
        mask_size=(2, 2),
        num_cuts=1,
        allow_missing_keys=True,
    )
    bundle = TensorBundle({"image": image})
    assert skip(bundle) is bundle


@pytest.mark.unit
def test_random_cutout_2d_supports_slice_mode_and_invalid_label():
    image = as_tensor(np.random.randn(8, 8, 1).astype(np.float32))
    label = np.zeros((8, 8, 1), dtype=np.float32)
    label[2:6, 2:6, 0] = 1.0

    out = RandomCutOut(
        keys=["image", "label"],
        mask_size=(2, 2),
        num_cuts=1,
        prob=1.0,
        cutout_mode="slice",
        invalid_label=0.0,
    )(TensorBundle({"image": image, "label": as_tensor(label)}))

    assert tuple(ops.shape(out["image"])) == (8, 8, 1)


@pytest.mark.unit
def test_random_cutout_mask_size_one_affects_at_least_one_pixel():
    image = as_tensor(np.ones((8, 8, 1), dtype=np.float32))
    label = as_tensor(np.ones((8, 8, 1), dtype=np.float32))

    out = RandomCutOut(
        keys=["image", "label"],
        mask_size=(1, 1),
        num_cuts=1,
        prob=1.0,
        fill_mode="constant",
        fill_value=0.0,
    )(TensorBundle({"image": image, "label": label}))

    assert np.any(ops.convert_to_numpy(out["image"]) == 0.0)


@pytest.mark.unit
def test_random_cutout_prob_zero_and_unsupported_rank_rejection():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))
    out = RandomCutOut(keys=["image", "label"], mask_size=(2, 2), num_cuts=1, prob=0.0)(
        TensorBundle({"image": image, "label": label})
    )
    assert not bool(ops.convert_to_numpy(out.get_applied_transforms()[-1]["applied"]))

    image_1d_like = as_tensor(np.ones((6, 1), dtype=np.float32))
    label_1d_like = as_tensor(np.ones((6, 1), dtype=np.float32))
    transform = RandomCutOut(keys=["image", "label"], mask_size=(2, 2), num_cuts=1)
    with pytest.raises(ValueError, match="currently supports only 2D or 3D inputs"):
        transform(TensorBundle({"image": image_1d_like, "label": label_1d_like}))


@pytest.mark.unit
def test_random_flip_records_random_trace():
    image = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    out = RandomFlip(keys=["image"], prob=1.0, spatial_axis=1)(TensorBundle({"image": image}))

    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandomFlip"
    assert bool(ops.convert_to_numpy(trace["applied"]))
    assert trace["random"] is True
    assert trace["invertible"] is False
    assert trace["kernel"] == "Flip"


@pytest.mark.unit
def test_random_flip_prob_zero_and_allow_missing_keys():
    image = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    out = RandomFlip(keys=["image"], prob=0.0, spatial_axis=1)(TensorBundle({"image": image}))
    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), ops.convert_to_numpy(image))
    assert not bool(ops.convert_to_numpy(out.get_applied_transforms()[-1]["applied"]))

    bundle = TensorBundle({"other": as_tensor(np.ones((2, 3, 1), dtype=np.float32))})
    transform = RandomFlip(keys=["image"], prob=1.0, spatial_axis=1, allow_missing_keys=True)
    assert transform(bundle) is bundle


@pytest.mark.unit
def test_random_flip_none_axis_records_noop_trace():
    image = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    out = RandomFlip(keys=["image"], prob=1.0, spatial_axis=None)(TensorBundle({"image": image}))

    trace = out.get_applied_transforms()[-1]
    assert trace["params"]["spatial_axis"] is None
    assert trace["applied"] is False


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


@pytest.mark.unit
def test_compose_inverse_skips_noninvertible_and_restores_invertible_transforms():
    image = as_tensor(np.arange(16, dtype=np.float32).reshape(4, 4, 1))
    transform = Compose(
        [
            ShiftIntensity(keys=["image"], offsets=2.0),
            Flip(keys=["image"], spatial_axis=1),
            Resize(keys=["image"], interpolation="bilinear", target_shape=(2, 2)),
        ]
    )

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(restored["image"])) == (4, 4, 1)

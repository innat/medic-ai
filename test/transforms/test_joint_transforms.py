import numpy as np
import pytest
from keras import ops

from medicai.transforms import (
    Compose,
    CropForeground,
    Flip,
    Orientation,
    RandomFlip,
    RandomRotate90,
    Resize,
    Rotate90,
    ScaleIntensityRange,
    ShiftIntensity,
    Spacing,
    TensorBundle,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_sample_only_spatial_transforms_reject_batch_layouts():
    with pytest.raises(ValueError, match="supports only input_layout values"):
        CropForeground(keys=["image"], input_layout="BHWC")

    with pytest.raises(ValueError, match="supports only input_layout values"):
        Orientation(keys=["image"], input_layout="BDHWC")

    with pytest.raises(ValueError, match="supports only input_layout values"):
        Spacing(keys=["image"], pixdim=(1.0, 1.0, 1.0), input_layout="BDHWC")


@pytest.mark.unit
def test_compose_inverse_restores_pipeline_with_multiple_scale_intensity_range_instances():
    image = as_tensor(np.array([[[0.0], [0.5], [1.0]]], dtype=np.float32))
    pipeline = Compose(
        [
            ScaleIntensityRange(
                keys=["image"],
                source_value_range=(0.0, 1.0),
                target_value_range=(-1.0, 1.0),
                input_layout="HWC",
            ),
            ScaleIntensityRange(
                keys=["image"],
                source_value_range=(-1.0, 1.0),
                target_value_range=(0.0, 2.0),
                input_layout="HWC",
            ),
        ]
    )

    forward = pipeline(TensorBundle({"image": image}))
    restored = pipeline.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
        rtol=1e-6,
    )
    assert restored.get_applied_transforms() == []


@pytest.mark.unit
def test_compose_inverse_restores_pipeline_with_multiple_shift_intensity_instances():
    image = as_tensor(np.arange(12, dtype=np.float32).reshape(3, 4, 1))
    pipeline = Compose(
        [
            ShiftIntensity(keys=["image"], offset=1.0, input_layout="HWC"),
            ShiftIntensity(keys=["image"], offset=-2.0, input_layout="HWC"),
        ]
    )

    forward = pipeline(TensorBundle({"image": image}))
    restored = pipeline.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )
    assert restored.get_applied_transforms() == []


@pytest.mark.unit
def test_compose_inverse_restores_pipeline_with_multiple_flip_instances():
    image = as_tensor(np.arange(12, dtype=np.float32).reshape(3, 4, 1))
    pipeline = Compose(
        [
            Flip(keys=["image"], spatial_axis=0, input_layout="HWC"),
            Flip(keys=["image"], spatial_axis=1, input_layout="HWC"),
        ]
    )

    forward = pipeline(TensorBundle({"image": image}))
    restored = pipeline.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )
    assert restored.get_applied_transforms() == []


@pytest.mark.unit
def test_compose_inverse_restores_pipeline_with_multiple_rotate90_instances():
    image = as_tensor(np.arange(12, dtype=np.float32).reshape(3, 4, 1))
    pipeline = Compose(
        [
            Rotate90(keys=["image"], k=1, input_layout="HWC"),
            Rotate90(keys=["image"], k=3, input_layout="HWC"),
        ]
    )

    forward = pipeline(TensorBundle({"image": image}))
    restored = pipeline.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )
    assert restored.get_applied_transforms() == []


@pytest.mark.unit
def test_random_flip_and_random_rotate90_accept_input_layout():
    image = as_tensor(np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3, 1))

    flip_out = RandomFlip(
        keys=["image"],
        prob=1.0,
        spatial_axis=2,
        input_layout="BHWC",
        seed=3,
    )(TensorBundle({"image": image}))
    rotate_out = RandomRotate90(
        keys=["image"],
        prob=1.0,
        max_k=3,
        spatial_axis=(1, 2),
        input_layout="BHWC",
        seed=5,
    )(TensorBundle({"image": image}))

    assert tuple(ops.shape(flip_out["image"])) == (2, 3, 3, 1)
    assert tuple(ops.shape(rotate_out["image"])) == (2, 3, 3, 1)
    assert flip_out.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"
    assert rotate_out.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


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
            ShiftIntensity(keys=["image"], offset=2.0, input_layout="HWC"),
            Flip(keys=["image"], spatial_axis=1, input_layout="HWC"),
            Resize(
                keys=["image"], interpolation="bilinear", target_shape=(2, 2), input_layout="HWC"
            ),
        ]
    )

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(restored["image"])) == (4, 4, 1)


@pytest.mark.unit
def test_compose_inverse_restores_prediction_bundle_for_crop_orientation_spacing_pipeline():
    image = np.zeros((6, 8, 10, 1), dtype=np.float32)
    image[1:5, 2:7, 3:9, 0] = 2.0
    label = np.zeros((6, 8, 10, 1), dtype=np.float32)
    label[2:4, 3:6, 4:8, 0] = 1.0
    affine = as_tensor(
        np.array(
            [
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 1.5, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )

    pipeline = Compose(
        [
            CropForeground(keys=["image", "label"], source_key="image", input_layout="DHWC"),
            Orientation(keys=["image", "label"], axcodes="RAS"),
            Spacing(
                keys=["image", "label"],
                pixdim=(1.0, 0.75, 1.5),
                interpolation=("trilinear", "nearest"),
            ),
        ]
    )

    forward = pipeline(
        TensorBundle({"image": as_tensor(image), "label": as_tensor(label)}, {"affine": affine})
    )

    # Mimic model output by replacing the traced segmentation key with a fresh prediction.
    prediction = ops.cast(forward["label"] > 0.0, forward["label"].dtype)
    prediction_bundle = TensorBundle(
        {"image": forward["image"], "label": prediction},
        dict(forward.meta),
    )
    prediction_bundle.meta["applied_transforms"] = list(forward.get_applied_transforms())

    restored = pipeline.inverse(prediction_bundle)

    assert tuple(ops.shape(restored["image"])) == image.shape
    assert tuple(ops.shape(restored["label"])) == label.shape
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["affine"]),
        ops.convert_to_numpy(affine),
        rtol=1e-6,
    )
    restored_label = ops.convert_to_numpy(restored["label"])
    assert restored_label.dtype == label.dtype
    assert set(np.unique(restored_label)).issubset({0.0, 1.0})
    assert restored.get_applied_transforms() == []


@pytest.mark.unit
def test_rotate90_inverse_does_not_consume_another_instance_trace():
    image = as_tensor(np.arange(9, dtype=np.float32).reshape(3, 3, 1))
    first = Rotate90(keys=["image"], k=1, input_layout="HWC")
    second = Rotate90(keys=["image"], k=2, input_layout="HWC")
    bundle = first(TensorBundle({"image": image}))

    untouched = second.inverse(bundle)

    assert untouched is bundle
    assert len(untouched.get_applied_transforms()) == 1
    restored = first.inverse(bundle)
    np.testing.assert_array_equal(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )



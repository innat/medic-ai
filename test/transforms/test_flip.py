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
def test_flip_supports_2d_and_3d_and_records_inverse_trace():
    image_2d = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    image_3d = as_tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1))

    flip_2d = Flip(keys=["image"], spatial_axis=1, input_layout="HWC")
    flip_3d = Flip(keys=["image"], spatial_axis=(0, 2), input_layout="DHWC")

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
        ops.convert_to_numpy(
            flip_2d.inverse(TensorBundle({"image": out_2d["image"]}, out_2d.meta))["image"]
        ),
        ops.convert_to_numpy(image_2d),
    )


@pytest.mark.unit
def test_flip_inverse_without_trace_is_noop():
    bundle = TensorBundle({"image": as_tensor(np.ones((4, 4, 1), dtype=np.float32))})
    transform = Flip(keys=["image"], spatial_axis=1, input_layout="HWC")

    restored = transform.inverse(bundle)

    assert restored is bundle


@pytest.mark.unit
def test_flip_supports_batch_mode_for_2d_and_3d_channel_last_tensors():
    batch_2d = as_tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1))
    batch_3d = as_tensor(np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5, 1))

    flip_2d = Flip(keys=["image"], spatial_axis=2, input_layout="BHWC")
    flip_3d = Flip(keys=["image"], spatial_axis=(1, 3), input_layout="BDHWC")

    out_2d = flip_2d(TensorBundle({"image": batch_2d}))
    out_3d = flip_3d(TensorBundle({"image": batch_3d}))

    np.testing.assert_allclose(
        ops.convert_to_numpy(out_2d["image"]),
        ops.convert_to_numpy(batch_2d)[:, :, ::-1, :],
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(out_3d["image"]),
        ops.convert_to_numpy(batch_3d)[:, ::-1, :, ::-1, :],
    )
    assert out_2d.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_flip_accepts_input_layout_with_real_tensor_axes():
    batch_2d = as_tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1))
    batch_3d = as_tensor(np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5, 1))

    flip_2d = Flip(keys=["image"], spatial_axis=2, input_layout="BHWC")
    flip_3d = Flip(keys=["image"], spatial_axis=(1, 3), input_layout="BDHWC")

    out_2d = flip_2d(TensorBundle({"image": batch_2d}))
    out_3d = flip_3d(TensorBundle({"image": batch_3d}))

    np.testing.assert_allclose(
        ops.convert_to_numpy(out_2d["image"]),
        ops.convert_to_numpy(batch_2d)[:, :, ::-1, :],
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(out_3d["image"]),
        ops.convert_to_numpy(batch_3d)[:, ::-1, :, ::-1, :],
    )
    assert out_2d.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_flip_accepts_numpy_mapping_inputs():
    image = np.arange(6, dtype=np.float32).reshape(2, 3, 1)

    out = Flip(keys=["image"], spatial_axis=1, input_layout="HWC")({"image": image})

    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), image[:, ::-1, :])


@pytest.mark.unit
def test_flip_uses_same_batch_kernel_for_sample_and_batch_modes():
    sample = as_tensor(np.arange(12, dtype=np.float32).reshape(3, 4, 1))
    batch = as_tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1))

    sample_out = Flip(keys=["image"], spatial_axis=1, input_layout="HWC")(
        TensorBundle({"image": sample})
    )
    batch_out = Flip(keys=["image"], spatial_axis=2, input_layout="BHWC")(
        TensorBundle({"image": batch})
    )

    np.testing.assert_allclose(
        ops.convert_to_numpy(sample_out["image"]),
        ops.convert_to_numpy(sample)[:, ::-1, :],
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(batch_out["image"]),
        ops.convert_to_numpy(batch)[:, :, ::-1, :],
    )


@pytest.mark.unit
def test_flip_requires_spatial_axis_and_invalid_axis_raises():
    image = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))

    with pytest.raises(ValueError, match="requires `spatial_axis`"):
        Flip(keys=["image"], spatial_axis=None, input_layout="HWC")

    with pytest.raises(ValueError):
        Flip(keys=["image"], spatial_axis=5, input_layout="HWC")(TensorBundle({"image": image}))


@pytest.mark.unit
def test_flip_negative_axis_resolves_against_tensor_rank():
    image = as_tensor(np.arange(12, dtype=np.float32).reshape(3, 4, 1))
    out = Flip(keys=["image"], spatial_axis=-2, input_layout="HWC")(TensorBundle({"image": image}))

    np.testing.assert_allclose(
        ops.convert_to_numpy(out["image"]),
        ops.convert_to_numpy(image)[:, ::-1, :],
    )


@pytest.mark.unit
def test_flip_validates_input_layout():
    with pytest.raises(ValueError, match="supports only input_layout values"):
        Flip(keys=["image"], spatial_axis=0, input_layout="CHW")


@pytest.mark.unit
def test_random_flip_records_random_trace():
    image = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    out = RandomFlip(keys=["image"], prob=1.0, spatial_axis=1, input_layout="HWC")(
        TensorBundle({"image": image})
    )

    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandomFlip"
    assert bool(ops.convert_to_numpy(trace["applied"]))
    assert trace["random"] is True
    assert trace["invertible"] is True
    assert trace["kernel"] == "Flip"


@pytest.mark.unit
def test_random_flip_supports_batch_layout_and_records_input_layout():
    image = as_tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1))
    out = RandomFlip(keys=["image"], prob=1.0, spatial_axis=2, input_layout="BHWC")(
        TensorBundle({"image": image})
    )

    np.testing.assert_allclose(
        ops.convert_to_numpy(out["image"]),
        ops.convert_to_numpy(image)[:, :, ::-1, :],
    )
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_random_flip_replays_with_same_integer_seed():
    image = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))

    first = RandomFlip(keys=["image"], prob=1.0, spatial_axis=1, seed=3, input_layout="HWC")(
        TensorBundle({"image": image})
    )
    second = RandomFlip(keys=["image"], prob=1.0, spatial_axis=1, seed=3, input_layout="HWC")(
        TensorBundle({"image": image})
    )

    np.testing.assert_allclose(
        ops.convert_to_numpy(first["image"]),
        ops.convert_to_numpy(second["image"]),
    )


@pytest.mark.unit
def test_random_flip_inverse_restores_when_applied():
    image = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    transform = RandomFlip(keys=["image"], prob=1.0, spatial_axis=1, input_layout="HWC")

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_random_flip_inverse_is_noop_when_not_applied():
    image = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    transform = RandomFlip(keys=["image"], prob=0.0, spatial_axis=1, input_layout="HWC")

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_random_flip_inverse_without_trace_is_noop():
    bundle = TensorBundle({"image": as_tensor(np.ones((4, 4, 1), dtype=np.float32))})
    transform = RandomFlip(keys=["image"], prob=1.0, spatial_axis=1, input_layout="HWC")

    restored = transform.inverse(bundle)

    assert restored is bundle


@pytest.mark.unit
def test_random_flip_prob_zero_and_allow_missing_keys():
    image = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    out = RandomFlip(keys=["image"], prob=0.0, spatial_axis=1, input_layout="HWC")(
        TensorBundle({"image": image})
    )
    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), ops.convert_to_numpy(image))
    assert not bool(ops.convert_to_numpy(out.get_applied_transforms()[-1]["applied"]))

    bundle = TensorBundle({"other": as_tensor(np.ones((2, 3, 1), dtype=np.float32))})
    transform = RandomFlip(
        keys=["image"], prob=1.0, spatial_axis=1, input_layout="HWC", allow_missing_keys=True
    )
    assert transform(bundle) is bundle


@pytest.mark.unit
def test_random_flip_requires_spatial_axis():
    image = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))

    with pytest.raises(ValueError, match="requires `spatial_axis`"):
        RandomFlip(keys=["image"], prob=1.0, spatial_axis=None, input_layout="HWC")

    with pytest.raises(ValueError, match="supports only input_layout values"):
        RandomFlip(keys=["image"], prob=1.0, spatial_axis=1, input_layout="CHW")



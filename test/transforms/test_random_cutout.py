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
def test_rand_cutout_preserves_shape_and_records_trace():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))

    out = RandomCutOut(
        keys=["image"],
        mask_size=(2, 2),
        num_cuts=2,
        prob=1.0,
        fill_mode="constant",
        input_layout="DHWC",
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (4, 5, 6, 1)
    assert tuple(ops.shape(out["label"])) == (4, 5, 6, 1)
    np.testing.assert_array_equal(
        ops.convert_to_numpy(out["label"]),
        ops.convert_to_numpy(label),
    )
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
        keys=["image"],
        mask_size=(2, 2),
        num_cuts=1,
        prob=1.0,
        input_layout="HWC",
    )(TensorBundle({"image": image_2d, "label": label_2d}))

    image_3d = as_tensor(np.random.randn(4, 8, 8, 1).astype(np.float32))
    label_3d = as_tensor(np.ones((4, 8, 8, 1), dtype=np.float32))
    out_3d = RandomCutOut(
        keys=["image"],
        mask_size=(2, 2),
        num_cuts=1,
        prob=1.0,
        input_layout="DHWC",
    )(TensorBundle({"image": image_3d, "label": label_3d}))

    assert tuple(ops.shape(out_2d["image"])) == (8, 8, 1)
    assert tuple(ops.shape(out_3d["image"])) == (4, 8, 8, 1)


@pytest.mark.unit
def test_random_cutout_samples_2d_centers_from_height_and_width():
    image = as_tensor(np.ones((8, 9, 1), dtype=np.float32))
    transform = RandomCutOut(
        keys=["image"],
        mask_size=(2, 2),
        num_cuts=8,
        prob=1.0,
        input_layout="HWC",
        seed=7,
    )

    centers = ops.convert_to_numpy(transform._sample_cutout_centers(image, spatial_rank=2))

    assert np.all(centers[:, 0] < 8)
    assert np.all(centers[:, 1] < 9)
    assert np.any(centers[:, 1] > 0)


@pytest.mark.unit
def test_random_cutout_supports_batch_layout_and_records_input_layout():
    image_2d = as_tensor(np.ones((2, 8, 8, 1), dtype=np.float32))
    label_2d = as_tensor(np.ones((2, 8, 8, 1), dtype=np.float32))
    image_3d = as_tensor(np.ones((2, 4, 8, 8, 1), dtype=np.float32))
    label_3d = as_tensor(np.ones((2, 4, 8, 8, 1), dtype=np.float32))

    out_2d = RandomCutOut(
        keys=["image"],
        mask_size=(2, 2),
        num_cuts=1,
        prob=1.0,
        input_layout="BHWC",
        seed=7,
    )(TensorBundle({"image": image_2d, "label": label_2d}))
    out_3d = RandomCutOut(
        keys=["image"],
        mask_size=(2, 2),
        num_cuts=1,
        prob=1.0,
        input_layout="BDHWC",
        seed=7,
    )(TensorBundle({"image": image_3d, "label": label_3d}))

    assert tuple(ops.shape(out_2d["image"])) == (2, 8, 8, 1)
    assert tuple(ops.shape(out_3d["image"])) == (2, 4, 8, 8, 1)
    assert out_2d.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"
    assert out_3d.get_applied_transforms()[-1]["params"]["input_layout"] == "BDHWC"


@pytest.mark.unit
def test_random_cutout_accepts_input_layout():
    image = as_tensor(np.ones((2, 8, 8, 1), dtype=np.float32))
    label = as_tensor(np.ones((2, 8, 8, 1), dtype=np.float32))

    out = RandomCutOut(
        keys=["image"],
        mask_size=(2, 2),
        num_cuts=1,
        prob=1.0,
        input_layout="bhwc",
        seed=7,
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (2, 8, 8, 1)
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_random_cutout_batch_mode_replays_with_same_integer_seed():
    image = as_tensor(np.ones((2, 8, 8, 1), dtype=np.float32))
    label = as_tensor(np.ones((2, 8, 8, 1), dtype=np.float32))

    first = RandomCutOut(
        keys=["image"],
        mask_size=(2, 2),
        num_cuts=1,
        prob=1.0,
        input_layout="BHWC",
        seed=11,
    )(TensorBundle({"image": image, "label": label}))
    second = RandomCutOut(
        keys=["image"],
        mask_size=(2, 2),
        num_cuts=1,
        prob=1.0,
        input_layout="BHWC",
        seed=11,
    )(TensorBundle({"image": image, "label": label}))

    np.testing.assert_allclose(
        ops.convert_to_numpy(first["image"]),
        ops.convert_to_numpy(second["image"]),
        rtol=1e-6,
    )


@pytest.mark.unit
def test_random_cutout_validates_arguments():
    with pytest.raises(ValueError, match="`keys` must contain exactly one image key"):
        RandomCutOut(keys=["image", "label"], mask_size=(2, 2), num_cuts=1, input_layout="HWC")

    with pytest.raises(ValueError, match="`mask_size` must be a sequence of two integers"):
        RandomCutOut(keys=["image"], mask_size=(2,), num_cuts=1, input_layout="HWC")

    with pytest.raises(ValueError, match="All values in `mask_size` must be positive integers"):
        RandomCutOut(keys=["image"], mask_size=(2, 0), num_cuts=1, input_layout="HWC")

    with pytest.raises(ValueError, match="`num_cuts` must be a positive integer"):
        RandomCutOut(keys=["image"], mask_size=(2, 2), num_cuts=0, input_layout="HWC")

    with pytest.raises(ValueError, match='`fill_mode` must be either "gaussian" or "constant"'):
        RandomCutOut(
            keys=["image"],
            mask_size=(2, 2),
            num_cuts=1,
            fill_mode="reflect",
            input_layout="HWC",
        )

    with pytest.raises(ValueError, match="`cutout_mode` must be one of"):
        RandomCutOut(
            keys=["image"],
            mask_size=(2, 2),
            num_cuts=1,
            cutout_mode="plane",
            input_layout="HWC",
        )

    with pytest.raises(ValueError, match="supports only input_layout values"):
        RandomCutOut(keys=["image"], mask_size=(2, 2), num_cuts=1, input_layout="CHW")


@pytest.mark.unit
def test_random_cutout_supports_slice_mode_gaussian_mode_and_allow_missing_keys():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))

    out = RandomCutOut(
        keys=["image"],
        mask_size=(2, 2),
        num_cuts=1,
        prob=1.0,
        fill_mode="gaussian",
        cutout_mode="slice",
        input_layout="DHWC",
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (4, 5, 6, 1)

    skip = RandomCutOut(
        keys=["image"],
        mask_size=(2, 2),
        num_cuts=1,
        input_layout="DHWC",
        allow_missing_keys=True,
    )
    bundle = TensorBundle({"image": image})
    assert skip(bundle) is bundle


@pytest.mark.unit
def test_random_cutout_2d_supports_slice_mode():
    image = as_tensor(np.random.randn(8, 8, 1).astype(np.float32))
    label = as_tensor(np.ones((8, 8, 1), dtype=np.float32))

    out = RandomCutOut(
        keys=["image"],
        mask_size=(2, 2),
        num_cuts=1,
        prob=1.0,
        cutout_mode="slice",
        input_layout="HWC",
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (8, 8, 1)


@pytest.mark.unit
def test_random_cutout_mask_size_one_affects_at_least_one_pixel():
    image = as_tensor(np.ones((8, 8, 1), dtype=np.float32))
    label = as_tensor(np.ones((8, 8, 1), dtype=np.float32))

    out = RandomCutOut(
        keys=["image"],
        mask_size=(1, 1),
        num_cuts=1,
        prob=1.0,
        fill_mode="constant",
        fill_value=0.0,
        input_layout="HWC",
    )(TensorBundle({"image": image, "label": label}))

    assert np.any(ops.convert_to_numpy(out["image"]) == 0.0)


@pytest.mark.unit
def test_random_cutout_prob_zero_and_unsupported_rank_rejection():
    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))
    out = RandomCutOut(
        keys=["image"],
        mask_size=(2, 2),
        num_cuts=1,
        prob=0.0,
        input_layout="DHWC",
    )(TensorBundle({"image": image, "label": label}))
    assert not bool(ops.convert_to_numpy(out.get_applied_transforms()[-1]["applied"]))

    image_1d_like = as_tensor(np.ones((6, 1), dtype=np.float32))
    label_1d_like = as_tensor(np.ones((6, 1), dtype=np.float32))
    transform = RandomCutOut(
        keys=["image"],
        mask_size=(2, 2),
        num_cuts=1,
        input_layout="HWC",
    )
    with pytest.raises(ValueError, match="expects input_layout='HWC' with rank 3"):
        transform(TensorBundle({"image": image_1d_like, "label": label_1d_like}))



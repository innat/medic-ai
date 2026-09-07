import numpy as np
import pytest
from keras import ops

from medicai.transforms import (
    RandomCropByPosNegLabel,
    TensorBundle,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_uses_spatial_crop_kernel():
    image = as_tensor(np.random.randn(6, 6, 6, 1).astype(np.float32))
    label = as_tensor(np.zeros((6, 6, 6, 1), dtype=np.float32))
    label = ops.convert_to_tensor(
        np.pad(np.ones((2, 2, 2, 1), dtype=np.float32), ((2, 2), (2, 2), (2, 2), (0, 0)))
    )

    out = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(3, 3, 3),
        pos=1,
        neg=1,
        input_layout="DHWC",
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (3, 3, 3, 1)
    assert tuple(ops.shape(out["label"])) == (3, 3, 3, 1)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandomCropByPosNegLabel"
    assert trace["kernel"] == "SpatialCrop"
    assert trace["random"] is True
    assert trace["invertible"] is True


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_supports_2d_and_3d():
    image_2d = as_tensor(np.random.randn(8, 8, 1).astype(np.float32))
    label_2d = as_tensor(np.pad(np.ones((2, 2, 1), dtype=np.float32), ((3, 3), (3, 3), (0, 0))))
    out_2d = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(4, 4),
        pos=1,
        neg=1,
        input_layout="HWC",
    )(TensorBundle({"image": image_2d, "label": label_2d}))

    image_3d = as_tensor(np.random.randn(6, 6, 6, 1).astype(np.float32))
    label_3d = ops.convert_to_tensor(
        np.pad(np.ones((2, 2, 2, 1), dtype=np.float32), ((2, 2), (2, 2), (2, 2), (0, 0)))
    )
    out_3d = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(3, 3, 3),
        pos=1,
        neg=1,
        input_layout="DHWC",
    )(TensorBundle({"image": image_3d, "label": label_3d}))

    assert tuple(ops.shape(out_2d["image"])) == (4, 4, 1)
    assert tuple(ops.shape(out_2d["label"])) == (4, 4, 1)
    assert tuple(ops.shape(out_3d["image"])) == (3, 3, 3, 1)
    assert tuple(ops.shape(out_3d["label"])) == (3, 3, 3, 1)


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_supports_batch_mode_and_records_input_layout():
    image_2d = as_tensor(np.random.randn(2, 8, 8, 1).astype(np.float32))
    label_2d = as_tensor(np.zeros((2, 8, 8, 1), dtype=np.float32))
    label_2d_np = ops.convert_to_numpy(label_2d)
    label_2d_np[:, 3:5, 3:5, 0] = 1.0
    label_2d = as_tensor(label_2d_np)

    out_2d = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(4, 4),
        pos=1,
        neg=1,
        input_layout="BHWC",
    )(TensorBundle({"image": image_2d, "label": label_2d}))

    image_3d = as_tensor(np.random.randn(2, 6, 6, 6, 1).astype(np.float32))
    label_3d = as_tensor(np.zeros((2, 6, 6, 6, 1), dtype=np.float32))
    label_3d_np = ops.convert_to_numpy(label_3d)
    label_3d_np[:, 2:4, 2:4, 2:4, 0] = 1.0
    label_3d = as_tensor(label_3d_np)

    out_3d = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(3, 3, 3),
        pos=1,
        neg=1,
        input_layout="BDHWC",
    )(TensorBundle({"image": image_3d, "label": label_3d}))

    assert tuple(ops.shape(out_2d["image"])) == (2, 4, 4, 1)
    assert tuple(ops.shape(out_2d["label"])) == (2, 4, 4, 1)
    assert tuple(ops.shape(out_3d["image"])) == (2, 3, 3, 3, 1)
    assert tuple(ops.shape(out_3d["label"])) == (2, 3, 3, 3, 1)
    assert out_2d.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"
    assert out_3d.get_applied_transforms()[-1]["params"]["input_layout"] == "BDHWC"


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_accepts_input_layout():
    image = as_tensor(np.random.randn(2, 8, 8, 1).astype(np.float32))
    label = as_tensor(np.zeros((2, 8, 8, 1), dtype=np.float32))
    label_np = ops.convert_to_numpy(label)
    label_np[:, 3:5, 3:5, 0] = 1.0
    label = as_tensor(label_np)

    out = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(4, 4),
        pos=1,
        neg=1,
        input_layout="BHWC",
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (2, 4, 4, 1)
    assert tuple(ops.shape(out["label"])) == (2, 4, 4, 1)
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_shares_sampled_crop_across_batched_input():
    image = as_tensor(np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6, 1))
    label = as_tensor(np.zeros((2, 6, 6, 1), dtype=np.float32))
    label_np = ops.convert_to_numpy(label)
    label_np[:, 2:4, 2:4, 0] = 1.0
    label = as_tensor(label_np)

    transform = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(3, 3),
        pos=1,
        neg=1,
        input_layout="BHWC",
        seed=23,
    )

    out = transform(TensorBundle({"image": image, "label": label}))
    crop_start = ops.convert_to_numpy(out.get_applied_transforms()[-1]["params"]["crop_start"])
    original = ops.convert_to_numpy(image)
    expected = original[:, crop_start[0] : crop_start[0] + 3, crop_start[1] : crop_start[1] + 3, :]

    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), expected)


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_inverse_restores_original_canvas_for_2d():
    image = as_tensor(np.zeros((6, 6, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[1:5, 1:5, 0] = np.arange(16, dtype=np.float32).reshape(4, 4)
    image = as_tensor(image_np)
    label = as_tensor(np.zeros((6, 6, 1), dtype=np.float32))
    label_np = ops.convert_to_numpy(label)
    label_np[3, 3, 0] = 1.0
    label = as_tensor(label_np)

    transform = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(4, 4),
        pos=1,
        neg=0,
        input_layout="HWC",
    )
    forward = transform(TensorBundle({"image": image, "label": label}))
    restored = transform.inverse(
        TensorBundle({"image": forward["image"], "label": forward["label"]}, forward.meta)
    )

    assert tuple(ops.shape(restored["image"])) == (6, 6, 1)
    assert tuple(ops.shape(restored["label"])) == (6, 6, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["label"]),
        ops.convert_to_numpy(label),
    )


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_inverse_restores_batched_input_canvas():
    image = as_tensor(np.zeros((2, 6, 6, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[:, 2:5, 2:5, 0] = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    image = as_tensor(image_np)
    label = as_tensor(np.zeros((2, 6, 6, 1), dtype=np.float32))
    label_np = ops.convert_to_numpy(label)
    label_np[:, 3, 3, 0] = 1.0
    label = as_tensor(label_np)

    transform = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(3, 3),
        pos=1,
        neg=0,
        input_layout="BHWC",
    )
    forward = transform(TensorBundle({"image": image, "label": label}))
    restored = transform.inverse(
        TensorBundle({"image": forward["image"], "label": forward["label"]}, forward.meta)
    )

    assert tuple(ops.shape(restored["image"])) == (2, 6, 6, 1)
    assert tuple(ops.shape(restored["label"])) == (2, 6, 6, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["label"]),
        ops.convert_to_numpy(label),
    )


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_inverse_restores_original_canvas_for_3d():
    image = as_tensor(np.zeros((6, 6, 6, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[2:5, 2:5, 2:5, 0] = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    image = as_tensor(image_np)
    label = as_tensor(np.zeros((6, 6, 6, 1), dtype=np.float32))
    label_np = ops.convert_to_numpy(label)
    label_np[3, 3, 3, 0] = 1.0
    label = as_tensor(label_np)

    transform = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(3, 3, 3),
        pos=1,
        neg=0,
        input_layout="DHWC",
    )
    forward = transform(TensorBundle({"image": image, "label": label}))
    restored = transform.inverse(
        TensorBundle({"image": forward["image"], "label": forward["label"]}, forward.meta)
    )

    assert tuple(ops.shape(restored["image"])) == (6, 6, 6, 1)
    assert tuple(ops.shape(restored["label"])) == (6, 6, 6, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["label"]),
        ops.convert_to_numpy(label),
    )


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_inverse_without_trace_is_noop():
    bundle = TensorBundle(
        {
            "image": as_tensor(np.ones((4, 4, 1), dtype=np.float32)),
            "label": as_tensor(np.ones((4, 4, 1), dtype=np.float32)),
        }
    )
    transform = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(2, 2),
        pos=1,
        neg=1,
        input_layout="HWC",
    )

    restored = transform.inverse(bundle)

    assert restored is bundle


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_validates_arguments():
    with pytest.raises(ValueError, match="pos and neg must be non-negative"):
        RandomCropByPosNegLabel(
            keys=["image", "label"],
            target_shape=(2, 2, 2),
            pos=-1,
            neg=1,
            input_layout="DHWC",
        )

    with pytest.raises(ValueError, match="pos and neg cannot both be zero"):
        RandomCropByPosNegLabel(
            keys=["image", "label"],
            target_shape=(2, 2, 2),
            pos=0,
            neg=0,
            input_layout="DHWC",
        )

    with pytest.raises(ValueError, match="requires a pair of image and label as keys"):
        RandomCropByPosNegLabel(
            keys=["image"], target_shape=(2, 2, 2), pos=1, neg=1, input_layout="DHWC"
        )

    with pytest.raises(ValueError, match="currently supports only num_samples=1"):
        RandomCropByPosNegLabel(
            keys=["image", "label"],
            target_shape=(2, 2, 2),
            pos=1,
            neg=1,
            num_samples=2,
            input_layout="DHWC",
        )


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_rejects_2d_and_supports_allow_missing_keys():
    image_1d_like = as_tensor(np.ones((6, 1), dtype=np.float32))
    label_1d_like = as_tensor(np.ones((6, 1), dtype=np.float32))
    transform = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(2, 2),
        pos=1,
        neg=1,
        input_layout="HWC",
    )

    with pytest.raises(ValueError, match="expects input_layout='HWC' with rank 3"):
        transform(TensorBundle({"image": image_1d_like, "label": label_1d_like}))

    image_2d = as_tensor(np.ones((6, 6, 1), dtype=np.float32))
    label_2d = as_tensor(np.ones((6, 6, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="`target_shape` must contain exactly 2 values"):
        RandomCropByPosNegLabel(
            keys=["image", "label"],
            target_shape=(2, 2, 2),
            pos=1,
            neg=1,
            input_layout="HWC",
        )(TensorBundle({"image": image_2d, "label": label_2d}))

    skip_transform = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(2, 2, 2),
        pos=1,
        neg=1,
        input_layout="DHWC",
        allow_missing_keys=True,
    )
    bundle = TensorBundle({"image": as_tensor(np.ones((4, 4, 4, 1), dtype=np.float32))})
    assert skip_transform(bundle) is bundle


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_validates_input_layout_and_layout_contract():
    with pytest.raises(ValueError, match="supports only input_layout values"):
        RandomCropByPosNegLabel(
            keys=["image", "label"],
            target_shape=(2, 2),
            pos=1,
            neg=1,
            input_layout="CHW",
        )

    image = as_tensor(np.ones((6, 6, 1), dtype=np.float32))
    label = as_tensor(np.ones((6, 6, 1), dtype=np.float32))
    transform = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(2, 2),
        pos=1,
        neg=1,
        input_layout="BHWC",
    )

    with pytest.raises(ValueError, match="expects input_layout='BHWC' with rank 4"):
        transform(TensorBundle({"image": image, "label": label}))


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_validates_image_reference_key():
    image = as_tensor(np.ones((4, 4, 4, 1), dtype=np.float32))
    label = as_tensor(np.ones((4, 4, 4, 1), dtype=np.float32))

    transform = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(2, 2, 2),
        pos=1,
        neg=1,
        input_layout="DHWC",
        image_reference_key="reference",
    )

    with pytest.raises(KeyError, match="reference"):
        transform(TensorBundle({"image": image, "label": label}))

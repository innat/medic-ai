import numpy as np
import pytest
from keras import ops

from medicai.transforms import (
    Resize,
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
        Resize(
            keys=["image", "label"],
            interpolation=interpolation,
            target_shape=target_shape,
            input_layout="HWC",
        )


@pytest.mark.unit
def test_resize_accepts_mapping_interpolation_and_allow_missing_keys():
    image = as_tensor(np.random.randn(5, 6, 1).astype(np.float32))
    transform = Resize(
        keys=["image", "label"],
        interpolation={"image": "bilinear", "label": "nearest"},
        target_shape=(3, 4),
        input_layout="HWC",
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
            input_layout="HWC",
        )


@pytest.mark.unit
def test_resize_transform_for_2d_and_3d():
    inputs_2d_sample = TensorBundle(
        {
            "image": as_tensor(np.random.randn(32, 32, 1).astype(np.float32)),
            "label": as_tensor(np.random.randint(0, 2, (32, 32, 1)).astype(np.float32)),
        }
    )
    out_2d_sample = Resize(
        keys=["image", "label"],
        interpolation=("bilinear", "nearest"),
        target_shape=(24, 20),
        input_layout="HWC",
    )(inputs_2d_sample)
    assert tuple(ops.shape(out_2d_sample["image"])) == (24, 20, 1)
    assert tuple(ops.shape(out_2d_sample["label"])) == (24, 20, 1)

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
        input_layout="BHWC",
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
        input_layout="DHWC",
    )(inputs_3d)
    assert tuple(ops.shape(out_3d["image"])) == (8, 10, 12, 1)
    assert tuple(ops.shape(out_3d["label"])) == (8, 10, 12, 1)
    trace = out_3d.get_applied_transforms()[-1]
    assert trace["name"] == "Resize"
    assert trace["invertible"] is True


@pytest.mark.unit
def test_resize_supports_batch_layout_for_3d_and_records_input_layout():
    image = as_tensor(np.random.randn(2, 8, 10, 12, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (2, 8, 10, 12, 1)).astype(np.float32))

    out = Resize(
        keys=["image", "label"],
        interpolation=("trilinear", "nearest"),
        target_shape=(4, 5, 6),
        input_layout="BDHWC",
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (2, 4, 5, 6, 1)
    assert tuple(ops.shape(out["label"])) == (2, 4, 5, 6, 1)
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "BDHWC"


@pytest.mark.unit
def test_resize_accepts_input_layout():
    image = as_tensor(np.random.randn(2, 6, 7, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (2, 6, 7, 1)).astype(np.float32))

    out = Resize(
        keys=["image", "label"],
        interpolation=("bilinear", "nearest"),
        target_shape=(4, 5),
        input_layout="BHWC",
    )(TensorBundle({"image": image, "label": label}))

    assert tuple(ops.shape(out["image"])) == (2, 4, 5, 1)
    assert tuple(ops.shape(out["label"])) == (2, 4, 5, 1)
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_resize_uses_same_batch_kernel_for_sample_and_batch_modes():
    sample_2d = as_tensor(np.random.randn(6, 8, 1).astype(np.float32))
    sample_3d = as_tensor(np.random.randn(5, 6, 7, 1).astype(np.float32))
    batch_2d = ops.stack([sample_2d, sample_2d * 2.0], axis=0)
    batch_3d = ops.stack([sample_3d, sample_3d * 2.0], axis=0)

    resize_2d = Resize(
        keys=["image"],
        interpolation="bilinear",
        target_shape=(3, 4),
        input_layout="HWC",
    )
    resize_3d = Resize(
        keys=["image"],
        interpolation="trilinear",
        target_shape=(3, 4, 5),
        input_layout="DHWC",
    )

    sample_2d_out = ops.convert_to_numpy(
        resize_2d.resize_batch_tensor(
            sample_2d[None, ...], "image", ops.convert_to_tensor([3, 4], dtype="int32")
        )
    )[0]
    sample_3d_out = ops.convert_to_numpy(
        resize_3d.resize_batch_tensor(
            sample_3d[None, ...],
            "image",
            ops.convert_to_tensor([3, 4, 5], dtype="int32"),
        )
    )[0]
    batch_2d_out = ops.convert_to_numpy(
        resize_2d.resize_batch_tensor(
            batch_2d, "image", ops.convert_to_tensor([3, 4], dtype="int32")
        )
    )
    batch_3d_out = ops.convert_to_numpy(
        resize_3d.resize_batch_tensor(
            batch_3d, "image", ops.convert_to_tensor([3, 4, 5], dtype="int32")
        )
    )

    assert sample_2d_out.shape == (3, 4, 1)
    assert sample_3d_out.shape == (3, 4, 5, 1)
    assert batch_2d_out.shape == (2, 3, 4, 1)
    assert batch_3d_out.shape == (2, 3, 4, 5, 1)
    np.testing.assert_allclose(batch_2d_out[0], sample_2d_out)
    np.testing.assert_allclose(batch_2d_out[1], sample_2d_out * 2.0)
    np.testing.assert_allclose(batch_3d_out[0], sample_3d_out)
    np.testing.assert_allclose(batch_3d_out[1], sample_3d_out * 2.0)


@pytest.mark.unit
def test_resize_validates_input_layout_and_layout_contract():
    with pytest.raises(ValueError, match="supports only input_layout values"):
        Resize(keys=["image"], interpolation="bilinear", target_shape=(4, 4), input_layout="CHW")

    transform = Resize(
        keys=["image"],
        interpolation="bilinear",
        target_shape=(4, 4),
        input_layout="HWC",
    )
    image = as_tensor(np.random.randn(2, 8, 8, 1).astype(np.float32))

    with pytest.raises(ValueError, match="expects input_layout='HWC' with rank 3"):
        transform(TensorBundle({"image": image}))


@pytest.mark.unit
def test_resize_inverse_restores_original_spatial_shape():
    image = as_tensor(np.random.randn(6, 8, 1).astype(np.float32))
    resize = Resize(
        keys=["image"], interpolation="bilinear", target_shape=(3, 4), input_layout="HWC"
    )

    forward = resize(TensorBundle({"image": image}))
    restored = resize.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(forward["image"])) == (3, 4, 1)
    assert tuple(ops.shape(restored["image"])) == (6, 8, 1)


@pytest.mark.unit
def test_resize_inverse_without_trace_is_noop():
    bundle = TensorBundle({"image": as_tensor(np.ones((4, 4, 1), dtype=np.float32))})
    resize = Resize(
        keys=["image"], interpolation="bilinear", target_shape=(2, 2), input_layout="HWC"
    )

    restored = resize.inverse(bundle)

    assert restored is bundle



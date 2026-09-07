import numpy as np
import pytest
from keras import ops

from medicai.transforms import (
    RandomRotate90,
    Rotate90,
    TensorBundle,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_rotate90_supports_2d_and_3d_and_records_inverse_trace():
    image_2d = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    image_3d = as_tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1))

    rotate_2d = Rotate90(keys=["image"], k=1, input_layout="HWC")
    rotate_3d = Rotate90(keys=["image"], k=3, spatial_axis=(1, 2), input_layout="DHWC")

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
        ops.convert_to_numpy(
            rotate_2d.inverse(TensorBundle({"image": out_2d["image"]}, out_2d.meta))["image"]
        ),
        ops.convert_to_numpy(image_2d),
    )


@pytest.mark.unit
def test_rotate90_supports_batch_mode_for_2d_and_3d_channel_last_tensors():
    batch_2d = as_tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1))
    batch_3d = as_tensor(np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5, 1))

    rotate_2d = Rotate90(keys=["image"], k=1, input_layout="BHWC")
    rotate_3d = Rotate90(keys=["image"], k=3, spatial_axis=(2, 3), input_layout="BDHWC")

    out_2d = rotate_2d(TensorBundle({"image": batch_2d}))
    out_3d = rotate_3d(TensorBundle({"image": batch_3d}))

    expected_2d = np.rot90(ops.convert_to_numpy(batch_2d), k=1, axes=(1, 2))
    expected_3d = np.rot90(ops.convert_to_numpy(batch_3d), k=3, axes=(2, 3))

    np.testing.assert_allclose(ops.convert_to_numpy(out_2d["image"]), expected_2d)
    np.testing.assert_allclose(ops.convert_to_numpy(out_3d["image"]), expected_3d)
    assert out_2d.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_rotate90_accepts_input_layout_with_real_tensor_axes():
    batch_2d = as_tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1))
    batch_3d = as_tensor(np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5, 1))

    rotate_2d = Rotate90(keys=["image"], k=1, spatial_axis=(1, 2), input_layout="BHWC")
    rotate_3d = Rotate90(keys=["image"], k=3, spatial_axis=(2, 3), input_layout="BDHWC")

    out_2d = rotate_2d(TensorBundle({"image": batch_2d}))
    out_3d = rotate_3d(TensorBundle({"image": batch_3d}))

    expected_2d = np.rot90(ops.convert_to_numpy(batch_2d), k=1, axes=(1, 2))
    expected_3d = np.rot90(ops.convert_to_numpy(batch_3d), k=3, axes=(2, 3))

    np.testing.assert_allclose(ops.convert_to_numpy(out_2d["image"]), expected_2d)
    np.testing.assert_allclose(ops.convert_to_numpy(out_3d["image"]), expected_3d)
    assert out_2d.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_rotate90_accepts_numpy_mapping_inputs():
    image = np.arange(6, dtype=np.float32).reshape(2, 3, 1)

    out = Rotate90(keys=["image"], k=1, input_layout="HWC")({"image": image})

    np.testing.assert_allclose(
        ops.convert_to_numpy(out["image"]), np.rot90(image, k=1, axes=(0, 1))
    )


@pytest.mark.unit
def test_rotate90_uses_same_batch_kernel_for_sample_and_batch_modes():
    sample_2d = as_tensor(np.arange(12, dtype=np.float32).reshape(3, 4, 1))
    sample_3d = as_tensor(np.arange(60, dtype=np.float32).reshape(3, 4, 5, 1))
    batch_2d = as_tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1))
    batch_3d = as_tensor(np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5, 1))

    rotate_2d = Rotate90(keys=["image"], k=1, input_layout="HWC")
    rotate_3d = Rotate90(keys=["image"], k=3, spatial_axis=(1, 2), input_layout="DHWC")

    expected_sample_2d = np.rot90(ops.convert_to_numpy(sample_2d), k=1, axes=(0, 1))
    expected_sample_3d = np.rot90(ops.convert_to_numpy(sample_3d), k=3, axes=(1, 2))
    expected_batch_2d = np.rot90(ops.convert_to_numpy(batch_2d), k=1, axes=(1, 2))
    expected_batch_3d = np.rot90(ops.convert_to_numpy(batch_3d), k=3, axes=(2, 3))

    np.testing.assert_allclose(
        ops.convert_to_numpy(rotate_2d.rotate_batch_tensor(sample_2d[None, ...]))[0],
        expected_sample_2d,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(rotate_3d.rotate_batch_tensor(sample_3d[None, ...]))[0],
        expected_sample_3d,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(rotate_2d.rotate_batch_tensor(batch_2d)),
        expected_batch_2d,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(rotate_3d.rotate_batch_tensor(batch_3d)),
        expected_batch_3d,
    )


@pytest.mark.unit
def test_rotate90_k_zero_is_noop_and_invalid_axes_raise():
    image = as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    out = Rotate90(keys=["image"], k=4, input_layout="HWC")(TensorBundle({"image": image}))
    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), ops.convert_to_numpy(image))
    assert out.get_applied_transforms() == []

    with pytest.raises(ValueError, match="must contain exactly two axes"):
        Rotate90(keys=["image"], k=1, spatial_axis=(0,), input_layout="HWC")(
            TensorBundle({"image": image})
        )


@pytest.mark.unit
def test_rotate90_negative_axes_resolve_against_tensor_rank():
    image = as_tensor(np.arange(12, dtype=np.float32).reshape(3, 4, 1))
    out = Rotate90(keys=["image"], k=1, spatial_axis=(0, -2), input_layout="HWC")(
        TensorBundle({"image": image})
    )

    expected = np.rot90(ops.convert_to_numpy(image), k=1, axes=(0, 1))
    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), expected)


@pytest.mark.unit
def test_rotate90_validates_input_layout():
    with pytest.raises(ValueError, match="supports only input_layout values"):
        Rotate90(keys=["image"], k=1, input_layout="CHW")


@pytest.mark.unit
@pytest.mark.parametrize("k", [1, 2, 3], ids=["k1", "k2", "k3"])
def test_rotate90_branch_parity_for_non_square_2d_inputs(k):
    image = as_tensor(np.arange(15, dtype=np.float32).reshape(3, 5, 1))

    out = Rotate90(keys=["image"], k=k, input_layout="HWC")(TensorBundle({"image": image}))
    expected = np.rot90(ops.convert_to_numpy(image), k=k, axes=(0, 1))

    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("k", "spatial_axis"),
    [
        (1, (1, 2)),
        (2, (1, 2)),
        (3, (1, 2)),
        (1, (0, 2)),
        (2, (0, 2)),
        (3, (0, 2)),
    ],
    ids=[
        "k1_hw",
        "k2_hw",
        "k3_hw",
        "k1_dw",
        "k2_dw",
        "k3_dw",
    ],
)
def test_rotate90_branch_parity_for_anisotropic_3d_inputs(k, spatial_axis):
    image = as_tensor(np.arange(60, dtype=np.float32).reshape(3, 4, 5, 1))

    out = Rotate90(keys=["image"], k=k, spatial_axis=spatial_axis, input_layout="DHWC")(
        TensorBundle({"image": image})
    )
    expected = np.rot90(ops.convert_to_numpy(image), k=k, axes=spatial_axis)

    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), expected)


@pytest.mark.unit
def test_random_rotate90_preserves_shape():
    image = as_tensor(np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2))
    out = RandomRotate90(keys=["image"], prob=1.0, max_k=3, input_layout="DHWC")(
        TensorBundle({"image": image})
    )
    assert tuple(ops.shape(out["image"])) == (1, 2, 2, 2)
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "RandomRotate90"
    assert bool(ops.convert_to_numpy(trace["applied"]))
    assert trace["random"] is True
    assert trace["invertible"] is True
    assert trace["kernel"] == "Rotate90"


@pytest.mark.unit
def test_random_rotate90_rejects_rectangular_rotation_plane():
    image = as_tensor(np.zeros((2, 3, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="equal sizes for the selected rotation axes"):
        RandomRotate90(keys=["image"], prob=1.0, input_layout="HWC")(TensorBundle({"image": image}))


@pytest.mark.unit
def test_random_rotate90_supports_batch_layout_and_records_input_layout():
    image = as_tensor(np.arange(18, dtype=np.float32).reshape(2, 3, 3, 1))
    out = RandomRotate90(keys=["image"], prob=1.0, max_k=3, input_layout="BHWC")(
        TensorBundle({"image": image})
    )

    assert tuple(ops.shape(out["image"])) == (2, 3, 3, 1)
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_random_rotate90_replays_with_same_integer_seed():
    image = as_tensor(np.arange(9, dtype=np.float32).reshape(3, 3, 1))

    first = RandomRotate90(keys=["image"], prob=1.0, max_k=3, seed=5, input_layout="HWC")(
        TensorBundle({"image": image})
    )
    second = RandomRotate90(keys=["image"], prob=1.0, max_k=3, seed=5, input_layout="HWC")(
        TensorBundle({"image": image})
    )

    np.testing.assert_allclose(
        ops.convert_to_numpy(first["image"]),
        ops.convert_to_numpy(second["image"]),
    )


@pytest.mark.unit
def test_random_rotate90_shares_sampled_rotation_across_batched_input():
    image = as_tensor(np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3, 1))
    transform = RandomRotate90(
        keys=["image"],
        prob=1.0,
        max_k=3,
        input_layout="BHWC",
        seed=9,
    )

    out = transform(TensorBundle({"image": image}))
    rotated = ops.convert_to_numpy(out["image"])
    original = ops.convert_to_numpy(image)
    k = int(ops.convert_to_numpy(out.get_applied_transforms()[-1]["params"]["k"]))

    np.testing.assert_allclose(rotated[0], np.rot90(original[0], k=k, axes=(0, 1)))
    np.testing.assert_allclose(rotated[1], np.rot90(original[1], k=k, axes=(0, 1)))


@pytest.mark.unit
def test_random_rotate90_inverse_restores_batched_input():
    image = as_tensor(np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3, 1))
    transform = RandomRotate90(
        keys=["image"],
        prob=1.0,
        max_k=3,
        input_layout="BHWC",
        seed=9,
    )

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_random_rotate90_inverse_restores_when_applied():
    image = as_tensor(np.arange(9, dtype=np.float32).reshape(3, 3, 1))
    transform = RandomRotate90(keys=["image"], prob=1.0, max_k=3, input_layout="HWC")

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_random_rotate90_inverse_is_noop_when_not_applied():
    image = as_tensor(np.arange(4, dtype=np.float32).reshape(2, 2, 1))
    transform = RandomRotate90(keys=["image"], prob=0.0, max_k=3, input_layout="HWC")

    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_random_rotate90_inverse_without_trace_is_noop():
    bundle = TensorBundle({"image": as_tensor(np.ones((4, 4, 1), dtype=np.float32))})
    transform = RandomRotate90(keys=["image"], prob=1.0, max_k=3, input_layout="HWC")

    restored = transform.inverse(bundle)

    assert restored is bundle


@pytest.mark.unit
def test_random_rotate90_prob_zero_records_no_application():
    image = as_tensor(np.arange(4, dtype=np.float32).reshape(2, 2, 1))
    out = RandomRotate90(keys=["image"], prob=0.0, max_k=3, input_layout="HWC")(
        TensorBundle({"image": image})
    )

    np.testing.assert_allclose(ops.convert_to_numpy(out["image"]), ops.convert_to_numpy(image))
    assert not bool(ops.convert_to_numpy(out.get_applied_transforms()[-1]["applied"]))


@pytest.mark.unit
def test_random_rotate90_validates_max_k():
    with pytest.raises(ValueError, match="must be >= 1"):
        RandomRotate90(keys=["image"], max_k=0, input_layout="HWC")

    with pytest.raises(ValueError, match="supports only input_layout values"):
        RandomRotate90(keys=["image"], input_layout="CHW")


@pytest.mark.unit
def test_random_rotate90_prob_zero_is_noop_for_rectangular_input():
    image = as_tensor(np.arange(2 * 3, dtype=np.float32).reshape(2, 3, 1))
    transform = RandomRotate90(
        keys=["image"],
        prob=0.0,
        max_k=3,
        spatial_axis=(0, 1),
        input_layout="HWC",
    )

    output = transform(TensorBundle({"image": image}))

    np.testing.assert_array_equal(
        ops.convert_to_numpy(output["image"]),
        ops.convert_to_numpy(image),
    )



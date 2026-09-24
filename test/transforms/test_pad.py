import numpy as np
import pytest
from keras import ops

from medicai.transforms import Pad, PadIfNeeded, TensorBundle


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_pad_supports_sample_and_batch_layouts():
    sample = as_tensor(np.ones((3, 4, 1), dtype=np.float32))
    batch = as_tensor(np.ones((2, 3, 4, 1), dtype=np.float32))

    sample_result = Pad(keys=["image"], padding=((1, 2), (3, 4)), input_layout="HWC")(
        {"image": sample}
    )
    batch_result = Pad(keys=["image"], padding=((1, 2), (3, 4)), input_layout="BHWC")(
        {"image": batch}
    )

    assert tuple(ops.shape(sample_result["image"])) == (6, 11, 1)
    assert tuple(ops.shape(batch_result["image"])) == (2, 6, 11, 1)


@pytest.mark.unit
def test_pad_uses_per_key_constant_values_and_preserves_alignment():
    image = as_tensor(np.ones((2, 2, 1), dtype=np.float32))
    label = as_tensor(np.ones((2, 2, 1), dtype=np.int32))

    result = Pad(
        keys=["image", "label"],
        padding=1,
        fill_value={"image": -1.0, "label": 7},
        input_layout="HWC",
    )(TensorBundle({"image": image, "label": label}))

    np.testing.assert_array_equal(ops.convert_to_numpy(result["image"])[0, :, 0], -1.0)
    np.testing.assert_array_equal(ops.convert_to_numpy(result["label"])[0, :, 0], 7)
    np.testing.assert_array_equal(ops.convert_to_numpy(result["image"])[1:3, 1:3], 1.0)


@pytest.mark.unit
def test_pad_inverse_restores_original_sample_and_batch_shapes():
    for layout, shape, padding in (
        ("DHWC", (2, 3, 4, 1), ((1, 2), (2, 1), (3, 4))),
        ("BDHWC", (2, 2, 3, 4, 1), ((1, 2), (2, 1), (3, 4))),
    ):
        image = as_tensor(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
        transform = Pad(keys=["image"], padding=padding, input_layout=layout)
        forward = transform(TensorBundle({"image": image}))
        restored = transform.inverse(forward)

        assert tuple(ops.shape(restored["image"])) == shape
        np.testing.assert_array_equal(
            ops.convert_to_numpy(restored["image"]), ops.convert_to_numpy(image)
        )


@pytest.mark.unit
def test_pad_rejects_unsupported_fill_mode_and_negative_padding():
    with pytest.raises(ValueError, match="fill_mode"):
        Pad(keys=["image"], padding=1, fill_mode="edge", input_layout="HWC")

    with pytest.raises(ValueError, match="non-negative"):
        Pad(keys=["image"], padding=(-1, 0), input_layout="HWC")


@pytest.mark.unit
def test_pad_if_needed_reaches_minimum_shape_and_divisibility():
    image = as_tensor(np.ones((5, 7, 1), dtype=np.float32))
    transform = PadIfNeeded(
        keys=["image"],
        min_target_shape=(8, 8),
        divisible_by=(4, 4),
        input_layout="HWC",
    )
    result = transform({"image": image})

    assert tuple(ops.shape(result["image"])) == (8, 8, 1)
    trace = result.get_applied_transforms()[-1]
    assert trace["params"]["padding"] == ((1, 2), (0, 1))


@pytest.mark.unit
def test_pad_if_needed_is_noop_when_shape_is_already_valid():
    image = as_tensor(np.ones((8, 12, 1), dtype=np.float32))
    transform = PadIfNeeded(
        keys=["image"],
        min_target_shape=(8, 8),
        divisible_by=4,
        input_layout="HWC",
    )
    result = transform({"image": image})

    assert tuple(ops.shape(result["image"])) == (8, 12, 1)
    restored = transform.inverse(result)
    np.testing.assert_array_equal(
        ops.convert_to_numpy(restored["image"]), ops.convert_to_numpy(image)
    )


@pytest.mark.unit
@pytest.mark.parametrize("fill_mode", ["reflect", "symmetric"])
def test_pad_supports_portable_nonconstant_fill_modes(fill_mode):
    image = as_tensor(np.arange(2 * 3, dtype=np.float32).reshape(2, 3, 1))
    result = Pad(
        keys=["image"],
        padding=1,
        fill_mode=fill_mode,
        input_layout="HWC",
    )({"image": image})

    assert tuple(ops.shape(result["image"])) == (4, 5, 1)
    np.testing.assert_array_equal(
        ops.convert_to_numpy(result["image"])[1:3, 1:4],
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_pad_supports_per_key_fill_mode_mapping():
    image = as_tensor(np.arange(4, dtype=np.float32).reshape(2, 2, 1))
    label = as_tensor(np.ones((2, 2, 1), dtype=np.int32))
    result = Pad(
        keys=["image", "label"],
        padding=1,
        fill_mode={"image": "reflect", "label": "constant"},
        fill_value={"image": -1.0, "label": 5},
        input_layout="HWC",
    )({"image": image, "label": label})

    assert tuple(ops.shape(result["image"])) == (4, 4, 1)
    assert tuple(ops.shape(result["label"])) == (4, 4, 1)
    np.testing.assert_array_equal(
        ops.convert_to_numpy(result["image"])[1:3, 1:3],
        ops.convert_to_numpy(image),
    )
    np.testing.assert_array_equal(ops.convert_to_numpy(result["label"])[0, :, 0], 5)


@pytest.mark.unit
def test_pad_if_needed_supports_3d_sample_and_inverse():
    image = as_tensor(np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7, 1))
    transform = PadIfNeeded(
        keys=["image"],
        min_target_shape=(4, 8, 8),
        divisible_by=(2, 4, 4),
        input_layout="DHWC",
    )
    forward = transform({"image": image})

    assert tuple(ops.shape(forward["image"])) == (4, 8, 8, 1)
    restored = transform.inverse(forward)
    assert tuple(ops.shape(restored["image"])) == (3, 5, 7, 1)
    np.testing.assert_array_equal(
        ops.convert_to_numpy(restored["image"]), ops.convert_to_numpy(image)
    )


@pytest.mark.unit
def test_pad_allow_missing_keys_skips_absent_data():
    image = as_tensor(np.ones((2, 2, 1), dtype=np.float32))
    transform = Pad(
        keys=["image", "label"],
        padding=1,
        input_layout="HWC",
        allow_missing_keys=True,
    )
    result = transform({"image": image})

    assert tuple(ops.shape(result["image"])) == (4, 4, 1)


@pytest.mark.unit
def test_pad_rejects_missing_keys_by_default():
    image = as_tensor(np.ones((2, 2, 1), dtype=np.float32))
    transform = Pad(keys=["image", "label"], padding=1, input_layout="HWC")

    with pytest.raises(KeyError, match="label"):
        transform({"image": image})


@pytest.mark.unit
def test_pad_rejects_reflect_padding_that_reaches_input_size():
    image = as_tensor(np.ones((2, 3, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="reflect.*input size"):
        Pad(
            keys=["image"],
            padding=((2, 0), (0, 0)),
            fill_mode="reflect",
            input_layout="HWC",
        )({"image": image})


@pytest.mark.unit
def test_pad_if_needed_rejects_mismatched_key_spatial_shapes():
    image = as_tensor(np.ones((5, 6, 1), dtype=np.float32))
    label = as_tensor(np.ones((5, 7, 1), dtype=np.int32))
    transform = PadIfNeeded(
        keys=["image", "label"],
        min_target_shape=(8, 8),
        input_layout="HWC",
    )

    with pytest.raises(ValueError, match="share a spatial shape"):
        transform({"image": image, "label": label})


@pytest.mark.unit
@pytest.mark.parametrize("fill_mode", ["reflect", "symmetric"])
def test_pad_inverse_exactly_restores_nonconstant_modes(fill_mode):
    image = as_tensor(np.arange(3 * 4, dtype=np.float32).reshape(3, 4, 1))
    transform = Pad(
        keys=["image"],
        padding=((1, 1), (1, 1)),
        fill_mode=fill_mode,
        input_layout="HWC",
    )
    forward = transform({"image": image})
    restored = transform.inverse(forward)

    np.testing.assert_array_equal(
        ops.convert_to_numpy(restored["image"]), ops.convert_to_numpy(image)
    )

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

import numpy as np
import pytest
from keras import ops

from medicai.transforms import (
    SignalFillEmpty,
    TensorBundle,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


@pytest.mark.unit
def test_signal_fill_empty_replaces_invalid_values_and_records_trace():
    image = as_tensor(np.array([[[np.nan], [np.inf]], [[-np.inf], [1.0]]], dtype=np.float32))
    out = SignalFillEmpty(keys=["image"], fill_value=0.0, input_layout="HWC")(
        TensorBundle({"image": image})
    )

    filled = ops.convert_to_numpy(out["image"])
    assert np.isfinite(filled).all()
    trace = out.get_applied_transforms()[-1]
    assert trace["name"] == "SignalFillEmpty"
    assert trace["random"] is False
    assert trace["params"]["input_layout"] == "HWC"


@pytest.mark.unit
def test_signal_fill_empty_outputs_float32_tensor():
    image = as_tensor(np.array([[[np.nan], [1.0]]], dtype=np.float64))
    out = SignalFillEmpty(keys=["image"], fill_value=2.0, input_layout="HWC")(
        TensorBundle({"image": image})
    )

    assert ops.dtype(out["image"]) == "float32"
    np.testing.assert_allclose(
        ops.convert_to_numpy(out["image"]), np.array([[[2.0], [1.0]]], dtype=np.float32)
    )


@pytest.mark.unit
def test_signal_fill_empty_supports_batch_mode():
    image_2d = as_tensor(
        np.array([[[[np.nan]], [[1.0]]], [[[np.inf]], [[-np.inf]]]], dtype=np.float32)
    )
    image_3d = as_tensor(
        np.array([[[[[np.nan]]], [[[-np.inf]]]], [[[[1.0]]], [[[np.inf]]]]], dtype=np.float32)
    )

    out_2d = SignalFillEmpty(keys=["image"], fill_value=0.0, input_layout="BHWC")(
        TensorBundle({"image": image_2d})
    )
    out_3d = SignalFillEmpty(keys=["image"], fill_value=2.0, input_layout="BDHWC")(
        TensorBundle({"image": image_3d})
    )

    assert tuple(ops.shape(out_2d["image"])) == (2, 2, 1, 1)
    assert tuple(ops.shape(out_3d["image"])) == (2, 2, 1, 1, 1)
    assert np.isfinite(ops.convert_to_numpy(out_2d["image"])).all()
    assert np.isfinite(ops.convert_to_numpy(out_3d["image"])).all()


@pytest.mark.unit
def test_signal_fill_empty_accepts_input_layout():
    image = as_tensor(
        np.array([[[[np.nan]], [[1.0]]], [[[np.inf]], [[-np.inf]]]], dtype=np.float32)
    )

    out = SignalFillEmpty(keys=["image"], fill_value=0.0, input_layout="BHWC")(
        TensorBundle({"image": image})
    )

    assert tuple(ops.shape(out["image"])) == (2, 2, 1, 1)
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "BHWC"


@pytest.mark.unit
def test_signal_fill_empty_uses_same_pixel_kernel_for_sample_and_batch_modes():
    sample = as_tensor(np.array([[[np.nan], [1.0], [np.inf]]], dtype=np.float32))
    batch = as_tensor(
        np.array(
            [
                [[[np.nan], [1.0], [np.inf]]],
                [[[2.0], [-np.inf], [3.0]]],
            ],
            dtype=np.float32,
        )
    )
    transform = SignalFillEmpty(keys=["image"], fill_value=0.0, input_layout="HWC")

    sample_filled = transform.nan_to_num_batch(sample)
    batch_filled = transform.nan_to_num_batch(batch)

    assert np.isfinite(ops.convert_to_numpy(sample_filled)).all()
    assert np.isfinite(ops.convert_to_numpy(batch_filled)).all()
    np.testing.assert_allclose(
        ops.convert_to_numpy(sample_filled)[0, 0, 0],
        0.0,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(batch_filled)[0, 0, 0, 0],
        0.0,
        rtol=1e-6,
    )



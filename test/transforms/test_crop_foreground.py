import numpy as np
import pytest
from keras import ops

from medicai.transforms import (
    CropForeground,
    TensorBundle,
)


def as_tensor(array, dtype=None):
    return ops.convert_to_tensor(np.asarray(array), dtype=dtype)


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

    out_2d = CropForeground(keys=["image"], source_key="image", input_layout="HWC")(
        TensorBundle({"image": image_2d})
    )
    out_3d = CropForeground(keys=["image"], source_key="image", input_layout="DHWC")(
        TensorBundle({"image": image_3d})
    )

    assert tuple(ops.shape(out_2d["image"])) == (2, 2, 1)
    assert tuple(ops.shape(out_3d["image"])) == (2, 2, 2, 1)
    assert out_2d.get_applied_transforms()[-1]["name"] == "CropForeground"


@pytest.mark.unit
def test_crop_foreground_accepts_input_layout_and_rejects_batch_layout():
    image = as_tensor(
        np.pad(
            np.ones((4, 4, 1), dtype=np.float32),
            ((2, 2), (2, 2), (0, 0)),
        )
    )
    out = CropForeground(keys=["image"], source_key="image", input_layout="HWC")(
        TensorBundle({"image": image})
    )

    assert tuple(ops.shape(out["image"])) == (4, 4, 1)
    assert out.get_applied_transforms()[-1]["params"]["input_layout"] == "HWC"

    with pytest.raises(ValueError, match="supports only input_layout values"):
        CropForeground(keys=["image"], source_key="image", input_layout="BHWC")


@pytest.mark.unit
def test_crop_foreground_empty_mask_returns_full_image_and_can_disable_metadata():
    image = as_tensor(np.zeros((4, 5, 1), dtype=np.float32))
    out = CropForeground(
        keys=["image"],
        source_key="image",
        select_fn=lambda x: x > 10,
        start_coord_key=None,
        end_coord_key=None,
        input_layout="HWC",
    )(TensorBundle({"image": image}))

    assert tuple(ops.shape(out["image"])) == (4, 5, 1)
    assert "foreground_start_coord" not in out.meta
    assert "foreground_end_coord" not in out.meta


@pytest.mark.unit
def test_crop_foreground_accepts_string_like_single_key_input():
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

    # Python evaluates ("image") as a plain string. The transform should
    # still normalize it to a single-key collection.
    out = CropForeground(keys=("image"), source_key="image", input_layout="HWC")(
        TensorBundle({"image": image})
    )

    assert tuple(ops.shape(out["image"])) == (2, 2, 1)


@pytest.mark.unit
def test_crop_foreground_defaults_source_key_for_single_key_input():
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

    out = CropForeground(keys=["image"], input_layout="HWC")(TensorBundle({"image": image}))

    assert tuple(ops.shape(out["image"])) == (2, 2, 1)


@pytest.mark.unit
def test_crop_foreground_requires_source_key_for_multi_key_input():
    with pytest.raises(ValueError, match="`source_key` must be provided"):
        CropForeground(keys=["image", "label"], input_layout="HWC")


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
        input_layout="HWC",
    )(TensorBundle({"image": as_tensor(image_np)}))

    shape = tuple(ops.shape(out["image"]))
    assert shape[0] % 2 == 0
    assert shape[1] % 2 == 0


@pytest.mark.unit
def test_crop_foreground_preserves_margin_when_foreground_touches_boundary():
    source = np.zeros((10, 10, 1), dtype=np.float32)
    source[:1, :1, 0] = 1.0
    transform = CropForeground(
        keys=["image"],
        source_key="image",
        margin=4,
        allow_smaller=False,
        input_layout="HWC",
    )

    result = transform(TensorBundle({"image": as_tensor(source)}))

    assert tuple(ops.shape(result["image"])) == (9, 9, 1)
    np.testing.assert_array_equal(
        ops.convert_to_numpy(result.meta["foreground_start_coord"]),
        np.array([0, 0]),
    )
    np.testing.assert_array_equal(
        ops.convert_to_numpy(result.meta["foreground_end_coord"]),
        np.array([9, 9]),
    )


@pytest.mark.unit
def test_crop_foreground_shifts_divisible_extent_away_from_image_edge():
    source = np.zeros((10, 10, 1), dtype=np.float32)
    source[8:10, 8:10, 0] = 1.0
    transform = CropForeground(
        keys=["image"],
        source_key="image",
        k_divisible=4,
        input_layout="HWC",
    )

    result = transform(TensorBundle({"image": as_tensor(source)}))

    assert tuple(ops.shape(result["image"])) == (4, 4, 1)
    np.testing.assert_array_equal(
        ops.convert_to_numpy(result.meta["foreground_start_coord"]),
        np.array([6, 6]),
    )
    np.testing.assert_array_equal(
        ops.convert_to_numpy(result.meta["foreground_end_coord"]),
        np.array([10, 10]),
    )


@pytest.mark.unit
def test_crop_foreground_inverse_restores_original_canvas_for_2d():
    image = as_tensor(np.zeros((6, 7, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[2:5, 3:6, 0] = 1.0
    image = as_tensor(image_np)

    transform = CropForeground(keys=["image"], source_key="image", input_layout="HWC")
    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(forward["image"])) == (3, 3, 1)
    assert tuple(ops.shape(restored["image"])) == (6, 7, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_crop_foreground_inverse_restores_original_canvas_for_3d():
    image = as_tensor(np.zeros((5, 6, 7, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[1:4, 2:5, 3:6, 0] = 1.0
    image = as_tensor(image_np)

    transform = CropForeground(keys=["image"], source_key="image", input_layout="DHWC")
    forward = transform(TensorBundle({"image": image}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    assert tuple(ops.shape(forward["image"])) == (3, 3, 3, 1)
    assert tuple(ops.shape(restored["image"])) == (5, 6, 7, 1)
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored["image"]),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_crop_foreground_inverse_places_prediction_back_on_original_canvas():
    image = np.zeros((6, 7, 1), dtype=np.float32)
    image[2:5, 3:6, 0] = 2.0
    label = np.zeros((6, 7, 1), dtype=np.float32)
    transform = CropForeground(keys=["image", "label"], source_key="image", input_layout="HWC")

    forward = transform(TensorBundle({"image": as_tensor(image), "label": as_tensor(label)}))
    prediction = ops.ones_like(forward["label"])
    prediction_bundle = TensorBundle(
        {"image": forward["image"], "label": prediction},
        dict(forward.meta),
    )
    prediction_bundle.meta["applied_transforms"] = list(forward.get_applied_transforms())

    restored = transform.inverse(prediction_bundle)

    expected = np.zeros((6, 7, 1), dtype=np.float32)
    expected[2:5, 3:6, 0] = 1.0
    np.testing.assert_allclose(ops.convert_to_numpy(restored["label"]), expected)


@pytest.mark.unit
def test_crop_foreground_inverse_zero_pads_discarded_context():
    image = np.arange(42, dtype=np.float32).reshape(6, 7, 1)
    source = np.zeros((6, 7, 1), dtype=np.float32)
    source[2:5, 3:6, 0] = 1.0
    transform = CropForeground(keys=["image"], source_key="source", input_layout="HWC")

    forward = transform(TensorBundle({"image": as_tensor(image), "source": as_tensor(source)}))
    restored = transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))

    restored_np = ops.convert_to_numpy(restored["image"])
    expected = np.zeros((6, 7, 1), dtype=np.float32)
    expected[2:5, 3:6, :] = image[2:5, 3:6, :]
    np.testing.assert_allclose(restored_np, expected)


@pytest.mark.unit
def test_crop_foreground_inverse_without_trace_is_noop():
    bundle = TensorBundle({"image": as_tensor(np.ones((4, 5, 1), dtype=np.float32))})
    transform = CropForeground(keys=["image"], source_key="image", input_layout="HWC")

    restored = transform.inverse(bundle)

    assert restored is bundle

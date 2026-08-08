import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from medicai.transforms import (
    Compose,
    CropForeground,
    Flip,
    LambdaTransform,
    NormalizeIntensity,
    Orientation,
    RandomChoice,
    RandomCropByPosNegLabel,
    RandomCutOut,
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
def test_intensity_transforms_run_under_tf_function():
    normalize = NormalizeIntensity(keys=["image"], nonzero=True)
    scale = ScaleIntensityRange(
        keys=["image"], input_min=0.0, input_max=1.0, output_min=-1.0, output_max=1.0
    )
    shift = ShiftIntensity(keys=["image"], offset=0.25)
    fill = SignalFillEmpty(keys=["image"], fill_value=0.0)

    image = as_tensor(np.array([[[0.0], [1.0]], [[np.nan], [0.5]]], dtype=np.float32))

    @tf.function
    def apply_transforms(x):
        out_fill = fill({"image": x})["image"]
        out_norm = normalize({"image": out_fill})["image"]
        out_scale = scale({"image": tf.clip_by_value(out_fill, 0.0, 1.0)})["image"]
        out_shift = shift({"image": out_fill})["image"]
        return out_norm, out_scale, out_shift

    normed, scaled, shifted = apply_transforms(image)

    assert tuple(ops.shape(normed)) == (2, 2, 1)
    assert tuple(ops.shape(scaled)) == (2, 2, 1)
    assert tuple(ops.shape(shifted)) == (2, 2, 1)


@pytest.mark.unit
def test_signal_fill_empty_supports_batch_mode_under_tf_function():
    fill_2d = SignalFillEmpty(keys=["image"], fill_value=0.0, input_mode="batch")
    fill_3d = SignalFillEmpty(keys=["image"], fill_value=2.0, input_mode="batch")

    image_2d = as_tensor(
        np.array([[[[np.nan]], [[1.0]]], [[[np.inf]], [[-np.inf]]]], dtype=np.float32)
    )
    image_3d = as_tensor(
        np.array([[[[[np.nan]]], [[[-np.inf]]]], [[[[1.0]]], [[[np.inf]]]]], dtype=np.float32)
    )

    @tf.function
    def apply_transforms(x2, x3):
        out_2d = fill_2d({"image": x2})
        out_3d = fill_3d({"image": x3})
        return (
            out_2d["image"],
            out_2d.get_applied_transforms()[-1]["params"]["input_mode"],
            out_3d["image"],
            out_3d.get_applied_transforms()[-1]["params"]["input_mode"],
        )

    out_2d, mode_2d, out_3d, mode_3d = apply_transforms(image_2d, image_3d)

    assert np.isfinite(ops.convert_to_numpy(out_2d)).all()
    assert np.isfinite(ops.convert_to_numpy(out_3d)).all()
    assert mode_2d == "batch"
    assert mode_3d == "batch"


@pytest.mark.unit
def test_normalize_intensity_sparse_nonzero_paths_remain_shape_stable_under_tf_function():
    normalize_global = NormalizeIntensity(keys=["image"], nonzero=True, channel_wise=False)
    normalize_channel = NormalizeIntensity(keys=["image"], nonzero=True, channel_wise=True)

    image = as_tensor(
        np.array(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[3.0, 0.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        )
    )

    @tf.function
    def apply_transforms(x):
        global_out = normalize_global({"image": x})["image"]
        channel_out = normalize_channel({"image": x})["image"]
        return global_out, channel_out

    global_out, channel_out = apply_transforms(image)

    assert tuple(ops.shape(global_out)) == (2, 2, 2)
    assert tuple(ops.shape(channel_out)) == (2, 2, 2)
    assert np.isfinite(ops.convert_to_numpy(global_out)).all()
    assert np.isfinite(ops.convert_to_numpy(channel_out)).all()
    assert ops.convert_to_numpy(global_out)[0, 0, 0] == 0.0
    assert ops.convert_to_numpy(channel_out)[0, 0, 0] == 0.0
    assert ops.convert_to_numpy(channel_out)[0, 0, 1] == 0.0


@pytest.mark.unit
def test_normalize_intensity_supports_batch_mode_under_tf_function():
    normalize_2d = NormalizeIntensity(keys=["image"], input_mode="batch")
    normalize_3d = NormalizeIntensity(keys=["image"], input_mode="batch")

    image_2d = as_tensor(np.ones((2, 3, 4, 1), dtype=np.float32))
    image_3d = as_tensor(np.ones((2, 3, 4, 5, 1), dtype=np.float32))

    @tf.function
    def apply_transforms(x2, x3):
        out_2d = normalize_2d({"image": x2})
        out_3d = normalize_3d({"image": x3})
        return (
            out_2d["image"],
            out_2d.get_applied_transforms()[-1]["params"]["input_mode"],
            out_3d["image"],
            out_3d.get_applied_transforms()[-1]["params"]["input_mode"],
        )

    out_2d, mode_2d, out_3d, mode_3d = apply_transforms(image_2d, image_3d)

    assert tuple(ops.shape(out_2d)) == (2, 3, 4, 1)
    assert tuple(ops.shape(out_3d)) == (2, 3, 4, 5, 1)
    assert np.isfinite(ops.convert_to_numpy(out_2d)).all()
    assert np.isfinite(ops.convert_to_numpy(out_3d)).all()
    assert mode_2d == "batch"
    assert mode_3d == "batch"


@pytest.mark.unit
def test_spatial_rank_agnostic_transforms_run_under_tf_function():
    crop = SpatialCrop(keys=["image"], crop_size=(3, 4), crop_start=(1, 1))
    flip = Flip(keys=["image"], spatial_axis=1)
    rotate = Rotate90(keys=["image"], k=1)
    resize = Resize(keys=["image"], interpolation="bilinear", target_shape=(4, 5))
    foreground = CropForeground(keys=["image"], source_key="image")

    image = as_tensor(
        np.array(
            [
                [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
                [[0.0], [1.0], [1.0], [1.0], [1.0], [0.0]],
                [[0.0], [1.0], [1.0], [1.0], [1.0], [0.0]],
                [[0.0], [1.0], [1.0], [1.0], [1.0], [0.0]],
                [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
            ],
            dtype=np.float32,
        )
    )

    @tf.function
    def apply_transforms(x):
        cropped = crop({"image": x})["image"]
        flipped = flip({"image": x})["image"]
        rotated = rotate({"image": x})["image"]
        resized = resize({"image": x})["image"]
        foregrounded = foreground({"image": x})["image"]
        return cropped, flipped, rotated, resized, foregrounded

    cropped, flipped, rotated, resized, foregrounded = apply_transforms(image)

    assert tuple(ops.shape(cropped)) == (3, 4, 1)
    assert tuple(ops.shape(flipped)) == (5, 6, 1)
    assert tuple(ops.shape(rotated)) == (6, 5, 1)
    assert tuple(ops.shape(resized)) == (4, 5, 1)
    assert tuple(ops.shape(foregrounded)) == (3, 4, 1)


@pytest.mark.unit
def test_spatial_crop_supports_batch_mode_under_tf_function():
    crop_2d = SpatialCrop(keys=["image"], crop_size=(3, 4), crop_start=(1, 1), input_mode="batch")
    crop_3d = SpatialCrop(
        keys=["image"],
        crop_size=(2, 3, 4),
        crop_start=(1, 1, 1),
        input_mode="batch",
    )

    image_2d = as_tensor(np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6, 1))
    image_3d = as_tensor(np.arange(2 * 4 * 5 * 6, dtype=np.float32).reshape(2, 4, 5, 6, 1))

    @tf.function
    def apply_transforms(x2, x3):
        out_2d = crop_2d({"image": x2})["image"]
        out_3d = crop_3d({"image": x3})["image"]
        return out_2d, out_3d

    out_2d, out_3d = apply_transforms(image_2d, image_3d)

    assert tuple(ops.shape(out_2d)) == (2, 3, 4, 1)
    assert tuple(ops.shape(out_3d)) == (2, 2, 3, 4, 1)


@pytest.mark.unit
def test_flip_and_rotate90_support_batch_mode_under_tf_function():
    flip_2d = Flip(keys=["image"], spatial_axis=1, input_mode="batch")
    flip_3d = Flip(keys=["image"], spatial_axis=2, input_mode="batch")
    rotate_2d = Rotate90(keys=["image"], k=1, spatial_axis=(0, 1), input_mode="batch")
    rotate_3d = Rotate90(keys=["image"], k=1, spatial_axis=(1, 2), input_mode="batch")

    image_2d = as_tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1))
    image_3d = as_tensor(np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5, 1))

    @tf.function
    def apply_transforms(x2, x3):
        out_flip_2d = flip_2d({"image": x2})["image"]
        out_flip_3d = flip_3d({"image": x3})["image"]
        out_rotate_2d = rotate_2d({"image": x2})["image"]
        out_rotate_3d = rotate_3d({"image": x3})["image"]
        return out_flip_2d, out_flip_3d, out_rotate_2d, out_rotate_3d

    out_flip_2d, out_flip_3d, out_rotate_2d, out_rotate_3d = apply_transforms(image_2d, image_3d)

    np.testing.assert_allclose(
        ops.convert_to_numpy(out_flip_2d),
        ops.convert_to_numpy(image_2d)[:, :, ::-1, :],
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(out_flip_3d),
        ops.convert_to_numpy(image_3d)[:, :, :, ::-1, :],
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(out_rotate_2d),
        np.rot90(ops.convert_to_numpy(image_2d), k=1, axes=(1, 2)),
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(out_rotate_3d),
        np.rot90(ops.convert_to_numpy(image_3d), k=1, axes=(2, 3)),
    )


@pytest.mark.unit
def test_random_flip_and_rotate90_support_batch_mode_under_tf_function():
    random_flip = RandomFlip(keys=["image"], prob=1.0, spatial_axis=1, input_mode="batch")
    random_rotate90 = RandomRotate90(
        keys=["image"],
        prob=1.0,
        max_k=3,
        spatial_axis=(0, 1),
        input_mode="batch",
    )

    image = as_tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1))

    @tf.function
    def apply_transforms(x):
        flipped = random_flip({"image": x})
        rotated = random_rotate90({"image": x})
        return (
            flipped["image"],
            flipped.get_applied_transforms()[-1]["params"]["input_mode"],
            rotated["image"],
            rotated.get_applied_transforms()[-1]["params"]["input_mode"],
        )

    flipped_image, flipped_mode, rotated_image, rotated_mode = apply_transforms(image)

    np.testing.assert_allclose(
        ops.convert_to_numpy(flipped_image),
        ops.convert_to_numpy(image)[:, :, ::-1, :],
    )
    assert flipped_mode == "batch"
    assert rotated_mode == "batch"
    assert tuple(ops.shape(rotated_image)) in {(2, 4, 3, 1), (2, 3, 4, 1)}


@pytest.mark.unit
def test_resize_runs_under_tf_function_for_3d():
    resize = Resize(
        keys=["image", "label"],
        interpolation=("trilinear", "nearest"),
        target_shape=(4, 5, 6),
    )
    image = as_tensor(np.random.randn(6, 7, 8, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (6, 7, 8, 1)).astype(np.float32))

    @tf.function
    def apply_transform(x, y):
        result = resize({"image": x, "label": y})
        return result["image"], result["label"]

    out_image, out_label = apply_transform(image, label)

    assert tuple(ops.shape(out_image)) == (4, 5, 6, 1)
    assert tuple(ops.shape(out_label)) == (4, 5, 6, 1)


@pytest.mark.unit
def test_resize_supports_batch_mode_under_tf_function():
    resize_2d = Resize(
        keys=["image"],
        interpolation="bilinear",
        target_shape=(4, 5),
        input_mode="batch",
    )
    resize_3d = Resize(
        keys=["image"],
        interpolation="trilinear",
        target_shape=(3, 4, 5),
        input_mode="batch",
    )

    image_2d = as_tensor(np.random.randn(2, 6, 7, 1).astype(np.float32))
    image_3d = as_tensor(np.random.randn(2, 5, 6, 7, 1).astype(np.float32))

    @tf.function
    def apply_transforms(x2, x3):
        out_2d = resize_2d({"image": x2})["image"]
        out_3d = resize_3d({"image": x3})["image"]
        return out_2d, out_3d

    out_2d, out_3d = apply_transforms(image_2d, image_3d)

    assert tuple(ops.shape(out_2d)) == (2, 4, 5, 1)
    assert tuple(ops.shape(out_3d)) == (2, 3, 4, 5, 1)


@pytest.mark.unit
def test_scale_intensity_range_supports_batch_mode_under_tf_function():
    scale_2d = ScaleIntensityRange(
        keys=["image"],
        input_min=0.0,
        input_max=255.0,
        output_min=0.0,
        output_max=1.0,
        input_mode="batch",
    )
    scale_3d = ScaleIntensityRange(
        keys=["image"],
        input_min=0.0,
        input_max=1.0,
        output_min=-1.0,
        output_max=1.0,
        input_mode="batch",
    )

    image_2d = as_tensor(np.full((2, 3, 4, 1), 128.0, dtype=np.float32))
    image_3d = as_tensor(np.full((2, 3, 4, 5, 1), 0.5, dtype=np.float32))

    @tf.function
    def apply_transforms(x2, x3):
        out_2d = scale_2d({"image": x2})
        out_3d = scale_3d({"image": x3})
        return (
            out_2d["image"],
            out_2d.get_applied_transforms()[-1]["params"]["input_mode"],
            out_3d["image"],
            out_3d.get_applied_transforms()[-1]["params"]["input_mode"],
        )

    out_2d, mode_2d, out_3d, mode_3d = apply_transforms(image_2d, image_3d)

    np.testing.assert_allclose(ops.convert_to_numpy(out_2d), 128.0 / 255.0, rtol=1e-6)
    np.testing.assert_allclose(ops.convert_to_numpy(out_3d), 0.0, rtol=1e-6)
    assert mode_2d == "batch"
    assert mode_3d == "batch"


@pytest.mark.unit
def test_spacing_and_orientation_run_under_tf_function():
    spacing = Spacing(keys=["image", "label"], pixdim=(0.5, 0.5, 0.5))
    orientation = Orientation(keys=["image", "label"], axcodes="RAS")

    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))
    affine = as_tensor(np.eye(4, dtype=np.float32))

    @tf.function
    def apply_transforms(x, y, a):
        spaced = spacing({"image": x, "label": y}, {"affine": a})
        oriented = orientation(
            {"image": spaced["image"], "label": spaced["label"]}, {"affine": spaced["affine"]}
        )
        return oriented["image"], oriented["label"], oriented["affine"]

    out_image, out_label, out_affine = apply_transforms(image, label, affine)

    assert tuple(ops.shape(out_image)) == (12, 10, 8, 1)
    assert tuple(ops.shape(out_label)) == (12, 10, 8, 1)
    assert tuple(ops.shape(out_affine)) == (4, 4)


@pytest.mark.unit
def test_orientation_forward_and_inverse_run_under_tf_function():
    orientation = Orientation(keys=["image", "label"], axcodes="RAS")

    image = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))
    affine = as_tensor(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )

    @tf.function
    def apply_and_inverse(x, y, a):
        forward = orientation({"image": x, "label": y}, {"affine": a})
        restored = orientation.inverse(forward)
        return restored["image"], restored["label"], restored["affine"]

    restored_image, restored_label, restored_affine = apply_and_inverse(image, label, affine)

    assert tuple(ops.shape(restored_image)) == (4, 5, 6, 1)
    assert tuple(ops.shape(restored_label)) == (4, 5, 6, 1)
    np.testing.assert_allclose(ops.convert_to_numpy(restored_affine), ops.convert_to_numpy(affine))


@pytest.mark.unit
def test_compose_crop_orientation_spacing_pipeline_runs_forward_and_inverse_under_tf_function():
    pipeline = Compose(
        [
            CropForeground(keys=["image", "label"], source_key="image"),
            Orientation(keys=["image", "label"], axcodes="RAS"),
            Spacing(
                keys=["image", "label"],
                pixdim=(1.0, 0.75, 1.5),
                interpolation=("trilinear", "nearest"),
            ),
        ]
    )

    image = np.zeros((6, 8, 10, 1), dtype=np.float32)
    image[1:5, 2:7, 3:9, 0] = 2.0
    label = np.zeros((6, 8, 10, 1), dtype=np.float32)
    label[2:4, 3:6, 4:8, 0] = 1.0
    affine = np.array(
        [
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    image = as_tensor(image)
    label = as_tensor(label)
    affine = as_tensor(affine)

    @tf.function
    def apply_and_inverse(x, y, a):
        forward = pipeline({"image": x, "label": y}, {"affine": a})
        prediction = tf.cast(forward["label"] > 0.0, forward["label"].dtype)
        bundle = TensorBundle(
            {"image": forward["image"], "label": prediction},
            dict(forward.meta),
        )
        bundle.meta["applied_transforms"] = list(forward.get_applied_transforms())
        restored = pipeline.inverse(bundle)
        return restored["image"], restored["label"], restored["affine"]

    restored_image, restored_label, restored_affine = apply_and_inverse(image, label, affine)

    assert tuple(ops.shape(restored_image)) == (6, 8, 10, 1)
    assert tuple(ops.shape(restored_label)) == (6, 8, 10, 1)
    np.testing.assert_allclose(ops.convert_to_numpy(restored_affine), ops.convert_to_numpy(affine))


@pytest.mark.unit
def test_random_choice_runs_under_tf_function_when_num_choices_is_one():
    choice = RandomChoice(
        transforms=[
            ShiftIntensity(keys=["image"], offset=1.0),
            Flip(keys=["image"], spatial_axis=1),
        ],
        num_choices=1,
        prob=1.0,
    )
    image = as_tensor(np.ones((4, 4, 1), dtype=np.float32))

    @tf.function
    def apply_transform(x):
        return choice({"image": x})["image"]

    output = apply_transform(image)
    assert tuple(ops.shape(output)) == (4, 4, 1)


@pytest.mark.unit
def test_random_choice_runs_under_tf_function_for_multi_choice():
    choice = RandomChoice(
        transforms=[
            ShiftIntensity(keys=["image"], offset=1.0),
            ShiftIntensity(keys=["image"], offset=2.0),
            ShiftIntensity(keys=["image"], offset=4.0),
        ],
        num_choices=2,
        prob=1.0,
    )
    image = as_tensor(np.ones((4, 4, 1), dtype=np.float32))

    @tf.function
    def apply_transform(x):
        return choice({"image": x})["image"]

    outputs = []
    for _ in range(10):
        outputs.append(float(ops.convert_to_numpy(apply_transform(image))[0, 0, 0]))

    assert all(value in {4.0, 6.0, 7.0} for value in outputs)
    assert len(set(outputs)) >= 2


@pytest.mark.unit
def test_random_rank_agnostic_transforms_run_under_tf_function():
    random_flip = RandomFlip(keys=["image"], prob=1.0, spatial_axis=0)
    random_rotate90 = RandomRotate90(keys=["image"], prob=1.0, max_k=3)
    random_spatial_crop = RandomSpatialCrop(keys=["image"], crop_size=(3, 4), random_center=False)
    random_shift = RandomShiftIntensity(keys=["image"], offset=0.25, prob=1.0)

    image = as_tensor(np.random.randn(5, 6, 1).astype(np.float32))

    @tf.function
    def apply_transforms(x):
        flipped = random_flip({"image": x})["image"]
        rotated = random_rotate90({"image": x})["image"]
        cropped = random_spatial_crop({"image": x})["image"]
        shifted = random_shift({"image": x})["image"]
        return flipped, rotated, cropped, shifted

    flipped, rotated, cropped, shifted = apply_transforms(image)

    assert tuple(ops.shape(flipped)) == (5, 6, 1)
    assert tuple(ops.shape(rotated)) == (6, 5, 1) or tuple(ops.shape(rotated)) == (5, 6, 1)
    assert tuple(ops.shape(cropped)) == (3, 4, 1)
    assert tuple(ops.shape(shifted)) == (5, 6, 1)


@pytest.mark.unit
def test_random_flip_and_rotate90_inverse_support_batch_mode_under_tf_function():
    random_flip = RandomFlip(keys=["image"], prob=1.0, spatial_axis=1, input_mode="batch")
    random_rotate90 = RandomRotate90(
        keys=["image"],
        prob=1.0,
        max_k=3,
        spatial_axis=(0, 1),
        input_mode="batch",
    )
    image = as_tensor(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4, 1))

    @tf.function
    def apply_and_inverse(x):
        flipped = random_flip({"image": x})
        restored_flip = random_flip.inverse(flipped)

        rotated = random_rotate90({"image": x})
        restored_rotate = random_rotate90.inverse(rotated)
        return restored_flip["image"], restored_rotate["image"]

    restored_flip, restored_rotate = apply_and_inverse(image)

    np.testing.assert_allclose(
        ops.convert_to_numpy(restored_flip),
        ops.convert_to_numpy(image),
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(restored_rotate),
        ops.convert_to_numpy(image),
    )


@pytest.mark.unit
def test_shift_intensity_supports_batch_mode_under_tf_function():
    shift_2d = ShiftIntensity(keys=["image"], offset=0.5, input_mode="batch")
    shift_3d = ShiftIntensity(keys=["image"], offset=-0.25, input_mode="batch")

    image_2d = as_tensor(np.ones((2, 3, 4, 1), dtype=np.float32))
    image_3d = as_tensor(np.ones((2, 3, 4, 5, 1), dtype=np.float32))

    @tf.function
    def apply_transforms(x2, x3):
        out_2d = shift_2d({"image": x2})
        out_3d = shift_3d({"image": x3})
        return (
            out_2d["image"],
            out_2d.get_applied_transforms()[-1]["params"]["input_mode"],
            out_3d["image"],
            out_3d.get_applied_transforms()[-1]["params"]["input_mode"],
        )

    out_2d, mode_2d, out_3d, mode_3d = apply_transforms(image_2d, image_3d)

    np.testing.assert_allclose(ops.convert_to_numpy(out_2d), 1.5, rtol=1e-6)
    np.testing.assert_allclose(ops.convert_to_numpy(out_3d), 0.75, rtol=1e-6)
    assert mode_2d == "batch"
    assert mode_3d == "batch"


@pytest.mark.unit
def test_random_spatial_crop_supports_batch_mode_under_tf_function():
    random_spatial_crop_2d = RandomSpatialCrop(
        keys=["image"],
        crop_size=(3, 4),
        random_center=False,
        input_mode="batch",
    )
    random_spatial_crop_3d = RandomSpatialCrop(
        keys=["image"],
        crop_size=(2, 3, 4),
        random_center=False,
        input_mode="batch",
    )

    image_2d = as_tensor(np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6, 1))
    image_3d = as_tensor(np.arange(2 * 4 * 5 * 6, dtype=np.float32).reshape(2, 4, 5, 6, 1))

    @tf.function
    def apply_transforms(x2, x3):
        out_2d = random_spatial_crop_2d({"image": x2})["image"]
        out_3d = random_spatial_crop_3d({"image": x3})["image"]
        return out_2d, out_3d

    out_2d, out_3d = apply_transforms(image_2d, image_3d)

    assert tuple(ops.shape(out_2d)) == (2, 3, 4, 1)
    assert tuple(ops.shape(out_3d)) == (2, 2, 3, 4, 1)


@pytest.mark.unit
def test_random_spatial_crop_forward_and_inverse_run_under_tf_function_in_batch_mode():
    random_spatial_crop = RandomSpatialCrop(
        keys=["image"],
        crop_size=(3, 4),
        random_center=False,
        input_mode="batch",
    )

    image = as_tensor(np.zeros((2, 5, 6, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[:, 1:4, 1:5, 0] = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    image = as_tensor(image_np)

    @tf.function
    def apply_transform(x):
        forward = random_spatial_crop({"image": x})
        restored = random_spatial_crop.inverse(forward)
        return restored["image"]

    restored = apply_transform(image)

    assert tuple(ops.shape(restored)) == (2, 5, 6, 1)
    np.testing.assert_allclose(ops.convert_to_numpy(restored), ops.convert_to_numpy(image))


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_runs_under_tf_function_for_2d_and_3d():
    crop_2d = RandomCropByPosNegLabel(keys=["image", "label"], target_shape=(4, 4), pos=1, neg=1)
    crop_3d = RandomCropByPosNegLabel(keys=["image", "label"], target_shape=(3, 3, 3), pos=1, neg=1)

    image_2d = as_tensor(np.random.randn(8, 8, 1).astype(np.float32))
    label_2d = as_tensor(np.pad(np.ones((2, 2, 1), dtype=np.float32), ((3, 3), (3, 3), (0, 0))))

    image_3d = as_tensor(np.random.randn(6, 6, 6, 1).astype(np.float32))
    label_3d = as_tensor(
        np.pad(np.ones((2, 2, 2, 1), dtype=np.float32), ((2, 2), (2, 2), (2, 2), (0, 0)))
    )

    @tf.function
    def apply_2d(x, y):
        result = crop_2d({"image": x, "label": y})
        return result["image"], result["label"]

    @tf.function
    def apply_3d(x, y):
        result = crop_3d({"image": x, "label": y})
        return result["image"], result["label"]

    out_2d_image, out_2d_label = apply_2d(image_2d, label_2d)
    out_3d_image, out_3d_label = apply_3d(image_3d, label_3d)

    assert tuple(ops.shape(out_2d_image)) == (4, 4, 1)
    assert tuple(ops.shape(out_2d_label)) == (4, 4, 1)
    assert tuple(ops.shape(out_3d_image)) == (3, 3, 3, 1)
    assert tuple(ops.shape(out_3d_label)) == (3, 3, 3, 1)


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_runs_under_tf_function_in_batch_mode():
    crop_2d = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(4, 4),
        pos=1,
        neg=1,
        input_mode="batch",
    )
    crop_3d = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(3, 3, 3),
        pos=1,
        neg=1,
        input_mode="batch",
    )

    image_2d = as_tensor(np.random.randn(2, 8, 8, 1).astype(np.float32))
    label_2d = as_tensor(np.zeros((2, 8, 8, 1), dtype=np.float32))
    label_2d_np = ops.convert_to_numpy(label_2d)
    label_2d_np[:, 3:5, 3:5, 0] = 1.0
    label_2d = as_tensor(label_2d_np)

    image_3d = as_tensor(np.random.randn(2, 6, 6, 6, 1).astype(np.float32))
    label_3d = as_tensor(np.zeros((2, 6, 6, 6, 1), dtype=np.float32))
    label_3d_np = ops.convert_to_numpy(label_3d)
    label_3d_np[:, 2:4, 2:4, 2:4, 0] = 1.0
    label_3d = as_tensor(label_3d_np)

    @tf.function
    def apply_2d(x, y):
        result = crop_2d({"image": x, "label": y})
        return result["image"], result["label"], result.get_applied_transforms()[-1]["params"]["input_mode"]

    @tf.function
    def apply_3d(x, y):
        result = crop_3d({"image": x, "label": y})
        return result["image"], result["label"], result.get_applied_transforms()[-1]["params"]["input_mode"]

    out_2d_image, out_2d_label, out_2d_mode = apply_2d(image_2d, label_2d)
    out_3d_image, out_3d_label, out_3d_mode = apply_3d(image_3d, label_3d)

    assert tuple(ops.shape(out_2d_image)) == (2, 4, 4, 1)
    assert tuple(ops.shape(out_2d_label)) == (2, 4, 4, 1)
    assert tuple(ops.shape(out_3d_image)) == (2, 3, 3, 3, 1)
    assert tuple(ops.shape(out_3d_label)) == (2, 3, 3, 3, 1)
    assert out_2d_mode == "batch"
    assert out_3d_mode == "batch"


@pytest.mark.unit
def test_random_crop_by_pos_neg_label_forward_and_inverse_run_under_tf_function_in_batch_mode():
    crop = RandomCropByPosNegLabel(
        keys=["image", "label"],
        target_shape=(3, 3),
        pos=1,
        neg=0,
        input_mode="batch",
    )

    image = as_tensor(np.zeros((2, 6, 6, 1), dtype=np.float32))
    image_np = ops.convert_to_numpy(image)
    image_np[:, 2:5, 2:5, 0] = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    image = as_tensor(image_np)

    label = as_tensor(np.zeros((2, 6, 6, 1), dtype=np.float32))
    label_np = ops.convert_to_numpy(label)
    label_np[:, 3, 3, 0] = 1.0
    label = as_tensor(label_np)

    @tf.function
    def apply_transform(x, y):
        forward = crop({"image": x, "label": y})
        restored = crop.inverse(forward)
        return restored["image"], restored["label"]

    restored_image, restored_label = apply_transform(image, label)

    assert tuple(ops.shape(restored_image)) == (2, 6, 6, 1)
    assert tuple(ops.shape(restored_label)) == (2, 6, 6, 1)
    np.testing.assert_allclose(ops.convert_to_numpy(restored_image), ops.convert_to_numpy(image))
    np.testing.assert_allclose(ops.convert_to_numpy(restored_label), ops.convert_to_numpy(label))


@pytest.mark.unit
def test_random_rotate_and_cutout_run_under_tf_function():
    random_rotate = RandomRotate(keys=["image", "label"], factor=0.2, prob=1.0)
    random_cutout_2d = RandomCutOut(keys=["image", "label"], mask_size=(2, 2), num_cuts=1, prob=1.0)
    random_cutout_slicewise = RandomCutOut(
        keys=["image", "label"], mask_size=(2, 2), num_cuts=1, prob=1.0
    )

    image_3d = as_tensor(np.random.randn(4, 5, 6, 1).astype(np.float32))
    label_3d = as_tensor(np.random.randint(0, 2, (4, 5, 6, 1)).astype(np.float32))

    image_2d = as_tensor(np.random.randn(8, 8, 1).astype(np.float32))
    label_2d = as_tensor(np.random.randint(0, 2, (8, 8, 1)).astype(np.float32))

    @tf.function
    def apply_rotate(x, y):
        result = random_rotate({"image": x, "label": y})
        return result["image"], result["label"]

    @tf.function
    def apply_cutout_2d(x, y):
        result = random_cutout_2d({"image": x, "label": y})
        return result["image"]

    @tf.function
    def apply_cutout_3d(x, y):
        result = random_cutout_slicewise({"image": x, "label": y})
        return result["image"]

    rotated_image, rotated_label = apply_rotate(image_3d, label_3d)
    cutout_2d = apply_cutout_2d(image_2d, label_2d)
    cutout_3d = apply_cutout_3d(image_3d, label_3d)

    assert tuple(ops.shape(rotated_image)) == (4, 5, 6, 1)
    assert tuple(ops.shape(rotated_label)) == (4, 5, 6, 1)
    assert tuple(ops.shape(cutout_2d)) == (8, 8, 1)
    assert tuple(ops.shape(cutout_3d)) == (4, 5, 6, 1)


@pytest.mark.unit
def test_random_rotate_supports_batch_mode_under_tf_function():
    random_rotate = RandomRotate(keys=["image", "label"], factor=0.2, prob=1.0, input_mode="batch")

    image = as_tensor(np.random.randn(2, 4, 5, 6, 1).astype(np.float32))
    label = as_tensor(np.random.randint(0, 2, (2, 4, 5, 6, 1)).astype(np.float32))

    @tf.function
    def apply_rotate(x, y):
        result = random_rotate({"image": x, "label": y})
        return result["image"], result["label"]

    rotated_image, rotated_label = apply_rotate(image, label)

    assert tuple(ops.shape(rotated_image)) == (2, 4, 5, 6, 1)
    assert tuple(ops.shape(rotated_label)) == (2, 4, 5, 6, 1)


@pytest.mark.unit
def test_random_cutout_supports_batch_mode_under_tf_function():
    random_cutout_2d = RandomCutOut(
        keys=["image", "label"],
        mask_size=(2, 2),
        num_cuts=1,
        prob=1.0,
        input_mode="batch",
        seed=13,
    )
    random_cutout_3d = RandomCutOut(
        keys=["image", "label"],
        mask_size=(2, 2),
        num_cuts=1,
        prob=1.0,
        input_mode="batch",
        seed=13,
    )

    image_2d = as_tensor(np.ones((2, 8, 8, 1), dtype=np.float32))
    label_2d = as_tensor(np.ones((2, 8, 8, 1), dtype=np.float32))
    image_3d = as_tensor(np.ones((2, 4, 5, 6, 1), dtype=np.float32))
    label_3d = as_tensor(np.ones((2, 4, 5, 6, 1), dtype=np.float32))

    @tf.function
    def apply_cutout_2d(x, y):
        result = random_cutout_2d({"image": x, "label": y})
        return result["image"], result.get_applied_transforms()[-1]["params"]["input_mode"]

    @tf.function
    def apply_cutout_3d(x, y):
        result = random_cutout_3d({"image": x, "label": y})
        return result["image"], result.get_applied_transforms()[-1]["params"]["input_mode"]

    cutout_2d, mode_2d = apply_cutout_2d(image_2d, label_2d)
    cutout_3d, mode_3d = apply_cutout_3d(image_3d, label_3d)

    assert tuple(ops.shape(cutout_2d)) == (2, 8, 8, 1)
    assert tuple(ops.shape(cutout_3d)) == (2, 4, 5, 6, 1)
    assert mode_2d == "batch"
    assert mode_3d == "batch"


@pytest.mark.unit
def test_lambda_transform_and_compose_run_under_tf_function():
    lambda_transform = LambdaTransform(
        keys=["image"],
        fn=lambda tensor: tensor + 2.0,
        inverse_fn=lambda tensor: tensor - 2.0,
        meta_fn=lambda meta: {**meta, "lambda_forward": True},
    )
    pipeline = Compose(
        [
            LambdaTransform(keys=["image"], fn=lambda tensor: tensor * 3.0, name="triple"),
            LambdaTransform(keys=["image"], fn=lambda tensor: tensor + 1.0, name="plus_one"),
        ]
    )

    image = as_tensor(np.ones((4, 4, 1), dtype=np.float32))

    @tf.function
    def apply_lambda(x):
        forward = lambda_transform({"image": x})
        restored = lambda_transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))
        composed = pipeline({"image": x})["image"]
        return forward["image"], restored["image"], composed

    forward, restored, composed = apply_lambda(image)

    np.testing.assert_allclose(ops.convert_to_numpy(forward), 3.0)
    np.testing.assert_allclose(ops.convert_to_numpy(restored), 1.0)
    np.testing.assert_allclose(ops.convert_to_numpy(composed), 4.0)


@pytest.mark.unit
def test_lambda_transform_probabilistic_inverse_runs_under_tf_function():
    lambda_transform = LambdaTransform(
        keys=["image"],
        fn=lambda tensor: tensor + 2.0,
        inverse_fn=lambda tensor: tensor - 2.0,
        prob=1.0,
    )

    image = as_tensor(np.ones((4, 4, 1), dtype=np.float32))

    @tf.function
    def apply_lambda(x):
        forward = lambda_transform({"image": x})
        restored = lambda_transform.inverse(TensorBundle({"image": forward["image"]}, forward.meta))
        return forward["image"], restored["image"]

    forward, restored = apply_lambda(image)

    np.testing.assert_allclose(ops.convert_to_numpy(forward), 3.0)
    np.testing.assert_allclose(ops.convert_to_numpy(restored), 1.0)

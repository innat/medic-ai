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

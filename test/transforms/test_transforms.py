import tensorflow as tf

from medicai.transforms import (
    Compose,
    CropForeground,
    Orientation,
    RandCropByPosNegLabel,
    RandRotate90,
    RandShiftIntensity,
    Resize,
    ScaleIntensityRange,
    Spacing,
    TensorBundle,
)


def test_resize_transform():
    # Create dummy data
    image = tf.random.normal((1, 128, 128, 1))
    label = tf.random.uniform((1, 128, 128, 1), maxval=4, dtype=tf.int32)
    inputs = TensorBundle({"image": image, "label": label})

    # check 2D resize
    resize_transform = Resize(keys=["image", "label"], spatial_shape=(96, 96))
    resized_2d = resize_transform(inputs)
    assert resized_2d.data["image"].shape == (1, 96, 96, 1)
    assert resized_2d.data["label"].shape == (1, 96, 96, 1)

    # check 3D resize
    image_3d = tf.random.normal((128, 128, 128, 1))
    label_3d = tf.random.uniform((128, 128, 128, 1), maxval=4, dtype=tf.int32)
    inputs_3d = TensorBundle({"image": image_3d, "label": label_3d})
    resize_transform = Resize(keys=["image", "label"], spatial_shape=(64, 96, 96))
    resized_3d = resize_transform(inputs)
    assert resized_3d.data["image"].shape == (64, 96, 96, 1)
    assert resized_3d.data["label"].shape == (64, 96, 96, 1)

    # check only one key (image)
    image_only = tf.random.normal((128, 128, 128, 1))
    inputs_image_only = TensorBundle({"image": image_only})
    resize_transform = Resize(keys=["image"], spatial_shape=(64, 96, 64))
    resized_image_only = resize_transform(inputs)
    assert resized_image_only.data["image"].shape == (64, 96, 64, 1)


def test_scale_intensity():
    image = tf.constant([[[5.0, 5.0], [5.0, 5.0]]], dtype=tf.float32)
    inputs = TensorBundle({"image": image})
    scale = ScaleIntensityRange(keys=["image"], a_min=5.0, a_max=5.0, b_min=0.0, b_max=1.0)
    output = scale(inputs)
    expected = tf.constant([[[0.0, 0.0], [0.0, 0.0]]], dtype=tf.float32)
    assert tf.reduce_all(tf.equal(output.data["image"], expected))


def test_rand_shift_intensity():
    image = tf.constant([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=tf.float32)
    inputs = TensorBundle({"image": image})
    shift = RandShiftIntensity(keys=["image"], offsets=(-0.2, 0.8), prob=1.0)
    output = shift(inputs)
    assert output.data["image"].shape == (1, 2, 2, 1)
    shifted_image = output.data["image"]
    original_image = inputs.data["image"]
    assert tf.reduce_all(shifted_image >= original_image - 0.8)
    assert tf.reduce_all(shifted_image <= original_image + 0.8)


def test_rand_rotate90():
    image = tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=tf.float32)
    inputs = TensorBundle({"image": image})
    rotate = RandRotate90(keys=["image"], prob=1.0, max_k=3)
    output = rotate(inputs)
    assert output.data["image"].shape == (1, 2, 2, 2)


def test_rand_crop_by_pos_neg_label():
    image = tf.constant(
        [[[[0.1], [0.5], [0.9]], [[0.2], [0.6], [0.8]], [[0.3], [0.7], [0.4]]]], dtype=tf.float32
    )
    label = tf.constant([[[[0], [1], [0]], [[1], [0], [1]], [[0], [1], [0]]]], dtype=tf.float32)
    inputs = TensorBundle({"image": image, "label": label})
    crop = RandCropByPosNegLabel(
        keys=["image", "label"], spatial_size=(2, 2, 2), pos=1, neg=1, num_samples=1
    )
    output = crop(inputs)
    assert output.data["image"].shape == (1, 2, 2, 1)
    assert output.data["label"].shape == (1, 2, 2, 1)


def test_crop_foreground():
    image = tf.constant([[[[0], [1], [0]], [[1], [1], [1]], [[0], [0], [1]]]], dtype=tf.float32)
    label = tf.constant([[[[0], [2], [0]], [[2], [2], [2]], [[0], [0], [2]]]], dtype=tf.int32)
    inputs = TensorBundle({"image": image, "label": label})
    crop = CropForeground(keys=("image", "label"), source_key="image")
    output = crop(inputs)
    assert output.data["image"].shape == (1, 3, 3, 1)
    assert output.data["label"].shape == (1, 3, 3, 1)


def test_tensor_bundle():
    image = tf.random.normal((10, 10, 10, 1))
    label = tf.random.uniform((10, 10, 10, 1), maxval=5, dtype=tf.int32)
    affine = tf.constant(
        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=tf.float32
    )
    meta = {"affine": affine, "pixdim": [1.0, 1.0, 1.0]}
    inputs = TensorBundle({"image": image, "label": label}, meta)

    # Test get_data
    assert tf.reduce_all(inputs.get_data("image") == image)
    assert tf.reduce_all(inputs.get_data("label") == label)
    assert inputs.get_data("affine") is None #affine is in meta.

    # Test get_meta
    assert tf.reduce_all(inputs.get_meta("affine") == affine)
    assert inputs.get_meta("pixdim") == [1.0, 1.0, 1.0]
    assert inputs.get_meta('image') is None #image is in data.

    # Test __getitem__
    assert tf.reduce_all(inputs["image"] == image)
    assert tf.reduce_all(inputs["label"] == label)
    assert tf.reduce_all(inputs["affine"] == affine)
    assert inputs["pixdim"] == [1.0, 1.0, 1.0]

    # Test set_data
    new_image = tf.random.normal((5, 5, 5, 1))
    inputs.set_data("image", new_image)
    assert tf.reduce_all(inputs.get_data("image") == new_image)

    # Test set_meta
    new_affine = tf.eye(4)
    inputs.set_meta("affine", new_affine)
    assert tf.reduce_all(inputs.get_meta("affine") == new_affine)


def test_compose():
    image = tf.random.normal((10, 10, 10, 1))
    label = tf.random.uniform((10, 10, 10, 1), maxval=5, dtype=tf.int32)
    affine = tf.constant(
        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=tf.float32
    )
    meta = {"affine": affine, "pixdim": [1.0, 1.0, 1.0]}
    inputs = {"image": image, "label": label}

    orientation_transform = Orientation(keys=["image", "label"], axcodes="RAS")
    spacing_transform = Spacing(keys=["image", "label"], pixdim=[0.5, 0.5, 0.5])

    composed_transform = Compose([spacing_transform, orientation_transform])
    output = composed_transform(inputs, meta)

    assert output.data["image"].shape == (20, 20, 20, 1)
    assert output.data["label"].shape == (20, 20, 20, 1)


def test_orientation_ras_image_label():
    image = tf.random.normal((10, 10, 10, 1))
    label = tf.random.uniform((10, 10, 10, 1), maxval=5, dtype=tf.int32)
    affine = tf.constant([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=tf.float32)
    meta = {"affine": affine}
    inputs = TensorBundle({"image": image, "label": label}, meta)
    transform = Orientation(keys=["image", "label"], axcodes="RAS")
    output = transform(inputs)
    assert tf.reduce_all(tf.equal(output.data["image"], image))
    assert tf.reduce_all(tf.equal(output.data["label"], label))


def test_spacing_upsample_image_label():
    image = tf.random.normal((10, 10, 10, 1))
    segmentation = tf.random.uniform((10, 10, 10, 1), maxval=5, dtype=tf.int32)
    meta = {"pixdim": [1.0, 1.0, 1.0]}
    inputs = TensorBundle({"my_image": image, "my_segmentation": segmentation}, meta)
    transform = Spacing(keys=["my_image", "my_segmentation"], pixdim=[2.0, 2.0, 2.0])
    output = transform(inputs)
    assert output.data["my_image"].shape == (5, 5, 5, 1)
    assert output.data["my_segmentation"].shape == (5, 5, 5, 1)

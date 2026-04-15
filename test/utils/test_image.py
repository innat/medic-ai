import pytest
import tensorflow as tf

from medicai.utils.image import resize_volumes


@pytest.mark.unit
def test_resize_volumes_trilinear_shape_and_dtype():
    volumes = tf.random.uniform((2, 4, 6, 8, 1), dtype=tf.float32)
    out = resize_volumes(volumes, depth=2, height=3, width=4, method="trilinear")

    assert out.shape == (2, 2, 3, 4, 1)
    assert out.dtype == volumes.dtype


@pytest.mark.unit
def test_resize_volumes_nearest_shape_and_dtype():
    volumes = tf.random.uniform((1, 5, 5, 5, 2), maxval=10, dtype=tf.int32)
    out = resize_volumes(volumes, depth=3, height=4, width=2, method="nearest")

    assert out.shape == (1, 3, 4, 2, 2)
    assert out.dtype == volumes.dtype


@pytest.mark.unit
def test_resize_volumes_unsupported_method_raises():
    volumes = tf.ones((1, 2, 2, 2, 1), dtype=tf.float32)
    with pytest.raises(ValueError, match="Unsupported resize method"):
        resize_volumes(volumes, depth=2, height=2, width=2, method="bicubic")

import tensorflow as tf

from medicai.transforms import TensorBundle


def test_tensorbundle_creation():
    data = {
        "image": tf.random.normal((10, 10, 10, 1)),
        "label": tf.random.uniform((10, 10, 10, 1), maxval=5, dtype=tf.int32),
    }
    meta = {
        "affine": tf.constant(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=tf.float32
        ),
        "pixdim": [1.0, 1.0, 1.0],
    }
    mt = TensorBundle(data, meta)

    assert mt.data == data
    assert mt.meta == meta
    assert all(k in mt.meta for k in meta.keys())


def test_tensorbundle_getitem_data():
    data = {"image": tf.random.normal((10, 10, 10, 1))}
    mt = TensorBundle(data)
    assert tf.reduce_all(tf.equal(mt["image"], data["image"]))


def test_tensorbundle_getitem_meta():
    meta = {
        "affine": tf.constant(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=tf.float32
        )
    }
    mt = TensorBundle({}, meta)
    assert tf.reduce_all(tf.equal(mt["affine"], meta["affine"]))


def test_tensorbundle_setitem_data():
    data = {"image": tf.random.normal((10, 10, 10, 1))}
    mt = TensorBundle(data)
    new_image = tf.random.normal((5, 5, 5, 1))
    mt["image"] = new_image
    assert tf.reduce_all(tf.equal(mt["image"], new_image))


def test_tensorbundle_setitem_meta():
    mt = TensorBundle({})
    mt["affine"] = tf.constant(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=tf.float32
    )
    assert tf.reduce_all(
        tf.equal(
            mt["affine"],
            tf.constant([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=tf.float32),
        )
    )

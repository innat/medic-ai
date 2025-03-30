import tensorflow as tf

from medicai.models import SwinTransformer, SwinUNETR


def test_swin_unetr():
    num_classes = 4
    input_shape = (96, 96, 96, 1)
    model = SwinUNETR(input_shape=input_shape, num_classes=num_classes)
    assert model.input_shape == (None, 96, 96, 96, 1)
    dummy_input = tf.random.normal((1, 96, 96, 96, 1))
    output = model(dummy_input)
    assert output.shape == (1, 96, 96, 96, num_classes)


def test_swin_transformer():
    num_classes = 4
    input_shape = (96, 96, 96, 1)
    model = SwinTransformer(input_shape=input_shape, num_classes=num_classes)
    assert model.input_shape == (None, 96, 96, 96, 1)
    dummy_input = tf.random.normal((1, 96, 96, 96, 1))
    output = model(dummy_input)
    assert output.shape == (1, num_classes)

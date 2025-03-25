import tensorflow as tf

from src.medicai.models import SwinUNETR


def test_swin_unetr():
    num_classes = 4
    input_shape = (96, 96, 96, 1)
    model = SwinUNETR(input_shape=input_shape, out_channels=num_classes)
    assert model.input_shape == (None, 96, 96, 96, 1)
    dummy_input = tf.random.normal((1, 96, 96, 96, 1))
    output = model(dummy_input)
    assert output.shape == (1, 96, 96, 96, num_classes)

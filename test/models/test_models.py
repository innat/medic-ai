import tensorflow as tf

from medicai.models import (
    UNETR,
    DenseNet121,
    DenseUNet121,
    SegFormer,
    SwinTransformer,
    SwinUNETR,
    TransUNet,
    ViT,
)


def test_unet():
    num_classes = 1
    input_shape = (64, 64, 64, 1)
    model = DenseUNet121(input_shape=input_shape, num_classes=num_classes)
    dummy_input = tf.random.normal((1, 64, 64, 64, 1))
    output = model(dummy_input)
    assert output.shape == (1, 64, 64, 64, num_classes)

    input_shape = (64, 64, 1)
    model = DenseUNet121(input_shape=input_shape, num_classes=num_classes)
    dummy_input = tf.random.normal((1, 64, 64, 1))
    output = model(dummy_input)
    assert output.shape == (1, 64, 64, num_classes)


def test_densenet():
    num_classes = 1
    input_shape = (64, 64, 64, 1)
    model = DenseNet121(input_shape=input_shape, num_classes=num_classes)
    dummy_input = tf.random.normal((1, 64, 64, 64, 1))
    output = model(dummy_input)
    assert output.shape == (1, num_classes)

    input_shape = (64, 64, 1)
    model = DenseNet121(input_shape=input_shape, num_classes=num_classes)
    dummy_input = tf.random.normal((1, 64, 64, 1))
    output = model(dummy_input)
    assert output.shape == (1, num_classes)


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


def test_unetr():
    batch_size = 1
    D, H, W, C = 96, 96, 96, 1
    num_classes = 3

    # test for 3D
    dummy_input = tf.random.normal((batch_size, D, H, W, C))
    model = UNETR(input_shape=(D, H, W, C), num_classes=num_classes)
    output = model(dummy_input)
    assert model.input_shape == (None, 96, 96, 96, 1)
    assert output.shape == (batch_size, D, H, W, 3)

    # test for 2D
    dummy_input = tf.random.normal((batch_size, H, W, C))
    model = UNETR(input_shape=(H, W, C), num_classes=num_classes)
    output = model(dummy_input)
    assert model.input_shape == (None, 96, 96, 1)
    assert output.shape == (batch_size, H, W, 3)


def test_vit():
    batch_size = 4
    D, H, W, C = 16, 32, 32, 1
    num_classes = 10

    vit2d = ViT(
        input_shape=(H, W, 3),
        num_classes=num_classes,
        pooling="token",
        intermediate_dim=128,
        classifier_activation="softmax",
        dropout=0.1,
    )
    x2d = tf.random.normal((batch_size, H, W, 3))
    y2d = vit2d(x2d)
    assert y2d.shape == (batch_size, num_classes)

    vit3d = ViT(
        input_shape=(D, H, W, C),  # D, H, W, C
        num_classes=num_classes,
        pooling="gap",
        intermediate_dim=None,
        classifier_activation=None,
        dropout=0.1,
        name="Vit3D",
    )
    x3d = tf.random.normal((batch_size, D, H, W, C))
    y3d = vit3d(x3d)
    assert y3d.shape == (batch_size, num_classes)


def test_segformer():
    batch_size = 1
    D, H, W, C = 96, 96, 96, 1
    num_classes = 3

    # test for 3D
    dummy_input = tf.random.normal((batch_size, D, H, W, C))
    model = SegFormer(input_shape=(D, H, W, C), num_classes=num_classes)
    output = model(dummy_input)
    assert model.input_shape == (None, 96, 96, 96, 1)
    assert output.shape == (batch_size, D, H, W, 3)

    # test for 2D
    dummy_input = tf.random.normal((batch_size, H, W, C))
    model = SegFormer(input_shape=(H, W, C), num_classes=num_classes)
    output = model(dummy_input)
    assert model.input_shape == (None, 96, 96, 1)
    assert output.shape == (batch_size, H, W, 3)


def test_transunet():
    batch_size = 1
    D, H, W, C = 96, 96, 96, 1
    num_classes = 3

    # test for 3D
    dummy_input = tf.random.normal((batch_size, D, H, W, C))
    model = TransUNet(input_shape=(D, H, W, C), num_classes=num_classes)
    output = model(dummy_input)
    assert model.input_shape == (None, 96, 96, 96, 1)
    assert output.shape == (batch_size, D, H, W, 3)

    # test for 2D
    dummy_input = tf.random.normal((batch_size, H, W, C))
    model = TransUNet(input_shape=(H, W, C), num_classes=num_classes)
    output = model(dummy_input)
    assert model.input_shape == (None, 96, 96, 1)
    assert output.shape == (batch_size, H, W, 3)

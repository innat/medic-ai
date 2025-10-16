import pytest
import tensorflow as tf

from medicai.models import (
    UNETR,
    ConvNeXtV2Atto,
    DenseNet121,
    EfficientNetB0,
    SegFormer,
    SwinTiny,
    SwinTinyV2,
    SwinUNETR,
    TransUNet,
    UNet,
    UNetPlusPlus,
    ViTBase,
)


def test_unet():
    num_classes = 1
    input_shape = (64, 64, 64, 1)
    model = UNet(input_shape=input_shape, num_classes=num_classes, encoder_name="densenet121")
    dummy_input = tf.random.normal((1, 64, 64, 64, 1))
    output = model(dummy_input)
    assert output.shape == (1, 64, 64, 64, num_classes)

    input_shape = (64, 64, 1)
    model = UNet(input_shape=input_shape, num_classes=num_classes, encoder_name="densenet121")
    dummy_input = tf.random.normal((1, 64, 64, 1))
    output = model(dummy_input)
    assert output.shape == (1, 64, 64, num_classes)


@pytest.mark.parametrize(
    "input_shape",
    [
        (64, 64, 64, 1),
        (64, 64, 1),
    ],
)
def test_unet_pp(input_shape):
    num_classes = 1
    model = UNetPlusPlus(
        input_shape=input_shape, num_classes=num_classes, encoder_name="efficientnet_b0"
    )
    dummy_input = tf.random.normal((1, *input_shape))
    output = model(dummy_input)
    expected_shape = (1, *input_shape[:-1], num_classes)
    assert output.shape == expected_shape


@pytest.mark.parametrize(
    "input_shape",
    [
        (64, 64, 64, 1),
        (64, 64, 1),
    ],
)
def test_convnext(input_shape):
    num_classes = 1
    model = ConvNeXtV2Atto(input_shape=input_shape, num_classes=num_classes)
    dummy_input = tf.random.normal((1, *input_shape))
    output = model(dummy_input)
    expected_shape = (1, *input_shape[:-1], num_classes)
    assert output.shape == expected_shape


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


def test_efficientnet():
    num_classes = 1
    input_shape = (64, 64, 64, 1)
    model = EfficientNetB0(input_shape=input_shape, num_classes=num_classes)
    dummy_input = tf.random.normal((1, 64, 64, 64, 1))
    output = model(dummy_input)
    assert output.shape == (1, num_classes)

    input_shape = (64, 64, 1)
    model = EfficientNetB0(input_shape=input_shape, num_classes=num_classes)
    dummy_input = tf.random.normal((1, 64, 64, 1))
    output = model(dummy_input)
    assert output.shape == (1, num_classes)


def test_swin_unetr():
    # test for 3D
    num_classes = 4
    input_shape = (96, 96, 96, 1)
    model = SwinUNETR(input_shape=input_shape, num_classes=num_classes, encoder_name="swin_tiny")
    assert model.input_shape == (None, 96, 96, 96, 1)
    dummy_input = tf.random.normal((1, 96, 96, 96, 1))
    output = model(dummy_input)
    assert output.shape == (1, 96, 96, 96, num_classes)

    # test for 2D
    num_classes = 4
    input_shape = (96, 96, 1)
    model = SwinUNETR(input_shape=input_shape, num_classes=num_classes, encoder_name="swin_tiny_v2")
    assert model.input_shape == (None, 96, 96, 1)
    dummy_input = tf.random.normal((1, 96, 96, 1))
    output = model(dummy_input)
    assert output.shape == (1, 96, 96, num_classes)


def test_swin_transformer():
    # test for 3D
    num_classes = 4
    input_shape = (96, 96, 96, 1)
    model = SwinTinyV2(input_shape=input_shape, num_classes=num_classes)
    assert model.input_shape == (None, 96, 96, 96, 1)
    dummy_input = tf.random.normal((1, 96, 96, 96, 1))
    output = model(dummy_input)
    assert output.shape == (1, num_classes)

    # test for 2D
    num_classes = 4
    input_shape = (96, 96, 1)
    model = SwinTiny(input_shape=input_shape, num_classes=num_classes)
    assert model.input_shape == (None, 96, 96, 1)
    dummy_input = tf.random.normal((1, 96, 96, 1))
    output = model(dummy_input)
    assert output.shape == (1, num_classes)


def test_unetr():
    batch_size = 1
    D, H, W, C = 96, 96, 96, 1
    num_classes = 3

    # test for 3D
    dummy_input = tf.random.normal((batch_size, D, H, W, C))
    model = UNETR(input_shape=(D, H, W, C), num_classes=num_classes, encoder_name="vit_base")
    output = model(dummy_input)
    assert model.input_shape == (None, 96, 96, 96, 1)
    assert output.shape == (batch_size, D, H, W, 3)

    # test for 2D
    dummy_input = tf.random.normal((batch_size, H, W, C))
    model = UNETR(input_shape=(H, W, C), num_classes=num_classes, encoder_name="vit_base")
    output = model(dummy_input)
    assert model.input_shape == (None, 96, 96, 1)
    assert output.shape == (batch_size, H, W, 3)


def test_vit():
    batch_size = 4
    D, H, W, C = 16, 32, 32, 1
    num_classes = 10

    vit2d = ViTBase(
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

    vit3d = ViTBase(
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
    model = SegFormer(input_shape=(D, H, W, C), num_classes=num_classes, encoder_name="mit_b0")
    output = model(dummy_input)
    assert model.input_shape == (None, 96, 96, 96, 1)
    assert output.shape == (batch_size, D, H, W, 3)

    # test for 2D
    dummy_input = tf.random.normal((batch_size, H, W, C))
    model = SegFormer(input_shape=(H, W, C), num_classes=num_classes, encoder_name="mit_b0")
    output = model(dummy_input)
    assert model.input_shape == (None, 96, 96, 1)
    assert output.shape == (batch_size, H, W, 3)


def test_transunet():
    batch_size = 1
    D, H, W, C = 96, 96, 96, 1
    num_classes = 3
    patch_size = 3

    # test for 3D
    dummy_input = tf.random.normal((batch_size, D, H, W, C))
    model = TransUNet(
        input_shape=(D, H, W, C),
        encoder_name="densenet121",
        num_classes=num_classes,
        patch_size=patch_size,
    )
    output = model(dummy_input)
    assert model.input_shape == (None, 96, 96, 96, 1)
    assert output.shape == (batch_size, D, H, W, 3)

    # test for 2D
    dummy_input = tf.random.normal((batch_size, H, W, C))
    model = TransUNet(
        input_shape=(H, W, C),
        encoder_name="densenet121",
        num_classes=num_classes,
        patch_size=patch_size,
    )
    output = model(dummy_input)
    assert model.input_shape == (None, 96, 96, 1)
    assert output.shape == (batch_size, H, W, 3)

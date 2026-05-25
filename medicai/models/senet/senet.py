import keras

from medicai.utils import DescribeMixin, registration

from ..resnet import (
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet50v2,
    ResNet50vd,
    ResNet101,
    ResNet101v2,
    ResNet152,
    ResNet152v2,
    ResNet200vd,
    ResNeXt50,
    ResNeXt101,
)


class SEInitMixin:
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("se_block", True)
        kwargs.setdefault("se_ratio", 16)
        kwargs.setdefault("se_activation", "relu")
        super().__init__(*args, **kwargs)


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNet18(SEInitMixin, ResNet18, DescribeMixin):
    """SE-ResNet18: ResNet18 backbone with Squeeze-and-Excitation blocks."""

    pass


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNet34(SEInitMixin, ResNet34, DescribeMixin):
    """SE-ResNet34: ResNet34 backbone with Squeeze-and-Excitation blocks."""

    pass


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNet50(SEInitMixin, ResNet50, DescribeMixin):
    """SE-ResNet50: ResNet50 backbone with Squeeze-and-Excitation blocks."""

    pass


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(name="seresnet50_v2", family="senet")
class SEResNet50v2(SEInitMixin, ResNet50v2, DescribeMixin):
    """SE-ResNet50v2: ResNet50v2 backbone with Squeeze-and-Excitation blocks."""

    pass


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(name="seresnet50_vd", family="senet")
class SEResNet50vd(SEInitMixin, ResNet50vd, DescribeMixin):
    """SE-ResNet50vd: ResNet50vd backbone with Squeeze-and-Excitation blocks."""

    pass


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNet101(SEInitMixin, ResNet101, DescribeMixin):
    """SE-ResNet101: ResNet101 backbone with Squeeze-and-Excitation blocks."""

    pass


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(name="seresnet101_v2", family="senet")
class SEResNet101v2(SEInitMixin, ResNet101v2, DescribeMixin):
    """SE-ResNet101v2: ResNet101v2 backbone with Squeeze-and-Excitation blocks."""

    pass


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNet152(SEInitMixin, ResNet152, DescribeMixin):
    """SE-ResNet152: ResNet152 backbone with Squeeze-and-Excitation blocks."""

    pass


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(name="seresnet152_v2", family="senet")
class SEResNet152v2(SEInitMixin, ResNet152v2, DescribeMixin):
    """SE-ResNet152v2: ResNet152v2 backbone with Squeeze-and-Excitation blocks."""

    pass


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(name="seresnet200_vd", family="senet")
class SEResNet200vd(SEInitMixin, ResNet200vd, DescribeMixin):
    """SE-ResNet200vd: ResNet200vd backbone with Squeeze-and-Excitation blocks."""

    pass


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNeXt50(SEInitMixin, ResNeXt50, DescribeMixin):
    """SE-ResNeXt50: ResNeXt50 backbone with Squeeze-and-Excitation blocks."""

    pass


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNeXt101(SEInitMixin, ResNeXt101, DescribeMixin):
    """SE-ResNeXt101: ResNeXt101 backbone with Squeeze-and-Excitation blocks."""

    pass


SENET_DOCSTRING = """
This class provides a complete {name} model, including the convolutional
backbone and the classification head (the "top"). The backbone follows the
corresponding ResNet-family variant but augments its residual blocks with
**Squeeze-and-Excitation** (SE) channel recalibration.

Args:
    input_shape: A tuple specifying the input shape of the model,
        not including the batch size. Can be `(height, width, channels)` for
        2D or `(depth, height, width, channels)` for 3D.
    include_rescaling: A boolean indicating whether to include a
        ``Rescaling`` layer at the beginning of the model.
    include_top: A boolean indicating whether to include the fully
        connected classification layer at the top of the network. If
        `False`, the model's output will be the features from the
        backbone, without the final classifier.
    num_classes: An integer specifying the number of classes for the
        classification layer. This is only relevant if `include_top`
        is `True`.
    pooling: (Optional) A string specifying the type of pooling to
        apply to the output of the backbone. Can be `"avg"` for global
        average pooling or `"max"` for global max pooling. This is only
        relevant if `include_top` is `False`.
    classifier_activation: A string specifying the activation function
        to use for the classification layer.
    name: (Optional) The name of the model.
    **kwargs: Additional keyword arguments.

Returns:
    A ``keras.Model`` whose output depends on the configuration:

      - If ``include_top=True``, the output is a classification tensor of
        shape ``(batch_size, num_classes)``.
      - If ``include_top=False`` and ``pooling`` is ``None``, the output is
        the final backbone feature tensor.
      - If ``include_top=False`` and ``pooling`` is ``"avg"`` or ``"max"``,
        the output is a pooled feature tensor.

Examples:
    PyTorch backend 2D feature extractor::

        import torch
        from medicai.models.senet import {name}

        model = {name}(
            input_shape=(224, 224, 3), include_top=True, num_classes=2
        )
        x = torch.randn((1, 224, 224, 3))
        y = model(x)
        print(y.shape) # torch.Size([1, 2])

References:
    - Squeeze-and-Excitation Networks. CVPR 2018. `arXiv:1709.01507 <https://arxiv.org/abs/1709.01507>`_
"""

SEResNet18.__doc__ = SENET_DOCSTRING.format(name="SEResNet18")
SEResNet34.__doc__ = SENET_DOCSTRING.format(name="SEResNet34")
SEResNet50.__doc__ = SENET_DOCSTRING.format(name="SEResNet50")
SEResNet50v2.__doc__ = SENET_DOCSTRING.format(name="SEResNet50v2")
SEResNet50vd.__doc__ = SENET_DOCSTRING.format(name="SEResNet50vd")
SEResNet101.__doc__ = SENET_DOCSTRING.format(name="SEResNet101")
SEResNet101v2.__doc__ = SENET_DOCSTRING.format(name="SEResNet101v2")
SEResNet152.__doc__ = SENET_DOCSTRING.format(name="SEResNet152")
SEResNet152v2.__doc__ = SENET_DOCSTRING.format(name="SEResNet152v2")
SEResNet200vd.__doc__ = SENET_DOCSTRING.format(name="SEResNet200vd")
SEResNeXt50.__doc__ = SENET_DOCSTRING.format(name="SEResNeXt50")
SEResNeXt101.__doc__ = SENET_DOCSTRING.format(name="SEResNeXt101")

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


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNet18(ResNet18, DescribeMixin):
    """SE-ResNet18: ResNet18 backbone with Squeeze-and-Excitation blocks."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("se_block", True)
        kwargs.setdefault("se_ratio", 16)
        kwargs.setdefault("se_activation", "relu")
        super().__init__(*args, **kwargs)


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNet34(ResNet34, DescribeMixin):
    """SE-ResNet34: ResNet34 backbone with Squeeze-and-Excitation blocks."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("se_block", True)
        kwargs.setdefault("se_ratio", 16)
        kwargs.setdefault("se_activation", "relu")
        super().__init__(*args, **kwargs)


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNet50(ResNet50, DescribeMixin):
    """SE-ResNet50: ResNet50 backbone with Squeeze-and-Excitation blocks."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("se_block", True)
        kwargs.setdefault("se_ratio", 16)
        kwargs.setdefault("se_activation", "relu")
        super().__init__(*args, **kwargs)


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNet50v2(ResNet50v2, DescribeMixin):
    """SE-ResNet50v2: ResNet50v2 backbone with Squeeze-and-Excitation blocks."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("se_block", True)
        kwargs.setdefault("se_ratio", 16)
        kwargs.setdefault("se_activation", "relu")
        super().__init__(*args, **kwargs)


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNet50vd(ResNet50vd, DescribeMixin):
    """SE-ResNet50vd: ResNet50vd backbone with Squeeze-and-Excitation blocks."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("se_block", True)
        kwargs.setdefault("se_ratio", 16)
        kwargs.setdefault("se_activation", "relu")
        super().__init__(*args, **kwargs)


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNet101(ResNet101, DescribeMixin):
    """SE-ResNet101: ResNet101 backbone with Squeeze-and-Excitation blocks."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("se_block", True)
        kwargs.setdefault("se_ratio", 16)
        kwargs.setdefault("se_activation", "relu")
        super().__init__(*args, **kwargs)


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNet101v2(ResNet101v2, DescribeMixin):
    """SE-ResNet101v2: ResNet101v2 backbone with Squeeze-and-Excitation blocks."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("se_block", True)
        kwargs.setdefault("se_ratio", 16)
        kwargs.setdefault("se_activation", "relu")
        super().__init__(*args, **kwargs)


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNet152(ResNet152, DescribeMixin):
    """SE-ResNet152: ResNet152 backbone with Squeeze-and-Excitation blocks."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("se_block", True)
        kwargs.setdefault("se_ratio", 16)
        kwargs.setdefault("se_activation", "relu")
        super().__init__(*args, **kwargs)


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNet152v2(ResNet152v2, DescribeMixin):
    """SE-ResNet152v2: ResNet152v2 backbone with Squeeze-and-Excitation blocks."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("se_block", True)
        kwargs.setdefault("se_ratio", 16)
        kwargs.setdefault("se_activation", "relu")
        super().__init__(*args, **kwargs)


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNet200vd(ResNet200vd, DescribeMixin):
    """SE-ResNet200vd: ResNet200vd backbone with Squeeze-and-Excitation blocks."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("se_block", True)
        kwargs.setdefault("se_ratio", 16)
        kwargs.setdefault("se_activation", "relu")
        super().__init__(*args, **kwargs)


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNeXt50(ResNeXt50, DescribeMixin):
    """SE-ResNeXt50: ResNeXt50 backbone with Squeeze-and-Excitation blocks."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("se_block", True)
        kwargs.setdefault("se_ratio", 16)
        kwargs.setdefault("se_activation", "relu")
        super().__init__(*args, **kwargs)


@keras.saving.register_keras_serializable(package="seresnet")
@registration.register(family="senet")
class SEResNeXt101(ResNeXt101, DescribeMixin):
    """SE-ResNeXt101: ResNeXt101 backbone with Squeeze-and-Excitation blocks."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("se_block", True)
        kwargs.setdefault("se_ratio", 16)
        kwargs.setdefault("se_activation", "relu")
        super().__init__(*args, **kwargs)

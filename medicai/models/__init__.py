from medicai.utils import registration

from .convnext import (
    ConvNeXtBackbone,
    ConvNeXtBackboneV2,
    ConvNeXtBase,
    ConvNeXtLarge,
    ConvNeXtSmall,
    ConvNeXtTiny,
    ConvNeXtV2Atto,
    ConvNeXtV2Base,
    ConvNeXtV2Femto,
    ConvNeXtV2Huge,
    ConvNeXtV2Large,
    ConvNeXtV2Nano,
    ConvNeXtV2Pico,
    ConvNeXtV2Small,
    ConvNeXtV2Tiny,
    ConvNeXtXLarge,
)
from .deeplabv3plus import DeepLabV3Plus
from .densenet import DenseNet121, DenseNet169, DenseNet201, DenseNetBackbone
from .efficientnet import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
    EfficientNetB8,
    EfficientNetBackbone,
    EfficientNetBackboneV2,
    EfficientNetL2,
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
    EfficientNetV2L,
    EfficientNetV2M,
    EfficientNetV2S,
)
from .mit import (
    MiTBackbone,
    MixViTB0,
    MixViTB1,
    MixViTB2,
    MixViTB3,
    MixViTB4,
    MixViTB5,
)
from .resnet import (
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
    ResNetBackbone,
    ResNeXt50,
    ResNeXt101,
)
from .segformer import SegFormer
from .senet import (
    SEResNet18,
    SEResNet34,
    SEResNet50,
    SEResNet50v2,
    SEResNet50vd,
    SEResNet101,
    SEResNet101v2,
    SEResNet152,
    SEResNet152v2,
    SEResNet200vd,
    SEResNeXt50,
    SEResNeXt101,
)
from .swin import (
    SwinBackbone,
    SwinBackboneV2,
    SwinBase,
    SwinBaseV2,
    SwinSmall,
    SwinSmallV2,
    SwinTiny,
    SwinTinyV2,
    SwinUNETR,
)
from .transunet import TransUNet
from .unet import AttentionUNet, UNet
from .unet_plus_plus import UNetPlusPlus
from .unetr import UNETR
from .vit import ViTBackbone, ViTBase, ViTHuge, ViTLarge
from .xception import XceptionBackbone


def list_models(family: str = None):
    return registration.list(family)


def create_model(name: str, **kwargs):
    return registration.create(name, **kwargs)

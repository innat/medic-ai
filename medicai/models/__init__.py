from medicai.utils import registration

from ..utils.model_utils import BACKBONE_ZOO
from .densenet import DenseNet121, DenseNet169, DenseNet201, DenseNetBackbone
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
)
from .segformer import SegFormer
from .swin import SwinBase, SwinSmall, SwinTiny, SwinUNETR
from .transunet import TransUNet
from .unet import UNet
from .unetr import UNETR
from .vit import ViTBackbone, ViTBase, ViTHuge, ViTLarge


def list_models(family: str = None):
    return registration.list(family)


def create_model(name: str):
    return registration.create(name)

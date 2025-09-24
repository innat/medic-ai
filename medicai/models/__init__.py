from medicai.utils import registration

from ..utils.model_utils import BACKBONE_ZOO
from .densenet import DenseNet121, DenseNet169, DenseNet201, DenseNetBackbone
from .mit import (
    MiTBackbone,
    MixVisionTransformerB0,
    MixVisionTransformerB1,
    MixVisionTransformerB2,
    MixVisionTransformerB3,
    MixVisionTransformerB4,
    MixVisionTransformerB5,
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
from .swin import SwinTransformer, SwinUNETR
from .transunet import TransUNet
from .unet import UNet
from .unetr import UNETR
from .vit import ViT, ViTBackbone


def list_models(family: str = None):
    return registration.list(family)

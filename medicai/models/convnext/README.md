# ConvNeXt

The **ConvNeXt** architecture modernizes standard CNNs supporting in both 2D and 3D tasks.

```python
from medicai.models import ConvNeXtTiny, ConvNeXtV2Atto

# Build 2D model with convnext_tiny
model = ConvNeXtTiny(
    input_shape=(224, 224, 3),
    num_classes=5,
    classifier_activation='sigmoid'
)

# Build 3D model with ConvNeXtV2Atto
model = ConvNeXtV2Atto(
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation=None
)
```

# Feature Pyramid Output

Models expose intermediate feature vectors via the `pyramid_outputs` attribute for downstream tasks (segmentation, detection).

```python
from medicai.models import ConvNeXtTiny

model = ConvNeXtTiny(
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation=None
)
model.pyramid_outputs
{
    'P1': <KerasTensor shape=(None, 24, 24, 24, 96), # 1/4 scale
    'P2': <KerasTensor shape=(None, 12, 12, 12, 192), # 1/8 scale
    'P3': <KerasTensor shape=(None, 6, 6, 6, 384), # 1/16 scale
    'P4': <KerasTensor shape=(None, 3, 3, 3, 768), # 1/32 scale
}
```

Notice that, the **ConvNeXt** model gives four feature vectors are scale down by `[1/4, 1/8, 1/16, 1/32]`. This is the expected hierarchical downscaling.

# Segmentation Model

**ConvNeXt** variants are used as encoders for segmentation models like `UNet`, `AttentionUNet`, and `UNet++`. 

```python
from medicai.models import UNet

unet_convnext = UNet(
    encoder_name='convnext_tiny',
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation='sigmoid',
    encoder_depth=4,
    head_upsample=2, # Restores spatial output resolution
)

unet_convnext.output 
# <KerasTensor shape=(None, 96, 96, 96, 3), dtype=float32>

unet_convnext.count_params() / 1e6
# 43.644483
```

The available encoder name or variants can be found by as follows and supported segmentation architect.

```python
import medicai
medicai.models.list_models(family='convnext')

                  Model Registry Catalog
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Segmentor        ┃ Backbone Family ┃ Variants           ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ • attention_unet │ convnext        │ • convnext_base    │
│ • unet           │                 │ • convnext_large   │
│ • unet_plus_plus │                 │ • convnext_small   │
│                  │                 │ • convnext_tiny    │
│                  │                 │ • convnext_xlarge  │
│                  │                 │ • convnextv2_atto  │
│                  │                 │ • convnextv2_base  │
│                  │                 │ • convnextv2_femto │
│                  │                 │ • convnextv2_huge  │
│                  │                 │ • convnextv2_large │
│                  │                 │ • convnextv2_nano  │
│                  │                 │ • convnextv2_pico  │
│                  │                 │ • convnextv2_small │
│                  │                 │ • convnextv2_tiny  │
└──────────────────┴─────────────────┴────────────────────┘
```
# EfficientNet

The **EfficientNet** architecture modernizes standard CNNs supporting in both 2D and 3D tasks.

```python
from medicai.models import EfficientNetB0

# Build 2D model
model = EfficientNetB0(
    input_shape=(224, 224, 3),
    num_classes=5,
    classifier_activation='sigmoid'
)

# Build 3D model
model = EfficientNetB0(
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation=None
)
```

# Feature Pyramid Output

Models expose intermediate feature vectors via the `pyramid_outputs` attribute for downstream tasks (segmentation, detection).

```python
from medicai.models import EfficientNetB0

model = EfficientNetB0(
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation=None
)
model.pyramid_outputs
{
    'P1': <KerasTensor shape=(None, 48, 48, 48, 64), 
    'P2': <KerasTensor shape=(None, 24, 24, 24, 256), 
    'P3': <KerasTensor shape=(None, 12, 12, 12, 512), 
    'P4': <KerasTensor shape=(None, 6, 6, 6, 1024), 
    'P5': <KerasTensor shape=(None, 3, 3, 3, 1024),
}
```


# Segmentation Model

**EfficientNet** variants are used as encoders for segmentation models like `UNet`, `AttentionUNet`, `UNet++`. 

```python
from medicai.models import AttentionUNet

attn_unet = AttentionUNet(
    encoder_name='efficientnet_b8',
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation='sigmoid',
)

attn_unet.output 
# <KerasTensor shape=(None, 96, 96, 96, 3), dtype=float32>

attn_unet.count_params() / 1e6
# 117.199165
```

By default, segmentation models take all features (`P1-P5`). But using `encoder_depth`, we can reduce the size of the model.

```python
from medicai.models import AttentionUNet

attn_unet = AttentionUNet(
    encoder_name='efficientnet_b8',
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    encoder_depth=4,
    classifier_activation='sigmoid',
)

attn_unet.output 
# <KerasTensor shape=(None, 96, 96, 96, 3), dtype=float32>

attn_unet.count_params() / 1e6
# 25.23412
```

The available encoder name or variants can be found by as follows and supported segmentation architect.

```python
import medicai
medicai.models.list_models(family='efficientnet')

                   Model Registry Catalog
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Segmentor        ┃ Backbone Family ┃ Variants            ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ • attention_unet │ efficientnet    │ • efficientnet_b0   │
│ • unet           │                 │ • efficientnet_b1   │
│ • unet_plus_plus │                 │ • efficientnet_b2   │
│                  │                 │ • efficientnet_b3   │
│                  │                 │ • efficientnet_b4   │
│                  │                 │ • efficientnet_b5   │
│                  │                 │ • efficientnet_b6   │
│                  │                 │ • efficientnet_b7   │
│                  │                 │ • efficientnet_b8   │
│                  │                 │ • efficientnet_l2   │
│                  │                 │ • efficientnetv2_b0 │
│                  │                 │ • efficientnetv2_b1 │
│                  │                 │ • efficientnetv2_b2 │
│                  │                 │ • efficientnetv2_b3 │
│                  │                 │ • efficientnetv2_l  │
│                  │                 │ • efficientnetv2_m  │
│                  │                 │ • efficientnetv2_s  │
└──────────────────┴─────────────────┴─────────────────────┘
```
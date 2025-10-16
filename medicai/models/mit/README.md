# Mix Vision Transformer (MixViT or MiT)

The **MiT** architecture modernizes standard CNNs supporting in both 2D and 3D tasks.

```python
from medicai.models import MixViTB0

# Build 2D model
model = MixViTB0(
    input_shape=(224, 224, 3),
    num_classes=5,
    classifier_activation='sigmoid'
)

# Build 3D model
model = MixViTB0(
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation=None
)
```

# Feature Pyramid Output

Models expose intermediate feature vectors via the `pyramid_outputs` attribute for downstream tasks (segmentation, detection).

```python
from medicai.models import MixViTB0

model = MixViTB0(
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

Notice that, the **MiT** model gives four feature vectors are scale down by `[1/4, 1/8, 1/16, 1/32]`. This is the expected hierarchical downscaling.

# Segmentation Model

**MiT** variants are used as encoders for segmentation models like `SegFormer`. 

```python
from medicai.models import SegFormer

segformer = SegFormer(
    encoder_name='mit_b0',
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation='sigmoid',
)

segformer.output 
# <KerasTensor shape=(None, 96, 96, 96, 3), dtype=float32>

segformer.count_params() / 1e6
# 4.534211
```


The available encoder name or variants can be found by as follows and supported segmentation architect.

```python
import medicai
medicai.models.list_models(family='mit')

             Model Registry Catalog
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Segmentor       ┃ Backbone Family ┃ Variants ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ • segformer     │ mit             │ • mit_b0 │
│                 │                 │ • mit_b1 │
│                 │                 │ • mit_b2 │
│                 │                 │ • mit_b3 │
│                 │                 │ • mit_b4 │
│                 │                 │ • mit_b5 │
└─────────────────┴─────────────────┴──────────┘
```
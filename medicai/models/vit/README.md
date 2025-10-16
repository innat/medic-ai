# Vision Transformer (ViT)

The **ViT** architecture modernizes standard CNNs supporting in both 2D and 3D tasks.

```python
from medicai.models import ViTBase, ViTHuge, ViTLarge

# Build 2D model
model = ViTBase(
    input_shape=(224, 224, 3),
    num_classes=5,
    classifier_activation='sigmoid'
)

# Build 3D model
model = ViTBase(
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation=None
)
```

# Feature Pyramid Output

Models expose intermediate feature vectors via the `pyramid_outputs` attribute for downstream tasks (segmentation, detection).

```python
from medicai.models import ViTBase

model = ViTBase(
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    num_layers=12,
    classifier_activation=None
)
model.pyramid_outputs
{
    'P1': <KerasTensor shape=(None, 217, 768)>, 
    'P2': <KerasTensor shape=(None, 217, 768)>, 
    'P3': <KerasTensor shape=(None, 217, 768)>, 
    'P4': <KerasTensor shape=(None, 217, 768)>, 
    'P5': <KerasTensor shape=(None, 217, 768)>, 
    'P6': <KerasTensor shape=(None, 217, 768)>, 
    'P7': <KerasTensor shape=(None, 217, 768)>, 
    'P8': <KerasTensor shape=(None, 217, 768)>, 
    'P9': <KerasTensor shape=(None, 217, 768)>, 
    'P10': <KerasTensor shape=(None, 217, 768)>, 
    'P11': <KerasTensor shape=(None, 217, 768)>, 
    'P12': <KerasTensor shape=(None, 217, 768)>, 
    'P13': <KerasTensor shape=(None, 217, 768)>}
```

Here, `P1` refers to the output of patch embedding, and rest are from transformer blocks.


# Segmentation Model

**ViT** variants are used as encoders for segmentation models like `UNETR`. 

```python
from medicai.models import UNETR

model = UNETR(
    encoder_name='vit_base',
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation='sigmoid',
)

model.output 
# <KerasTensor shape=(None, 96, 96, 96, 3), dtype=float32>

model.count_params() / 1e6
# 102.250387
```


The available encoder name or variants can be found by as follows and supported segmentation architect.

```python
import medicai
medicai.models.list_models(family='vit')

              Model Registry Catalog
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Segmentor       ┃ Backbone Family ┃ Variants    ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ • unetr         │ vit             │ • vit_base  │
│                 │                 │ • vit_huge  │
│                 │                 │ • vit_large │
└─────────────────┴─────────────────┴─────────────
```
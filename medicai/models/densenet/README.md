# DenseNet

The **DenseNet** architecture modernizes standard CNNs supporting in both 2D and 3D tasks.

```python
from medicai.models import DenseNet121

# Build 2D model
model = DenseNet121(
    input_shape=(224, 224, 3),
    num_classes=5,
    classifier_activation='sigmoid'
)

# Build 3D model
model = DenseNet121(
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation=None
)
```

# Feature Pyramid Output

Models expose intermediate feature vectors via the `pyramid_outputs` attribute for downstream tasks (segmentation, detection).

```python
from medicai.models import DenseNet121

model = DenseNet121(
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

**DenseNet** variants are used as encoders for segmentation models like `UNet`, `AttentionUNet`, `UNet++`, and `TransUNet`. 

```python
from medicai.models import UNetPlusPlus

unetpp_densenet = UNetPlusPlus(
    encoder_name='densenet121',
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation='sigmoid',
)

unetpp_densenet.output 
# <KerasTensor shape=(None, 96, 96, 96, 3), dtype=float32>

unetpp_densenet.count_params() / 1e6
# 48.532611
```

By default, segmentation models take all features (`P1-P5`). But using `encoder_depth`, we can reduce the size of the model.

```python
from medicai.models import UNetPlusPlus

unetpp_densenet = UNetPlusPlus(
    encoder_name='densenet121',
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    encoder_depth=3,
    classifier_activation='sigmoid',
)

unetpp_densenet.output 
# <KerasTensor shape=(None, 96, 96, 96, 3), dtype=float32>

unetpp_densenet.count_params() / 1e6
# 14.667139
```

The available encoder name or variants can be found by as follows and supported segmentation architect.

```python
import medicai
medicai.models.list_models(family='densenet')

                Model Registry Catalog
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Segmentor        ┃ Backbone Family ┃ Variants      ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ • attention_unet │ densenet        │ • densenet121 │
│ • trans_unet     │                 │ • densenet169 │
│ • unet           │                 │ • densenet201 │
│ • unet_plus_plus │                 │               │
└──────────────────┴─────────────────┴───────────────┘
```
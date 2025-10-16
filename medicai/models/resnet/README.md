# ResNet V1 and V2

The **ResNet** architecture modernizes standard CNNs supporting in both 2D and 3D tasks.

```python
from medicai.models import ResNet18

# Build 2D model
model = ResNet18(
    input_shape=(224, 224, 3),
    num_classes=5,
    classifier_activation='sigmoid'
)

# Build 3D model
model = ResNet18(
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation=None
)
```

# Feature Pyramid Output

Models expose intermediate feature vectors via the `pyramid_outputs` attribute for downstream tasks (segmentation, detection).

```python
from medicai.models import ResNet18

model = ResNet18(
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

**ResNet** variants are used as encoders for segmentation models like `UNet`, `UNet++`, `AttentionUNet`, and `TransUNet`. 

```python
from medicai.models import UNet

res_unet = UNet(
    encoder_name='resnet18',
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation='sigmoid',
)

res_unet.output 
# <KerasTensor shape=(None, 96, 96, 96, 3), dtype=float32>

res_unet.count_params() / 1e6
# 42.688179
```

By default, segmentation models take all features (`P1-P5`). But using `encoder_depth`, we can reduce the size of the model.

```python
from medicai.models import UNet

res_unet = UNet(
    encoder_name='resnet18',
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    encoder_depth=4,
    classifier_activation='sigmoid',
)

res_unet.output 
# <KerasTensor shape=(None, 96, 96, 96, 3), dtype=float32>

res_unet.count_params() / 1e6
# 14.822755
```

The available encoder name or variants can be found by as follows and supported segmentation architect.

```python
import medicai
medicai.models.list_models(family='resnet')

                Model Registry Catalog
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Segmentor        ┃ Backbone Family ┃ Variants      ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ • attention_unet │ resnet          │ • resnet101   │
│ • trans_unet     │                 │ • resnet101v2 │
│ • unet           │                 │ • resnet152   │
│ • unet_plus_plus │                 │ • resnet152v2 │
│                  │                 │ • resnet18    │
│                  │                 │ • resnet200vd │
│                  │                 │ • resnet34    │
│                  │                 │ • resnet50    │
│                  │                 │ • resnet50v2  │
│                  │                 │ • resnet50vd  │
└──────────────────┴─────────────────┴───────────────
```
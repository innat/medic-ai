# Xception

**Xception** (Extreme Inception) is a deep convolutional neural network architecture
that replaces the standard Inception modules with depthwise separable convolutions.
This design improves model efficiency by decoupling spatial and channel-wise
feature extraction, reducing the number of parameters while maintaining high
performance.

```python
from medicai.models import Xception

# Build 2D model
model = Xception(
    input_shape=(224, 224, 3),
    num_classes=5,
    classifier_activation='sigmoid'
)

# Build 3D model
model = Xception(
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation=None
)
```

# Feature Pyramid Output

**Xception** models expose intermediate feature vectors via the `pyramid_outputs` attribute for downstream tasks (segmentation, detection).

```python
from medicai.models import Xception

model = Xception(
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

**Xception** can be used as encoders for segmentation models like `UNet`, `UNet++`, `AttentionUNet`, and `TransUNet`. 

```python
import medicai
medicai.models.list_models(family='xception')

              Model Registry Catalog
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Segmentor        ┃ Backbone Family ┃ Variants   ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ • attention_unet │ xception        │ • xception │
│ • trans_unet     │                 │            │
│ • unet           │                 │            │
│ • unet_plus_plus │                 │            │
└──────────────────┴─────────────────┴────────────┘
```

Let's see how we can build the segmentation model with **Xception**.

```python
from medicai.models import UNet

# Build 2D segmentation model.
model = UNet(
    encoder_name='xception', input_shape=(224, 224, 3)
)

# Build 3D segmentation model.
model = UNet(
    encoder_name='xception', input_shape=(96, 96, 96, 3)
)
```

By default, segmentation models take all features (`P1-P5`). But using `encoder_depth`, we can reduce the size of the model.

```python
from medicai.models import UNet

# Build 2D segmentation model.
model = UNet(
    encoder_name='xception', input_shape=(224, 224, 3), encoder_depth=4,
)
```
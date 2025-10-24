# Squeeze-and-Excitation ResNet | ResNeXt

In **ResNet** or **ResNeXt**, after the residual convolutional path and before the addition with the identity (skip) branch, **Squeeze-and-Excitation** block is inserted. It supports both 2D and 3D input.

## SE-ResNet

```python
from medicai.models import SEResNet18

# Build 2D model
model = SEResNet18(
    input_shape=(224, 224, 3),
    num_classes=5,
    classifier_activation='sigmoid'
)

# Build 3D model
model = SEResNet18(
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation=None
)
```

## SE-ResNeXt 

Along with the **Squeeze-and-Excitation** block, the **SE-ResNeXt** architecture extends **ResNet** with **grouped convolutions** and follows the **ResNeXt** architecture described in [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431).

```python
from medicai.models import SEResNeXt50

# Build 2D model
# Initializes the SEResNeXt50-50 32x4 (seresnext50_32x4d) model
model = SEResNeXt50(
    input_shape=(224, 224, 3),
    num_classes=100,
    classifier_activation='sigmoid'
)

# Build 2D model
# Initializes the ResNeXt-101 32x8 (seresnext101_32x8d) model
model = SEResNeXt50(
    input_shape=(224, 224, 3),
    num_classes=100,
    classifier_activation='sigmoid'
)
```

**Custom**

We can also create custom more complex model. Let's build **ResNeXt** with `32` cardinality and `4` bottleneck width with **ResNet101** config along with **Squeeze-and-Excitation** block.

```python
from medicai.models import ResNetBackbone

# build `resnext101_64x4d`.
backbone = ResNetBackbone(
    input_shape=(96, 96, 96, 4),
    num_blocks=[3, 4, 23, 3],
    block_type="bottleneck_block",

    # resnext-config
    groups=64,
    width_per_group=4,

    # se-config
    se_block=False,
    se_ratio=16,
    se_activation="relu",
)
backbone.count_params() / 1e6

# build classification model.
model = keras.Sequential(
    [
        backbone,
        keras.layers.GlobalAveragePooling3D(),
        keras.layers.Dense(1, activation='sigmoid')
    ]
)
```

# Feature Pyramid Output

These models expose intermediate feature vectors via the `pyramid_outputs` attribute for downstream tasks (segmentation, detection).

```python
from medicai.models import SEResNeXt50

model = SEResNeXt50(
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

All **SE-Net** variants can be used as encoders for segmentation models like `UNet`, `UNet++`, `AttentionUNet`, and `TransUNet`. 

```python
from medicai.models import UNet

unet = UNet(
    encoder_name='seresnext50',
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation='sigmoid',
)
unet.output 
# <KerasTensor shape=(None, 96, 96, 96, 3), dtype=float32>
```

By default, segmentation models take all features (`P1-P5`). But using `encoder_depth`, we can reduce the size of the model.

```python
from medicai.models import UNet

res_unet = UNet(
    encoder_name='seresnext50',
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    encoder_depth=4,
    classifier_activation='sigmoid',
)

res_unet.output 
# <KerasTensor shape=(None, 96, 96, 96, 3), dtype=float32>
```

The available `encoder_name` or variants can be found by as follows and supported segmentation architect.

```python
import medicai
medicai.models.list_models(family='senet')

                 Model Registry Catalog
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Segmentor        ┃ Backbone Family ┃ Variants        ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ • attention_unet │ senet           │ • seresnet101   │
│ • trans_unet     │                 │ • seresnet101v2 │
│ • unet           │                 │ • seresnet152   │
│ • unet_plus_plus │                 │ • seresnet152v2 │
│                  │                 │ • seresnet18    │
│                  │                 │ • seresnet200vd │
│                  │                 │ • seresnet34    │
│                  │                 │ • seresnet50    │
│                  │                 │ • seresnet50v2  │
│                  │                 │ • seresnet50vd  │
│                  │                 │ • seresnext101  │
│                  │                 │ • seresnext50   │
└──────────────────┴─────────────────┴─────────────────┘
```
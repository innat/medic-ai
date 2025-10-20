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

# ResNeXt 

The **ResNeXt** architecture, a standard CNNs supporting in both 2D and 3D tasks. This model extends **ResNet** with **grouped convolutions** and follows the **ResNeXt** architecture described in [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431).

```python
from medicai.models import ResNeXt50, ResNeXt101, ResNetBackbone

# Build 2D model
# Initializes the ResNeXt-50 32x4 (resnext50_32x4d) model
model = ResNeXt50(
    input_shape=(224, 224, 3),
    num_classes=100,
    classifier_activation='sigmoid'
)

# Build 2D model
# Initializes the ResNeXt-101 32x8 (resnext101_32x8d) model
model = ResNeXt101(
    input_shape=(224, 224, 3),
    num_classes=100,
    classifier_activation='sigmoid'
)
```

We can also create custom more complex model. Let's build **ResNeXt** with `32` cardinality and `4` bottleneck width with **ResNet101** config.

```python
# build `resnext101_64x4d`.
backbone = ResNetBackbone(
    input_shape=(96, 96, 96, 4),
    num_blocks=[3, 4, 23, 3],
    block_type="bottleneck_block",
    groups=64,
    width_per_group=4
)
backbone.count_params() / 1e6 # 92M

# build classification model.
model = keras.Sequential(
    [
        backbone,
        keras.layers.GlobalAveragePooling3D(),
        keras.layers.Dense(1, activation='sigmoid')
    ]
)
model.summary()
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ResNetBackbone (ResNetBackbone)      │ (None, 3, 3, 3, 2048)       │      92,359,680 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling3d             │ (None, 2048)                │               0 │
│ (GlobalAveragePooling3D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 1)                   │           2,049 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 92,361,729 (352.33 MB)
 Trainable params: 92,158,849 (351.56 MB)
 Non-trainable params: 202,880 (792.50 KB)
```

# Feature Pyramid Output

Models expose intermediate feature vectors via the `pyramid_outputs` attribute for downstream tasks (segmentation, detection).

```python
from medicai.models import ResNet18, ResNeXt50

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
└──────────────────┴─────────────────┴───────────────┘
```
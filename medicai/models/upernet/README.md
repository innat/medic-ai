# UPerNet: Unified Perceptual Parsing Network

<img width="904" height="534" alt="image" src="https://github.com/user-attachments/assets/e26fdea4-6e60-4db9-a2ae-d32a8e5eaac0" />

**UPerNet** ([Unified Perceptual Parsing Network](https://arxiv.org/abs/1807.10221)) is a robust, general-purpose deep learning architecture for dense prediction tasks such as semantic segmentation, scene parsing, and medical image segmentation. This implementation supports both 2D and 3D variants of UPerNet, with multiple pre-built encoders including `EfficientNet`, `Swin Transformer`, `ConvNeXt`, and many more. The **UPerNet** combines two key modules:

1. **Pyramid Pooling Module (PPM):** Extracts multi-scale contextual information from the deepest feature map (e.g., $P5$) using adaptive pooling at multiple scales.
2. **Feature Pyramid Network (FPN):** Implements a **top-down fusion pathway** that progressively merges semantic-rich deep features with high-resolution shallow features (e.g., $P4$, $P3$, $P2$, $P1$).

## Build Model

You can easily instantiate a **UPerNet** model by specifying an encoder backbone (`encoder_name`) and the input dimensions (`input_shape`). The `input_shape` automatically determines whether a `2D` or `3D` model will be built.

```python
from medicai.models import UPerNet

# Example 1: 3D UPerNet for volumetric segmentation
model_3d = UPerNet(
    encoder_name="efficientnetv2_m", 
    input_shape=(96, 96, 96, 1),
    num_classes=2
)

# Example 2: 2D UPerNet for image segmentation
model_2d = UPerNet(
    encoder_name="resnet50", 
    input_shape=(256, 256, 3),
    num_classes=19
)

# Example 3: 2D ConvNeXt backbone for image segmentation
model = UPerNet(
    encoder_name="convnext_tiny",
    input_shape=(224, 224, 3),
    num_classes=5
)
```

**Encoder Feature Access**

The encoder exposes its intermediate multi-scale feature maps through the attribute model `model.encoder.pyramid_outputs`. Each key corresponds to a pyramid stage, where $P1$ represents the earliest (shallowest) feature map and $P(n+1)$ represents the deepest.

```python
model = UPerNet(
    encoder_name="seresnext50", 
    input_shape=(96, 96, 96, 3)
)
model.encoder.pyramid_outputs
{
    "P1": <KerasTensor shape=(None, 48, 48, 48, 24)>,
    "P2": <KerasTensor shape=(None, 24, 24, 24, 40)>,
    "P3": <KerasTensor shape=(None, 12, 12, 12, 64)>,
    "P4": <KerasTensor shape=(None, 6, 6, 6, 176)>,
    "P5": <KerasTensor shape=(None, 3, 3, 3, 2048)>
}
```

According to the original **UPerNet design**, the Pyramid Pooling Module (PPM) operates on the deepest feature ($P5$), while the Feature Pyramid Network (FPN) fuses the intermediate stages ($P4$, $P3$, $P2$) in a top-down manner. For encoders with only four stages (e.g., **ConvNeXt**), the model uses $P4$ for the PPM and [$P3$, $P2$, $P1$] for the FPN. Note, unlike other segmentation models, **UPerNet** does not expose an `encoder_depth` argument to manually configure the number of encoder stages.


| Encoder Depth | Pyramid Pooling Module | Feature Pyramid Network |
| :-----------: | :-----------------------: | :----------------------: |
| 5 (Default)   | $P5$                      | $P4$, $P3$, $P2$         |
| 4             | $P4$                      | $P3$, $P2$, $P1$         |


## Model Customization Examples

Different backbones may produce varying numbers of feature maps or pyramid stages.
For example, the **Swin Transformer** backbone outputs five stages, but the last two may share identical spatial resolutions. In such cases, you can adjust the `head_upsample` parameter to align the decoder output with the input resolution.

```python
from medicai.models import UPerNet

# Swin Transformer backbone
model = UPerNet(
    encoder_name="swin_tiny",
    input_shape=(96,96,96,4),
    num_classes=3,
    head_upsample=8
)
```

## Hardware Compatibility

This implementation of **UPerNet** has been tested and verified on both **GPU** (`tensorflow` backend) and **TPU-VM** (`jax` backend) environments.
# UPerNet: Unified Perceptual Parsing Network

<img width="904" height="534" alt="image" src="https://github.com/user-attachments/assets/e26fdea4-6e60-4db9-a2ae-d32a8e5eaac0" />

**UPerNet** ([Unified Perceptual Parsing Network](https://arxiv.org/abs/1807.10221)) is a robust, general-purpose deep learning architecture for dense prediction tasks such as semantic segmentation, scene parsing, and medical image segmentation. It combines two key modules:

1. **Pyramid Pooling Module (PPM):** Extracts multi-scale contextual information from the deepest feature map (e.g., $P5$) using adaptive pooling at multiple scales.
2. **Feature Pyramid Network (FPN):** Implements a **top-down fusion pathway** that progressively merges semantic-rich deep features with high-resolution shallow features (e.g., $P4$, $P3$, $P2$, $P1$).

## Build Model

You can easily instantiate a **UPerNet** model by specifying an encoder backbone (`encoder_name`) and the input dimensions (`input_shape`).

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
```

**Encoder Feature Access**

The encoder exposes its intermediate multi-scale feature maps through the attribute model `encoder.pyramid_outputs`. Each key corresponds to a pyramid stage, where $P1$ represents the earliest (shallowest) feature map and $P(n+1)$ represents the deepest one.

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

## Encoder Depth

The parameter `encoder_depth` determines how many feature levels from the encoder are used by the **UPerNet** decoder.

| Encoder Depth | Pyramid Pooling Module | Feature Pyramid Network |
| :-----------: | :-----------------------: | :----------------------: |
| 5 (Default)   | $P5$                      | $P4$, $P3$, $P2$, $P1$   |
| 4             | $P4$                      | $P3$, $P2$, $P1$         |
| 3             | $P3$                      | $P2$, $P1$               |

```python
from medicai.models import UPerNet

# Example: 3D UPerNet with reduced encoder depth
model = UPerNet(
    encoder_name="efficientnetv2_m", 
    input_shape=(96, 96, 96, 1),
    encoder_depth=4
)
```

## Model Customization Examples

Different backbones may produce varying numbers of feature maps or pyramid stages.
For instance:

- **ConvNeXt** typically outputs $4$ stages.
- **Swin Transformer** outputs $5$ stages, but some have identical resolutions in the last two.

In such cases, we need to adjust both `encoder_depth` and `head_upsample` to match the architecture.

```python
from medicai.models import UPerNet

# ConvNeXt backbone
model = UPerNet(
    encoder_name="convnext_tiny",
    input_shape=(224, 224, 3),
    encoder_depth=4,
    head_upsample=4
)

# ConvNeXt with shallower pyramid
model = UPerNet(
    encoder_name="convnext_tiny",
    input_shape=(224, 224, 3),
    encoder_depth=3,
    head_upsample=4
)

# Swin Transformer backbone
model = UPerNet(
    encoder_name="swin_tiny",
    input_shape=(224, 224, 3),
    encoder_depth=5,
    head_upsample=4
)

# Swin Transformer with reduced depth
model = UPerNet(
    encoder_name="swin_tiny",
    input_shape=(224, 224, 3),
    encoder_depth=3,
    head_upsample=4
)
```

---

## ⚠️ Backend Limitation

> **Note**: The UPerNet implementation uses **Adaptive Average Pooling** in the **PPM** stage. Since **JAX** currently does not support `dynamic slicing` during adaptive pooling, this model does not support **JAX** backend training at the moment. It runs seamlessly on **TensorFlow** and **PyTorch** backends.

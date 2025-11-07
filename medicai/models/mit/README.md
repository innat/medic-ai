# Mix Vision Transformer (MixViT or MiT)

The Mix Vision Transformer (**MiT**) is a hierarchical vision transformer that modernizes standard CNN backbones. It efficiently processes both `2D` and `3D` inputs, making it suitable for a wide range of computer vision and medical imaging tasks.

The **MiT** architecture was first introduced in the **SegFormer** paper. In **SegFormer**, **MiT** serves as the encoder backbone, providing multi-scale feature representations through progressive spatial reduction.

## Build Model

You can easily instantiate a **MiT** model by selecting a variant (e.g., `MixViTB0`) and specifying the input shape. The model automatically adapts to `2D` or `3D` mode based on the input dimensionality.

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

## Feature Pyramid Output

Each **MiT** model exposes multi-scale feature maps through the `pyramid_outputs` attribute. These outputs can be used for downstream tasks such as semantic segmentation, object detection, or feature extraction.

```python
from medicai.models import MixViTB0

model = MixViTB0(
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation=None
)
model.pyramid_outputs
{
    'P1': <KerasTensor shape=(None, 24, 24, 24, 96)>,   # 1/4 scale
    'P2': <KerasTensor shape=(None, 12, 12, 12, 192)>,  # 1/8 scale
    'P3': <KerasTensor shape=(None, 6, 6, 6, 384)>,     # 1/16 scale
    'P4': <KerasTensor shape=(None, 3, 3, 3, 768)>      # 1/32 scale
}
```

Each feature map represents a progressively lower-resolution, higher-level representation of the input; forming a pyramid hierarchy commonly used in transformer-based segmentation and detection architectures.

## Spatial Reduction Ratios

The **MiT** encoder employs patch embedding layers with spatial reduction at each stage.
The reduction ratios differ slightly between `2D` and `3D` implementations:

| Version    | Spatial Reduction Ratios | Description                                                                         |
| ---------- | ------------------------------------ | ----------------------------------------------------------------------------------- |
| **2D MiT** | `[8, 4, 2, 1]`                       | Optimized for image-based tasks; maintains higher resolution early on.              |
| **3D MiT** | `[4, 2, 1, 1]`                       | Adjusted for volumetric data; ensures efficient memory use across depth dimensions. |

Both versions produce four feature levels at scales $[1/4, 1/8, 1/16, 1/32]$ relative to the input resolution.

## Pyramid Overview

| Pyramid Level | Downsample Factor | Feature Channels | Typical Use                      |
| ------------- | ----------------: | ---------------: | -------------------------------- |
| **P1**        |               1/4 |               96 | Shallow features       |
| **P2**        |               1/8 |              192 | Mid features               |
| **P3**        |              1/16 |              384 | Deep features           |
| **P4**        |              1/32 |              768 | Global features |

## Summary

- Unified transformer backbone for 2D and 3D tasks
- Outputs multi-scale hierarchical features via pyramid_outputs
- Uses stage-wise spatial reduction for efficient representation
- Serves as the encoder backbone for SegFormer and other segmentation architectures


---

**Reference**
- [SegFormer2D: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
- [SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation](https://arxiv.org/abs/2404.10156)
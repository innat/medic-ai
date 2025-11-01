# UPerNet

<img width="904" height="534" alt="image" src="https://github.com/user-attachments/assets/e26fdea4-6e60-4db9-a2ae-d32a8e5eaac0" />

**UPerNet ([Unified Perceptual Parsing Network](https://arxiv.org/abs/1807.10221))** is a robust, general-purpose deep learning framework for dense prediction tasks like **semantic segmentation**. It was introduced in the paper **"Unified Perceptual Parsing for Scene Understanding"** and achieves strong performance by effectively leveraging multi-scale features from a backbone extractor (like ConvNeXt or Swin Transformer). **UPerNet** has two key components:

1.  **Pyramid Pooling Module (PPM):** Captures multi-scale contextual information from the **deepest** feature map (bottleneck, e.g., P5).
2.  **Feature Pyramid Network (FPN):** Executes a **top-down pathway** that integrates rich semantic information from the deep layers with high-resolution details from the shallow layers (e.g., `P4`, `P3`, `P2`).


## Build Model

You can easily build a **UPerNet** model by specifying the `encoder_name` and `input_shape`.

```python
from medicai.models import UPerNet

# Build 3D UPerNet (e.g., for volumetric segmentation)
model_3d = UPerNet(
    encoder_name='efficientnetv2_m', 
    input_shape=(96, 96, 96, 1),
    num_classes=2 # Example for binary segmentation
)

# Build 2D UPerNet (e.g., for image segmentation)
model_2d = UPerNet(
    encoder_name='resnet50', 
    input_shape=(256, 256, 3),
    num_classes=19 # Example for a multi-class dataset
)
```

## Backbone Architectures and Custom Parameters

**UPerNet** can be built with various backbone architectures, including modern ones like **Swin Transformer** and **ConvNeXt**, allowing for high flexibility.

```python
input_shape = (96, 96, 96, 1) # Example 3D input shape

# Build 3D UPerNet with Swin Transformer
model_swin = UPerNet(
    input_shape=input_shape, 
    encoder_name='swin_tiny', 
    encoder_depth=4, 
    # Upsample the final output to match input size
    head_upsample=8, 
    # Reduce decoder channel size for a smaller model
    decoder_filters=128 
)

# Build 3D UPerNet with ConvNeXt
model_convnext = UPerNet(
    input_shape=input_shape, 
    encoder_name='convnext_tiny', 
    encoder_depth=4, 
    # Upsample the final output to match input size
    head_upsample=8 
)
```

## Encoder Depth

The `encoder_depth` parameter specifies how many stages of the backbone's feature pyramid are utilized by the **UPerNet** decoder. The required depth for UPerNet is `4` or `5`.

```python
from medicai.models import UPerNet

# Example of a 3D UPerNet with reduced depth
model = UPerNet(
    encoder_name='efficientnetv2_m', 
    input_shape=(96, 96, 96, 1),
    encoder_depth=4
)
```

## UPerNet Feature Selection Logic

The **UPerNet** decoder is designed to select and utilize a fixed number of feature maps from the backbone's pyramid outputs ($\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}$) based solely on the specified `encoder_depth`. The logic ensures the most relevant layers are fed to the **PPM** and **FPN** components.

Say, the backbone gives features $\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}$ (shallow to deep).

1. Feature Preparation (Based on `encoder_depth`)

The implementation sorts the required pyramid keys to identify the deepest feature map as the bottleneck.

```python
# Assuming required_keys = {'P1', 'P2', 'P3', 'P4', 'P5'} for depth 5
sorted_keys = sorted(
    required_keys, key=lambda x: int(x[1:]), reverse=True
) 
# Result for depth 5: ['P5', 'P4', 'P3', 'P2', 'P1']
```


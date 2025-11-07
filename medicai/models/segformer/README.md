# SegFormer

<img width="516" height="268" alt="Image" src="https://github.com/user-attachments/assets/b9b46a09-83af-439a-b824-9cfb879b1177" />

**SegFormer** model for `2D` or `3D` semantic segmentation. It combines a hierarchical **MixVisionTransformer** (MiT) encoder with a lightweight MLP decoder head. This design is highly efficient for semantic segmentation tasks on high-resolution images or volumes.

## Build Model

You can easily instantiate a **SegFormer** model by specifying an encoder backbone (`encoder_name`) and the input shape (`input_shape`). The dimensionality of `input_shape` automatically determines whether a `2D` or `3D` model is constructed.

```python
from medicai.models import SegFormer

# Build a 3D SegFormer
model = SegFormer(
    encoder_name='mit_b0', input_shape=(96,96,96,1)
)

# Build a 2D SegFormer
model = SegFormer(
    encoder_name='mit_b0', input_shape=(96,96,1)
)
```

**Encoder Feature Access**

The **SegFormer** encoder exposes its intermediate multi-scale feature maps through
`model.encoder.pyramid_outputs`. Each key corresponds to a pyramid stage, where $P1$ represents the shallowest features and $P4$ the deepest.

```python
model = SegFormer(
    encoder_name='mit_b0',
    input_shape=(224, 224, 3),
)
model.encoder.pyramid_outputs
{
    'P1': <KerasTensor shape=(None, 56, 56, 32),
    'P2': <KerasTensor shape=(None, 28, 28, 64),
    'P3': <KerasTensor shape=(None, 14, 14, 160),
    'P4': <KerasTensor shape=(None, 7, 7, 256),
}
```

By design, **SegFormer** performs four stages of downsampling with spatial resolutions
scaling by `[1/4, 1/8, 1/16, 1/32]` relative to the input.

## Encoder Depth

In this implementation, all four encoder stages are always utilized. There is no `encoder_depth` argument; the model structure is fixed according to the official architecture.

## 2D and 3D Implementation

Two main variants of **SegFormer** exist in the literature:

**SegFormer 2D**:
- Designed for general-purpose `2D` semantic segmentation.
- Uses Mix Vision Transformer (**MiT**) as the encoder backbone.
- Supports six **MiT** variants: `mit_b0` through `mit_b5`.
- When the input is `2D` (e.g., `(H, W, C)`), this implementation automatically builds the `2D` SegFormer model.

**SegFormer 3D**: 
- Extends the `2D` SegFormer to handle volumetric (`3D`) data by replacing `2D` operations with `3D` counterparts.
- Uses modified spatial reduction ratios:
    - 2D: `[8, 4, 2, 1]`
    - 3D: `[4, 2, 1, 1]`
- Officially demonstrated using only the `mit_b0` backbone.
- For other **MiT** variants (`mit_b1–mit_b5`), this implementation uses the same reduction strategy for compatibility.

When the input is `3D` (e.g., (`D, H, W, C`)), the model automatically constructs the `3D` variant.

## Custom Encoder

By default, **SegFormer** uses Mix Vision Transformer (**MiT**) as its encoder. The available backbone families are listed under:

```python
from medicai.models import SegFormer

SegFormer.ALLOWED_BACKBONE_FAMILIES
# ['mit']

model = SegFormer(
    encoder_name='mit_b0',
    input_shape=(224, 224, 3),
)
```

To integrate a custom encoder, provide it through the `encoder` argument. The only requirement is that the custom encoder must produce pyramid outputs with the same downsampling pattern: `[1/4, 1/8, 1/16, 1/32]`

```python
from medicai.models import SegFormer
from medicai.models import ConvNeXtV2Large

backbone = ConvNeXtV2Large(
    input_shape=(224, 224, 3),
    include_top=False
)
# backbone.pyramid_outputs

model = SegFormer(
    encoder=backbone,
    num_classes=3
)
```

---

**Reference**
- [SegFormer2D: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
- [SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation](https://arxiv.org/abs/2404.10156)
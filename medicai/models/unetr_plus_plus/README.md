# UNETR++

<img width="1158" height="455" alt="image" src="https://github.com/user-attachments/assets/7ac19033-6885-4f4a-80f3-653c36944a5d" />

**UNETR++** is an advanced transformer-based segmentation architecture designed for high-resolution 2D and 3D medical image segmentation. It extends the original **UNETR** by introducing hierarchical token representations, efficient paired attention (**EPA**), and a multi-scale feature pyramid, enabling more efficient and accurate modeling of both global context and fine spatial details.

## Build Model

We can easily instantiate a **UNETR++** model by specifying an encoder backbone (`encoder_name`) and the input dimensions (`input_shape`). The `input_shape` automatically determines whether a `2D` or `3D` model will be built.

```python
from medicai.models import UNETRPlusPlus

# Example 1: 3D UNETRPlusPlus for volumetric segmentation
model_3d = UNETRPlusPlus(
    encoder_name='unetr_plusplus_encoder',
    input_shape=(128,128,128,4),
    num_classes=3
)
# model.count_params() / 1e6 # 42.66M

# Example 2: 2D UNETRPlusPlus for image segmentation
model_2d = UNETRPlusPlus(
    encoder_name='unetr_plusplus_encoder',
    input_shape=(224,224,3),
    num_classes=3
)
# model.count_params() / 1e6 # 10.553819
```

# Dataset-Specific Configurations & Anisotropic Inputs

The official **UNETR++** paper provides four recommended configuration presets, each tailored to a popular medical segmentation dataset:

- **ACDC** (cardiac MRI)
- **Lung** (CT)
- **BraTS** (brain tumor MRI)
- **Synapse** (multi-organ CT)

These presets differ in **input resolution**, **patch size**, and **feature scaling**, reflecting the anisotropic nature of many medical imaging modalities.

**Default Configuration**

By default, the **UNETR++** model in this implementation is built using the BraTS configuration, which assumes **isotropic** 3D inputs and multi-modal channels.

```python
from medicai.models import UNETRPlusPlus

model = UNETRPlusPlus(
    encoder_name="unetr_plusplus_encoder",
    input_shape=(128, 128, 128, 4),
    num_classes=3
)
```
This setup is suitable for:
- Multi-channel MRI volumes
- Isotropic voxel spacing
- Brain tumor segmentation tasks

**Anisotropic Input Support**

Many medical datasets exhibit **anisotropic** voxel spacing, particularly along the depth (slice) dimension. To support such cases, **UNETR++** allows explicit control over the **patch size** at the encoder level, enabling **anisotropic** tokenization while keeping the decoder unchanged. To do this, we can manually instantiate the encoder and pass it into the **UNETR++** model.

```python
from medicai.models import UNETRPlusPlus, UNETRPlusPlusEncoder

encoder = UNETRPlusPlusEncoder(
    input_shape=(16, 160, 160, 1),
    patch_size=(1, 4, 4)  # preserve depth resolution
)
model = UNETRPlusPlus(
    encoder=encoder,
    num_classes=3
)

encoder = UNETRPlusPlusEncoder(
    input_shape=(64, 128, 128, 1),
    patch_size=(2, 4, 4)  # moderate depth downsampling
)
model = UNETRPlusPlus(
    encoder=encoder,
    num_classes=3
)

```

# Encoder Feature Access

The UNETR++ encoder exposes its hierarchical transformer features through the attribute model.encoder.pyramid_outputs. Each key corresponds to a transformer stage, where:
- $P1$ denotes the highest-resolution (shallowest) token feature map.
- $P4$ denotes the lowest-resolution (deepest) token feature map.
These multi-scale representations are used directly by the **UNETR++** decoder for skip connections.

```python
model = UNETRPlusPlus(
    encoder_name="unetr_plusplus_encoder",
    input_shape=(96, 96, 96, 1)
)

model.encoder.pyramid_outputs
{
    "P1": <KerasTensor shape=(None, 24, 24, 24, 32)>,
    "P2": <KerasTensor shape=(None, 12, 12, 12, 64)>,
    "P3": <KerasTensor shape=(None, 6, 6, 6, 128)>,
    "P4": <KerasTensor shape=(None, 3, 3, 3, 256)>
}
```

# Skip Connection Design Choice

Unlike **UNETR**, in **UNETR++** implementation uses element-wise addition (+) instead of concatenation (concat) for skip connections. Additive fusion requires strict channel and spatial alignment between encoder and decoder features, which tightly couples the decoder to a specific encoder design. This makes it difficult to support arbitrary backbones such as **ConvNeXt** or **MiT**, which produce heterogeneous feature dimensions.



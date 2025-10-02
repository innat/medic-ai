# Swin Transformer V1, V2, SwinUNETR

Swin Transformer and SwinUNETR for 2D and 3D data on classification and segmentation task.

- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883)
- [Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images](https://arxiv.org/abs/2201.01266)
- [SwinUNETR-V2: Stronger Swin Transformers with Stagewise Convolutions for 3D Medical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-43901-8_40)

**Note**: In `SwinUNETR-V2`, it does **not** build by combining `Swin-V2` + `UNETR`. Instead, it proposed an additional residual convolutional block after each stage of `Swin Transformer` model. So, in `SwinUNETR-V2` model, we can inject `Swin-V1` or `Swin-V2` encoder.

---

## Swin Transformer V1 and V2

```python
from medicai.models import SwinBackbone, SwinTiny, SwinSmall, SwinBase
from medicai.models import SwinBackboneV2, SwinTinyV2, SwinSmallV2, SwinBaseV2

# To build 3D model
model = SwinTiny(input_shape=(96,96,96,1))

# To build 2D model
model = SwinTiny(input_shape=(96,96,1))

# Arbitrary input shape
model = SwinTiny(
    input_shape=(8, 224, 256, 3), 
    num_classes=1,
    classifier_activation='sigmoid',
    patch_size=4,
    window_size=8
)
```

## Swin-UNETR

```python
from medicai.models import SwinUNETR

# Build with string encoder identifier.
swin_unetr = SwinUNETR(
    encoder_name='swin_tiny'  # 'swin_tiny', 'swin_small', ...
    input_shape=(96, 96, 96, 1),
    num_classes=1,
    classifier_activation='sigmoid',
)

# Build model with swin-v2 model.
swin_unetr = SwinUNETR(
    encoder_name='swin_tiny_v2'  # 'swin_tiny_v2', 'swin_small_v2', ...
    input_shape=(96, 96, 96, 1),
    num_classes=1,
    classifier_activation='sigmoid',
)
```

```python
# Build with encoder object.
from medicai.models import SwinTiny

# In SwinUNETR, the patch-merging doesn't apply in last stage.
# Therefore, downsampling_strategy = 'swin_unetr_like'
model = SwinTiny(
    input_shape=(96,96,96,1),
    patch_size=2, 
    downsampling_strategy='swin_unetr_like'
)
swin_unetr = SwinUNETR(encoder=model)
```

```python
# Custom encoder
from medicai.models import SwinBackbone, SwinBackboneV2

custom_encoder = SwinBackboneV2(
    input_shape=(64, 128, 128, 1),
    embed_dim=48,
    window_size=8,
    patch_size=2,
    downsampling_strategy='swin_unetr_like'
)
swin_unetr = SwinUNETR(encoder=custom_encoder)
```

## Swin-UNETR-V2 [WIP]
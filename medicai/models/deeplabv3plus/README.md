# DeepLabV3Plus

<img width="1100" height="580" alt="image" src="https://github.com/user-attachments/assets/c169636e-8be7-4ce6-b1d8-51ffb1f2a065" />

The **DeepLabV3Plus** model provides a highly effective solution for semantic segmentation, designed to be flexible across both 2D (Image) and 3D (Volumetric/Video) input data. This architecture fundamentally relies on an Encoder-Decoder structure:
- **Encoder**: Extracts deep, semantic, and context-rich features.
- **Decoder**: Merges the high-level features processed by the **Atrous Spatial Pyramid Pooling** (ASPP) module with fine-grained, low-level features (via a skip connection) to accurately delineate boundaries and produce fine segmentation masks.

**Model Initialization**

The model supports popular backbones like `resnet`, `densenet`, `efficientnet`, `convnext`, and `senet` families. The implementation dynamically selects 2D or 3D layers based on the `input_shape`. Check: `medicai.models.list_models()` for more details.

```python
from medicai.models import DeepLabV3Plus

# build 2D model with resnet backbone
model = DeepLabV3Plus(
    encoder_name='resnet50',
    input_shape=(224, 224, 3),
    num_classes=1,
    classifier_activation="sigmoid",
)

# build 3D model with se-resnext backbone
model = DeepLabV3Plus(
    encoder_name='seresnext50',
    input_shape=(96, 96, 96, 3),
    num_classes=1,
    classifier_activation="sigmoid",
)
```

**Encoder Feature Access and Selection**

The encoder exposes its feature maps through the `model.encoder.pyramid_outputs` attribute, where features are keyed by their stage, `P1` being the earliest stage.

```python
model = DeepLabV3Plus(
    encoder_name='seresnext50', 
    input_shape=(96, 96, 96, 3)
)
model.encoder.pyramid_outputs
{
    'P1': <KerasTensor shape=(None, 48, 48, 48, 24)>,
    'P2': <KerasTensor shape=(None, 24, 24, 24, 40)>,
    'P3': <KerasTensor shape=(None, 12, 12, 12, 64)>,
    'P4': <KerasTensor shape=(None, 6, 6, 6, 176)>,
    'P5': <KerasTensor shape=(None, 3, 3, 3, 2048)>
}
```

In this **DeepLabV3+** implementation, the two main features used in the decoder are:
1. **Deep Feature (High-Level Context)**: The deepest available feature map `P(n)` is fed into the **ASPP** module.
2. **Low-Level Feature (Boundary Detail)**: The `P2` feature map is consistently used for the low-level skip connection, which typically offers $\frac{1}{4}$ of the input resolution for fine-detail preservation.

**Encoder Depth**

The `encoder_depth` parameter allows you to control how many stages of the encoder backbone are utilized, which directly impacts the Deep Feature used for **ASPP**.


| Encoder Depth | Deep Feature (ASPP Input) | Low-Level Feature (Skip) |
| :-----------: | :-----------------------: | :----------------------: |
| 5 (Default)   | $P5$                      | $P2$                     |
| 4             | $P4$                      | $P2$                     |
| 3             | $P3$                      | $P2$                     |


For example, setting `encoder_depth=4` will use the $P4$ feature map as the **ASPP** input.


**Custom Encoder**

You can provide a pre-configured or custom encoder model via the `encoder` argument. When using a custom encoder, ensure the following:
1. It provides the `pyramid_outputs` dictionary attribute with appropriately keyed tensors ($P1, P2, \dots$).
2. The `encoder_depth` parameter matches the number of feature vectors provided (e.g., `encoder_depth=4` if your custom model only provides $P1$ to $P4$).
3. You may need to manually adjust the `head_upsample` factor to ensure the final output matches the input size, especially if the feature resolutions are non-standard.

```python
from medicai.models import ConvNeXtBackbone

# build custom encoder
backbone = ConvNeXtBackbone(
    input_shape=(64,64,64,1),
    depths=[3, 3, 9, 3],
    projection_dims=[96, 192, 384, 768],
)
# gives 4 feature vectors, i.e. `P1` to `P4`.
backbone.pyramid_outputs 

# build deeplabv3plus with custom encoder
model = DeepLabV3Plus(
    encoder=backbone,
    num_classes=1,
    classifier_activation="sigmoid",
    encoder_depth=4,
    head_upsample=8,
)
```

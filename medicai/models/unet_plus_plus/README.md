# UNet++

`UNet++` model for 2D and 3D semantic segmentation. `UNet++` is an advanced segmentation architecture that features dense skip connections between encoder and decoder paths. Unlike standard UNet which uses simple skip connections, UNet++ creates a dense grid of convolutional blocks that enable multi-scale feature fusion.

**build model**

```python
from medicai.models import UNetPlusPlus

# build 3d unet model
model = UNetPlusPlus(
    encoder_name='efficientnetv2_m', input_shape=(96, 96, 96, 1)
)

# build 2d unet model
model = UNetPlusPlus(
    encoder_name='efficientnetv2_m', input_shape=(96, 96, 1)
)
```

**decoder type**

By default, the `decoder_block_type="upsampling"`, but we can use trainable convolutional network. For example:

```python
from medicai.models import UNetPlusPlus

model = UNetPlusPlus(
    encoder_name='efficientnetv2_m', 
    input_shape=(96, 96, 96, 1),
    decoder_block_type="transpose"
)
```

**encoder depth**

We can use `encoder_depth` to specifying how many stages of the encoder backbone to use.

```python
from medicai.models import UNetPlusPlus

model = UNetPlusPlus(
    encoder_name='efficientnetv2_m', 
    input_shape=(96, 96, 96, 1),
    encoder_depth=3
)
```

By default, `encoder_depth` is set to `5`, referring to five stages of the feature pyramid: `[P1, P2, P3, P4, P5]`. If `encoder_depth=5`, where `P5` would be bottleneck, and create `4 x 4 = 16` node for skip connection of `[P1, P2, P3, P4]`. Similarly:
- If `encoder_depth=4`:
    - Bottleneck: `P4`
    - Skip connections: `[P3, P2, P1]`
    - Creates a `3×3` dense grid (`9` decoder nodes)
- If `encoder_depth=3`:
    - Bottleneck: `P3`
    - Skip connections: `[P2, P1]`
    - Creates a `2×2` dense grid (`4` decoder nodes)

Reducing the encoder depth significantly decreases model size and computational requirements while maintaining the dense connection benefits of `UNet++`.

**Dense Skip Connections**:

UNet++ uses dense connections between all nodes at the same resolution level. For `encoder_depth=5`:

```python
X0,0 -- X0,1 -- X0,2 -- X0,3 -- X0,4
  |       |       |       |
X1,0 -- X1,1 -- X1,2 -- X1,3
  |       |       |
X2,0 -- X2,1 -- X2,2
  |       |
X3,0 -- X3,1
  |
X4,0
```

Where:
- `X_{i,0}` are encoder features `(X4,0 = P5, X3,0 = P4, X2,0 = P3, X1,0 = P2, X0,0 = P1)`.
- Each node `X_{i,j}` receives inputs from its **parent node** and **all previous nodes** in the same row.
- A final bridge upsampling ensures output matches input resolution


**custom encoder**

We can also pass a custom encoder to build the unet model. But we need to make sure
it has the `encoder.pyramid_outputs` attributes with `5` feature vectors, i.e.,

```bash
{
    'P1': <KerasTensor shape=(None, 48, 48, 48, 24)>,
    'P2': <KerasTensor shape=(None, 24, 24, 24, 40)>,
    'P3': <KerasTensor shape=(None, 12, 12, 12, 64)>,
    'P4': <KerasTensor shape=(None, 6, 6, 6, 176)>,
    'P5': <KerasTensor shape=(None, 3, 3, 3, 2048)>
}
```

All the built-in models in `medicai` have these attributes.

```python
from medicai.models import UNetPlusPlus
from medicai.models import DenseNetBackbone

# config of densenet-264
encoder = DenseNetBackbone(
    blocks=[6, 12, 64, 48],
    input_shape=(96, 96, 96, 4),
)
# encoder.pyramid_outputs # ok

model = UNetPlusPlus(
    encoder=encoder, 
    encoder_depth=4,
    num_classes=3,
    classifier_activation='softmax',
)
```
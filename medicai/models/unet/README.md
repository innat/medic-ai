# UNet and AttentionUNet

## UNet

`UNet` model for semantic segmentation.

**build model**

```python
from medicai.models import UNet

# build 3d unet model
model = UNet(
    encoder_name='efficientnetv2_m', input_shape=(96, 96, 96, 1)
)

# build 2d unet model
model = UNet(
    encoder_name='efficientnetv2_m', input_shape=(96, 96, 1)
)
```

**decoder type**

By default, the `decoder_block_type="upsampling"`, but we can use trainable convolutional network. For example:

```python
from medicai.models import UNet

model = UNet(
    encoder_name='efficientnetv2_m', 
    input_shape=(96, 96, 96, 1),
    decoder_block_type="transpose"
)
```

**encoder depth**

We can use `encoder_depth` to specifying how many stages of the encoder backbone to use.

```python
from medicai.models import UNet

model = UNet(
    encoder_name='efficientnetv2_m', 
    input_shape=(96, 96, 96, 1),
    encoder_depth=3
)
```

**custom encoder**

By default, it is set `5`, referring five stage of feature pyramid layer,
i.e. `[P1, P2, P3, P4, P5]`. If `encoder_depth=3`, bottleneck key
would be `P3`, and `P2..P1` will be used for skip connection. This can be used to
reduce the size of the model.

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

All the built-in models in `medicai` have these attributes. So, we can do:

```python
from medicai.models import UNet
from medicai.models import DenseNetBackbone

# config of densenet-264
encoder = DenseNetBackbone(
    blocks=[6, 12, 64, 48],
    input_shape=(96, 96, 96, 4),
)
# encoder.pyramid_outputs # ok

model = UNet(
    encoder=encoder, 
    encoder_depth=4,
    num_classes=3,
    classifier_activation='softmax',
)
```

## AttentionUNet

Extends the `UNet` architecture by integrating **Attention Gates** in the decoder path, allowing the network to focus on relevant spatial regions. The API usage is same as `UNet`, mentioned above.

```python
AttentionUNet(
    input_shape=None,
    encoder_name=None,
    encoder=None,
    encoder_depth=5,
    decoder_block_type="upsampling",
    decoder_filters=(256, 128, 64, 32, 16),
    decoder_use_batchnorm=True,
    num_classes=1,
    classifier_activation="sigmoid",
    name=None,
)
```


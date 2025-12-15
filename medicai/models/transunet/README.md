# TransUNet

![](https://github.com/user-attachments/assets/a9f568d8-5e17-433f-a71d-4ed5c94da280)

[TransUNet](https://arxiv.org/abs/2310.07781) is a powerful medical image segmentation model that integrates **CNNs** for capturing local spatial details with **Transformers** for modeling global context. By effectively combining fine-grained texture information and long-range dependencies, it is well suited for tasks such as organ segmentation and often outperforms models based solely on CNNs or Transformers.

## Build Model

We can easily instantiate a **TransUNet** model by specifying an encoder backbone (`encoder_name`) and the input dimensions (`input_shape`). The `input_shape` automatically determines whether a `2D` or `3D` model will be built.

```python
from medicai.models import TransUNet

# Example 1: To build 3D model
model = TransUNet(
    encoder_name='resnet18', 
    input_shape=(96,96,96,1),
    num_classes=3,
    classifier_activation=None,
)

# Example 2: To build 2D model
model = TransUNet(
    encoder_name='efficientnet_b2', 
    input_shape=(96,96,1),
    num_classes=3,
    classifier_activation=None,
)
```

**Encoder Depth**

We can use `encoder_depth` to specifying how many stages of the encoder backbone to use. This will also reduce the model parameter for faster prototype.

```python
# Example 1: Tune encoder_depth
model = TransUNet(
    encoder_name='efficientnet_b2', 
    encoder_depth=4,
    input_shape=(96,96,96,1),
    num_classes=3,
    classifier_activation=None,
)
```

By default, `encoder_depth` is set to $5$ (bottleneck layer), referring to five stages of the feature pyramid: `[P1, P2, P3, P4, P5]`. If it is set to $4$, then `P4` will be used as bottleneck layer. The valid range of `encoder_depth` are `[3, 4, 5]`.

In addition to the **CNN** backbone, the second encoder of the **TransUNet**, the **Vision Transformer** (ViT) can also be tuned to control overall model capacity and computational cost. By adjusting the ViT configuration, we can reduce the model size while preserving segmentation performance.

```python
# Example 1: Tune vit encoder
model = TransUNet(
    encoder_name='densenet121', 
    input_shape=(96,96,96,1),
    num_vit_layers=12,
    num_heads=8,
    embed_dim=512,
    mlp_dim=1024,
)
```

Some notes about this parameter:

| Parameter        | Role                           | Scaling guideline                     | Effect on model                  |
| ---------------- | ------------------------------ | ------------------------------------- | -------------------------------- |
| `embed_dim`      | Transformer feature width      | Increase/decrease in steps (256–1024) | Dominates memory and compute     |
| `num_heads`      | Number of attention heads      | Keep `embed_dim / num_heads ≈ 32–64`  | Controls attention diversity     |
| `mlp_dim`        | Hidden size of MLP block       | `2×–4× embed_dim`                     | Affects per-layer capacity       |
| `num_vit_layers` | Number of Transformer layers   | 6–24 depending on model size          | Controls model depth             |


## Encoder Feature Access and Selection

The encoder exposes its feature maps through the `model.encoder.pyramid_outputs` attribute, where features are keyed by their stage, `P1` being the earliest stage. 

```python
model = TransUNet(
    encoder_name='efficientnet_b0', 
    input_shape=(96,96,96,1),
    num_classes=3,
    classifier_activation=None,
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

Additionally, we can also build a auxilary classifier model with `model.encoder`.

## Pre-configured Encoder and Custom Encoder

### Pre-configured Encoder

To inspect the default backbone families supported by **TransUNet**, you can query the allowed encoder families as follows:

```python
>>> import medicai
>>> from medicai.models import TransUNet

>>> TransUNet.ALLOWED_BACKBONE_FAMILIES
>>> ['densenet', 'resnet', 'efficientnet', 'senet', 'xception']
```

You can also list the available model variants within a specific backbone family:

```python
>>> medicai.models.list_models(family='densenet')
                Model Registry Catalog                
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Segmentor        ┃ Backbone Family ┃ Variants      ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ • attention_unet │ densenet        │ • densenet121 │
│ • deeplabv3plus  │                 │ • densenet169 │
│ • trans_unet     │                 │ • densenet201 │
│ • unet           │                 │ • densenet264 │
│ • unet_plus_plus │                 │               │
│ • upernet        │                 │               │
└──────────────────┴─────────────────┴───────────────┘
```

Once selected, a pre-configured backbone can be used directly when initializing the model:

```python
>>> model = TransUNet(
    encoder_name='densenet264', input_shape=(224, 224, 3)
)
```

### Custom Encoder

In addition to pre-configured backbones, **TransUNet** also supports custom encoder models via the `encoder` argument. When using a custom encoder with `medicai`, ensure that the following conditions are met:

1. The encoder exposes a `pyramid_outputs` dictionary containing multi-scale feature tensors (e.g., `P1, P2, …`).
2. The `encoder_depth` parameter matches the number of feature levels used (for example, `encoder_depth=4` if only `P1–P4` are provided).
3. If the encoder produces non-standard feature resolutions, a final resizing step may be required to restore the output to the original input size.


**Example: Using a Swin Transformer Encoder**

The following example demonstrates how to replace a **CNN** backbone with a **Swin Transformer** in **TransUNet**.

```python
from medicai.models import TransUNet, SwinTinyV2

backbone = SwinTinyV2(
    input_shape=(96,96,96,1),
    include_top=False
)
backbone.pyramid_outputs
{
    'P1': <KerasTensor shape=(None, 24, 24, 24, 40)>,
    'P2': <KerasTensor shape=(None, 12, 12, 12, 96)>,
    'P3': <KerasTensor shape=(None, 6, 6, 6, 192)>,
    'P4': <KerasTensor shape=(None, 3, 3, 3, 384)>
    'P5': <KerasTensor shape=(None, 3, 3, 3, 384)>
}
```

In this case, the `P5` feature is not required. Therefore, only the first four pyramid levels are used by setting `encoder_depth=4`:

```python
model = TransUNet(
    input_shape=(96,96,96,1),
    encoder=backbone,
    encoder_depth=4,
    classifier_activation='softmax',
    num_classes=3,
)
```

Inspecting the model output reveals that the spatial resolution is lower than the input:

```python
>>> model.output
KerasTensor shape=(None, 48, 48, 48, 3)
```

To match the segmentation output to the original input resolution, a final upsampling step can be applied:

```python
from medicai.models import SwinTinyV2, TransUNet
from medicai.layers import ResizingND

input_shape = (96, 96, 96)

backbone = SwinTinyV2(
    input_shape=input_shape + (1,),
    include_top=False,
)

segmentor = TransUNet(
    encoder=backbone,
    encoder_depth=4,
    num_classes=3,
    classifier_activation='softmax',
)

inputs = keras.Input(shape=input_shape + (1,))
x = segmentor(inputs)

outputs = ResizingND(
    target_shape=input_shape,
    interpolation='trilinear',
    align_corners=False,
)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

We can also replace the **Swin Transformer** with **ConvNeXt** model in above example code. This approach enables seamless integration of both pre-configured and custom encoders while maintaining flexibility over feature depth and output resolution.


---

**Note**: There are two published **TransUNet** for 2D and 3D task, with different decoder modelling. In 3D **TransUNet**, it is mentioned that, this model can be applied in 2D task by replacing the 3D ops to 2D ops. In this codebase, 3D version of TransUNet is implemented. If 2D input shape is passed, the built model would be 3D TransUNet in 2D version.

**Reference**
- [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)
- [3D TransUNet: Advancing Medical Image Segmentation through Vision Transformers](https://arxiv.org/abs/2310.07781)
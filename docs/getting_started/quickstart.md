---
orphan: true
---

# Quick start

This page walks through the core `medicai` workflow:

1. Select a Keras backend.
2. Build preprocessing transforms.
3. Explore and instantiate models.
4. Train with standard Keras or custom loops.
5. Run inference for `2D` and `3D` workloads.

## Verify setup

Set the Keras backend before importing `keras` or `medicai`:

```python
import os

# "tensorflow" | "torch" | "jax"
os.environ["KERAS_BACKEND"] = "jax"

import keras
import medicai

print(f"keras version  : {keras.version()}")
print(f"keras backend  : {keras.config.backend()}")
print(f"medicai version: {medicai.version()}")
```

## Transformations

`medicai.transforms` is designed for medical imaging workflows and is
implemented with TensorFlow-compatible operations for volumetric data.

```{eval-rst}
.. note::

   Volumetric transforms in ``medicai.transforms`` currently depend on
   TensorFlow operations and therefore require the ``tensorflow`` package
   to be installed, regardless of the selected Keras backend.

   This requirement only applies to the transform pipeline. Model training
   and inference can still use any supported backend, including ``torch``
   and ``jax`` without forcing you to train with TensorFlow.

   The transforms can be integrated into a variety of data loading workflows,
   including:

   - ``tf.data.Dataset``
   - ``keras.utils.PyDataset`` by converting samples to ``numpy``
   - ``torch.utils.data.Dataset`` and ``DataLoader`` by converting samples to ``numpy``

   In other words, TensorFlow is required for executing the transforms, but
   it does not restrict the backend used for model development, training,
   or inference.
```

Example preprocessing pipeline:

```python
from medicai.transforms import Compose, NormalizeIntensity, RandFlip

transforms = Compose(
    [
        NormalizeIntensity(keys=["image"]),
        RandFlip(
            keys=["image", "label"], 
            prob=0.5, 
            spatial_axis=[0]
        ),
    ]
)
```

## Models

Inspect the registered model zoo:

```sh
medicai.models.list_models()
```
```bash
                     Model Registry Catalog                      
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Segmentor        ┃ Backbone Family ┃ Variants                 ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ • attention_unet │ convnext        │ • convnext_base          │
│ • deeplabv3plus  │                 │ • convnext_large         │
│ • unet           │                 │ • convnext_small         │
│ • unet_plus_plus │                 │ • convnext_tiny          │
│ • upernet        │                 │ • convnext_v2_atto       │
│                  │                 │ • convnext_v2_base       │
│                  │                 │ • convnext_v2_femto      │
│                  │                 │ • convnext_v2_huge       │
│                  │                 │ • convnext_v2_large      │
│                  │                 │ • convnext_v2_nano       │
│                  │                 │ • convnext_v2_pico       │
│                  │                 │ • convnext_v2_small      │
│                  │                 │ • convnext_v2_tiny       │
│                  │                 │ • convnext_xlarge        │
├──────────────────┼─────────────────┼──────────────────────────┤
│ • attention_unet │ densenet        │ • densenet121            │
│ • deeplabv3plus  │                 │ • densenet169            │
│ • trans_unet     │                 │ • densenet201            │
│ • unet           │                 │ • densenet264            │
│ • unet_plus_plus │                 │                          │
│ • upernet        │                 │                          │
├──────────────────┼─────────────────┼──────────────────────────┤
│ • attention_unet │ efficientnet    │ • efficientnet_b0        │
│ • deeplabv3plus  │                 │ • efficientnet_b1        │
│ • trans_unet     │                 │ • efficientnet_b2        │
│ • unet           │                 │ • efficientnet_b3        │
│ • unet_plus_plus │                 │ • efficientnet_b4        │
│ • upernet        │                 │ • efficientnet_b5        │
│                  │                 │ • efficientnet_b6        │
│                  │                 │ • efficientnet_b7        │
│                  │                 │ • efficientnet_b8        │
│                  │                 │ • efficientnet_l2        │
│                  │                 │ • efficientnet_v2_b0     │
│                  │                 │ • efficientnet_v2_b1     │
│                  │                 │ • efficientnet_v2_b2     │
│                  │                 │ • efficientnet_v2_b3     │
│                  │                 │ • efficientnet_v2_l      │
│                  │                 │ • efficientnet_v2_m      │
│                  │                 │ • efficientnet_v2_s      │
├──────────────────┼─────────────────┼──────────────────────────┤
│ • segformer      │ mit             │ • mit_b0                 │
│ • upernet        │                 │ • mit_b1                 │
│                  │                 │ • mit_b2                 │
│                  │                 │ • mit_b3                 │
│                  │                 │ • mit_b4                 │
│                  │                 │ • mit_b5                 │
├──────────────────┼─────────────────┼──────────────────────────┤
│ • attention_unet │ resnet          │ • resnet101              │
│ • deeplabv3plus  │                 │ • resnet101_v2           │
│ • trans_unet     │                 │ • resnet152              │
│ • unet           │                 │ • resnet152_v2           │
│ • unet_plus_plus │                 │ • resnet18               │
│ • upernet        │                 │ • resnet200_vd           │
│                  │                 │ • resnet34               │
│                  │                 │ • resnet50               │
│                  │                 │ • resnet50_v2            │
│                  │                 │ • resnet50_vd            │
│                  │                 │ • resnext101             │
│                  │                 │ • resnext50              │
├──────────────────┼─────────────────┼──────────────────────────┤
│ • attention_unet │ senet           │ • seresnet101            │
│ • deeplabv3plus  │                 │ • seresnet101_v2         │
│ • trans_unet     │                 │ • seresnet152            │
│ • unet           │                 │ • seresnet152_v2         │
│ • unet_plus_plus │                 │ • seresnet18             │
│ • upernet        │                 │ • seresnet200_vd         │
│                  │                 │ • seresnet34             │
│                  │                 │ • seresnet50             │
│                  │                 │ • seresnet50_v2          │
│                  │                 │ • seresnet50_vd          │
│                  │                 │ • seresnext101           │
│                  │                 │ • seresnext50            │
├──────────────────┼─────────────────┼──────────────────────────┤
│ • swin_unetr     │ swin            │ • swin_base              │
│ • upernet        │                 │ • swin_base_v2           │
│                  │                 │ • swin_small             │
│                  │                 │ • swin_small_v2          │
│                  │                 │ • swin_tiny              │
│                  │                 │ • swin_tiny_v2           │
├──────────────────┼─────────────────┼──────────────────────────┤
│ • unetr_plusplus │ unetr_plusplus  │ • unetr_plusplus_encoder │
├──────────────────┼─────────────────┼──────────────────────────┤
│ • unetr          │ vit             │ • vit_base               │
│                  │                 │ • vit_huge               │
│                  │                 │ • vit_large              │
├──────────────────┼─────────────────┼──────────────────────────┤
│ • attention_unet │ xception        │ • xception               │
│ • trans_unet     │                 │                          │
│ • unet           │                 │                          │
│ • unet_plus_plus │                 │                          │
│ • upernet        │                 │                          │
└──────────────────┴─────────────────┴──────────────────────────┘
```

The registry groups models by task and backbone family, which makes it easy to
discover which encoders can be reused across multiple segmentation heads. We can also filter by family:


```python
medicai.models.list_models(family="vit")
```
```bash
              Model Registry Catalog               
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Segmentor       ┃ Backbone Family ┃ Variants    ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ • unetr         │ vit             │ • vit_base  │
│                 │                 │ • vit_huge  │
│                 │                 │ • vit_large │
└─────────────────┴─────────────────┴─────────────┘
```

Create a model from the registry:

```python
model = medicai.models.create_model(
    name="vit_base",
    input_shape=(224, 224, 3),
    num_classes=10,
    classifier_activation="softmax",
)

model.summary()
```

This is a convenient way when we want a preset architecture without importing
the class directly. We can also instantiate model classes directly:

```python
from medicai.models import ViTBase

ViTBase.class_describe()
```

The `class_describe()` helper prints the model docstring and constructor
details in a readable format, which is helpful when we are exploring a new
architecture interactively.

```python
model = ViTBase(
    input_shape=(224, 224, 3),
    num_classes=10,
    classifier_activation="softmax",
)

model.count_params() / 1e6
```

This returns the parameter count in millions, which is a quick way to compare
model sizes before training. The same model family can switch between ``2D`` and ``3D`` based on `input_shape`:

```python
model = ViTBase(
    input_shape=(128, 128, 128, 1),
    num_classes=10,
    classifier_activation="softmax",
)

model.count_params() / 1e6
model.instance_describe()
```

`instance_describe()` summarizes the concrete configuration of the model we
just built, which is useful when switching between `2D` and `3D` variants.

All encoder-style models expose intermediate feature maps through
`model.pyramid_outputs`:

```python
model.pyramid_outputs
```

For transformer backbones, these pyramid entries correspond to intermediate
token representations that can be reused by downstream heads or custom feature
extractors. We can reuse those features to build a feature extractor:

```python
feature_extractor = keras.Model(
    model.inputs,
    model.pyramid_outputs["P5"],
)
```

Transformer encoders can also be plugged into segmentation models such as
`UNETR`:

```python
model = medicai.models.create_model(
    name="unetr",
    encoder_name="vit_base",
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation="sigmoid",
)

model.count_params() / 1e6
```

Direct class-based construction works the same way:

```python
from medicai.models import UNETR

model = UNETR(
    encoder_name="vit_base",
    input_shape=(96, 96, 96, 4),
    num_classes=3,
    classifier_activation="sigmoid",
)

model.count_params() / 1e6
model.encoder.pyramid_outputs
```

In `UNETR`, the decoder consumes these encoder features internally for
multi-scale reconstruction.

## Training

`medicai` works with several training patterns:

- Standard Keras training with `model.fit()`
- Custom training loops with TensorFlow, PyTorch, or JAX
- Input pipelines built with `tf.data`, `torch.utils.data`, or `pygrain`

Example Keras workflow:

```python
model.compile(
    optimizer="adam",
    loss=medicai.losses.SparseDiceCELoss(
        from_logits=True, num_classes=5
    ),
    metrics=[
        medicai.metrics.SparseDiceMetric(
            from_logits=True, num_classes=5
        )
    ],
)

model.fit(
    train_dataset, validation_data=val_dataset, epochs=10
)
```

## Inference

For `2D` classification, `2D` segmentation, and `3D` classification, we can
use the standard Keras prediction API:

```python
predictions = model.predict(x)
```

For ``3D`` segmentation, sliding-window inference is usually the better choice for
large volumes:

```python
from medicai.utils import SlidingWindowInference

swi = SlidingWindowInference(
    model=model,
    num_classes=3,
    roi_size=(96, 96, 96),
    sw_batch_size=1,
    overlap=0.25,
)

predictions = swi(volume)
```

## Utility

### Grad-CAM

`medicai.utils.GradCAM` can be used for model interpretability across:

- `2D` classification
- `3D` classification
- `2D` segmentation
- `3D` segmentation
- all supported Keras backends: `tensorflow`, `torch`, and `jax`

Grad-CAM works by selecting an intermediate feature-producing layer and
computing a heatmap that shows which spatial regions contributed most to a
target class prediction.

Before creating the Grad-CAM utility, it is often helpful to inspect the model
layers and their output shapes so we can choose a meaningful target layer:

```python
for layer in model.layers:
    print(layer.name, layer.output.shape)
```

In general, a good target layer is one of the deeper convolutional or feature
projection layers that still preserves useful spatial structure. For
classification models, this is often the last convolution-style feature layer.
For segmentation models, it can be a decoder or encoder feature layer depending
on which region we want to explain.

Example:

```python
import numpy as np
from medicai.utils import GradCAM

cam = GradCAM(
    model=model,
    target_layer="decoder_stage1_conv_2_activation",
)

heatmap = cam.compute_heatmap(
    input_tensor=np.random.randn(1, 64, 128, 128, 1),
    target_class_index=3,
)
```

For segmentation models, `compute_heatmap()` also supports different masking
strategies such as `object`, `all`, and `single` to control how the target
region contributes to gradient computation.

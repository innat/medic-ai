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

The `medicai.transforms` is designed for medical imaging workflows and is
implemented with backend-native Keras operations for 2D and 3D medical data.
Set `KERAS_BACKEND` before importing `keras` or `medicai`.

```{eval-rst}
.. note::

   Transforms use ``keras.ops`` and run with the selected ``tensorflow``,
   ``torch``, or ``jax`` backend. They use channel-last layouts:

   - single 2D samples: ``HWC``
   - single 3D samples: ``DHWC``
   - batches of 2D samples: ``BHWC``
   - batches of 3D samples: ``BDHWC``

   Most transforms support both sample and batch layouts. Check the individual
   transform documentation for sample-only operations and backend-specific
   XLA or compiled-mode limitations.
```

Choose a dataloader that is supported by the selected Keras backend:

| Keras backend | PyGrain | `torch.utils.data` | `tf.data` | `keras.utils.PyDataset` |
| :--- | :---: | :---: | :---: | :---: |
| TensorFlow | ✓ | ✗ | ✓ | ✓ |
| Torch | ✓ | ✓ | ✗ | ✓ |
| JAX | ✓ | ✗ | ✗ | ✓ |

For an end-to-end pipeline that can be reused across all three backends,
prefer [**PyGrain**](https://github.com/google/grain) or `keras.utils.PyDataset`. PyGrain is recommended for
parallel data loading and supports multithreading and multiprocessing workers.

Some examples preprocessing with different backends:

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
from medicai.transforms import RandomElasticTransform

images = torch.randn((4, 224, 224, 3))
affine = torch.diag(torch.tensor([0.7, 0.7, 1.0, 1.0]))
transform = RandomElasticTransform(
    keys=["image"],
    input_layout="BHWC",
    alpha=2.0,
    sigma=3.0,
    displacement_units="mm",
    minimum_physical_spacing=0.7,
    field_interpolation="bspline",
    seed=108,
)
result = transform({"image": images}, {"affine": affine})
```

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import jax
from medicai.transforms import RandomRotate

transform = RandomRotate(
    keys=["image"],
    factor={"y": 0.1, "x": 0.1},
    prob=0.5,
    input_layout="DHWC",
)
image = jax.random.normal(
    jax.random.PRNGKey(7), shape=(32, 64, 64, 1)
)
result = transform({"image": image})
```
```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from medicai.transforms import Flip, RandomChoice, ShiftIntensity

transform = RandomChoice(
    transforms=[
        Flip(keys=["image"], spatial_axis=0, input_layout="HWC"),
        Flip(keys=["image"], spatial_axis=1, input_layout="HWC"),
        ShiftIntensity(keys=["image"], offset=0.1, input_layout="HWC"),
    ],
    num_choices=(1, 2),
    weights=[1.0, 1.0, 0.5],
)

image = keras.random.normal((64, 64, 1), seed=7)
result = transform({"image": image})
```

## Models

Inspect the registered model zoo:

```python
import medicai
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
- Input pipelines built with a backend-compatible dataloader. See the
  compatibility table in [Transformations](#transformations).

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

### GPU-side augmentation

Many of the batch-level `medicai.transforms` can run inside the model's training
step so augmentation is performed on the selected GPU instead of inside
the dataloader. The dataloader should return unaugmented batches in `B[D]HWC` layout.

```python
import keras
from medicai.transforms import Compose, RandomElasticTransform

augmentation = Compose(
    [
         RandomElasticTransform(
            keys=["image"],
            input_layout="BHWC",
            alpha=4.0,
            sigma=6.0,
            control_grid_spacing=(16, 16),
            field_interpolation="bilinear",
        )
    ]
)


class GPUAugmentedModel(keras.Model):
    def __init__(self, backbone, augmentation):
        super().__init__()
        self.backbone = backbone
        self.augmentation = augmentation

    def call(self, inputs, training=None):
        return self.backbone(inputs, training=training)

    def train_step(self, *args, **kwargs):
        if keras.config.backend() == "jax":
            return self._jax_train_step(*args, **kwargs)
        elif keras.config.backend() == "tensorflow":
            return self._tensorflow_train_step(*args, **kwargs)
        elif keras.config.backend() == "torch":
            return self._torch_train_step(*args, **kwargs)

    def _jax_train_step(self, state, data):
        images, labels = data
        images = self.augmentation({"image": images})["image"]
        return super().train_step(state, (images, labels))

    def _tensorflow_train_step(self, data):
        images, labels = data
        images = self.augmentation({"image": images})["image"]
        return super().train_step((images, labels))

    def _torch_train_step(self, data):
        images, labels = data
        images = self.augmentation({"image": images})["image"]
        return super().train_step((images, labels))


with keras.device("gpu:0"):
    augmented_model = GPUAugmentedModel(model, augmentation)
    augmented_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

augmented_model.fit(train_dataset, epochs=10)
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

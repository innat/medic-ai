---
orphan: true
---

# Quick starter

Create a segmentation model with a registered encoder:

```python
from medicai.models import UNet

model = UNet(
    input_shape=(96, 96, 96, 1),
    encoder_name="densenet121",
    num_classes=1,
)
```

Compose a simple 3D preprocessing pipeline:

```python
from medicai.transforms import Compose, NormalizeIntensity, RandFlip, Resize

transforms = Compose(
    [
        NormalizeIntensity(keys=["image"]),
        RandFlip(keys=["image", "label"], prob=0.5, spatial_axis=0),
        Resize(keys=["image", "label"], spatial_shape=(96, 96, 96)),
    ]
)
```

Inspect available model presets:

```python
from medicai.models import list_models

list_models()
```

```{eval-rst}
.. note::

   **Feature pyramid access**

   DenseNet models in ``medic-ai`` expose intermediate multi-scale features
   through ``model.pyramid_outputs``. The keys usually follow ``P1``, ``P2``,
   and deeper levels, where shallower features have higher spatial resolution
   and deeper features carry stronger semantic abstraction.

   These feature maps are produced from intermediate stages of the Keras
   Functional graph. Since they remain symbolic ``KerasTensor`` objects, they
   can be reused to build larger end-to-end models such as segmentation,
   detection, or other dense prediction pipelines on top of the same encoder.

   Example::

      from medicai.models import DenseNetBackbone

      model = DenseNetBackbone(
        input_shape=(224, 224, 3),
        blocks=[6, 12, 24, 16],
        growth_rate=32,
        bn_size=4,
        compression=0.5,
        dropout_rate=0.0,
        name="densenet_backbone",
    )
      model.pyramid_outputs
    {
        "P1": <KerasTensor shape=(None, 112, 112, 64)>,
        "P2": <KerasTensor shape=(None, 56, 56, 256)>,
        "P3": <KerasTensor shape=(None, 28, 28, 512)>,
        "P4": <KerasTensor shape=(None, 14, 14, 1024)>,
        "P5": <KerasTensor shape=(None, 7, 7, 1024)>,
    }
```
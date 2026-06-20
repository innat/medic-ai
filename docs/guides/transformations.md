# Transformations

`medicai.transforms` are designed for medical imaging workflows and are implemented with TensorFlow-compatible operations so they can be used in `tf.data` pipelines.

Most transforms accept either a sample dictionary or a `TensorBundle` and
return a `TensorBundle`. The transformed tensors stay under the same keys, so
if we pass inputs such as `image` and `label`, we retrieve the outputs using
those same keys after each transform or composed pipeline.

The package is organized into three groups:

- `medicai.transforms.spatial`: spatial transforms such as `Resize`,
  `SpatialCrop`, `Flip`, and `Rotate90`
- `medicai.transforms.intensity`: intensity transforms such as
  `NormalizeIntensity`, `ScaleIntensityRange`, and `ShiftIntensity`
- `medicai.transforms.random`: random augmentations built on top of the
  deterministic kernels where appropriate

## Input Conventions

Medic-AI transforms operate on channel-last sample tensors:

- 2D samples use `(H, W, C)`
- 3D samples use `(D, H, W, C)`

Most transforms are intentionally 2D/3D agnostic and expect users to provide
rank-appropriate spatial arguments explicitly. For example, `Resize` requires
the caller to pass both `spatial_shape` and `mode`, rather than assuming a 2D
or 3D default.

Two spatial transforms are intentionally 3D-only:

- `Spacing`
- `Orientation`

These transforms expect volumetric tensors and will raise clear validation
errors when called with 2D inputs.

## Data Container

```{eval-rst}
.. autoclass:: medicai.transforms.TensorBundle
```

## Building Pipelines

`Compose` chains transforms sequentially. It accepts either a raw mapping or
an existing `TensorBundle`, converts NumPy arrays to TensorFlow tensors, and
passes the same container through the whole pipeline.

```python
from medicai.transforms import Compose, NormalizeIntensity, RandomFlip, Resize

pipeline = Compose(
    [
        NormalizeIntensity(keys=["image"], nonzero=True),
        RandomFlip(keys=["image", "label"], prob=0.5, spatial_axis=0),
        Resize(
            keys=["image", "label"],
            mode=("trilinear", "nearest"),
            spatial_shape=(64, 128, 128),
        ),
    ]
)
```

## Spatial Transforms

```{eval-rst}
.. autoclass:: medicai.transforms.SpatialCrop

.. autoclass:: medicai.transforms.Flip

.. autoclass:: medicai.transforms.Rotate90

.. autoclass:: medicai.transforms.Resize

.. autoclass:: medicai.transforms.Spacing

.. autoclass:: medicai.transforms.Orientation

.. autoclass:: medicai.transforms.CropForeground

.. autoclass:: medicai.transforms.RandomCropByPosNegLabel

.. autoclass:: medicai.transforms.RandomSpatialCrop
```

## Intensity Transforms

```{eval-rst}
.. autoclass:: medicai.transforms.NormalizeIntensity

.. autoclass:: medicai.transforms.ScaleIntensityRange

.. autoclass:: medicai.transforms.ShiftIntensity

.. autoclass:: medicai.transforms.SignalFillEmpty

.. autoclass:: medicai.transforms.RandomShiftIntensity

```

## Random Transforms

```{eval-rst}
.. autoclass:: medicai.transforms.RandomFlip

.. autoclass:: medicai.transforms.RandomRotate90

.. autoclass:: medicai.transforms.RandomSpatialCrop

.. autoclass:: medicai.transforms.RandomCropByPosNegLabel

.. autoclass:: medicai.transforms.RandomRotate

.. autoclass:: medicai.transforms.RandomCutOut
```

## Base Abstractions

The public base classes are also exported from `medicai.transforms` so custom
transforms can use the same container, trace, and composition behavior as the
built-in transforms.

```{eval-rst}
.. autoclass:: medicai.transforms.Transform

.. autoclass:: medicai.transforms.KeyedTransform

.. autoclass:: medicai.transforms.RandomTransform

.. autoclass:: medicai.transforms.InvertibleTransform

.. autoclass:: medicai.transforms.Compose
```

## Creating Custom Transforms

The usual extension patterns are:

- inherit from `Transform` when you need full control over the input bundle
- inherit from `KeyedTransform` when the transform applies to a known set of
  data keys such as `image` and `label`
- inherit from `RandomTransform` when the transform needs probabilistic
  application through `sample_should_apply()`
- mix in `InvertibleTransform` when the transform can record enough metadata
  to support `inverse()`

### Custom Deterministic Transform

Use `KeyedTransform` when the transform updates one or more known keys.

```python
import tensorflow as tf

from medicai.transforms import KeyedTransform, TensorBundle


class MultiplyIntensity(KeyedTransform):
    def __init__(self, keys, factor, allow_missing_keys=False):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.factor = factor

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present_keys = self.apply_to_present_keys(
            bundle,
            lambda tensor, _: tensor * tf.cast(self.factor, tensor.dtype),
        )
        bundle.push_transform(
            self.build_trace_entry(
                params={"keys": list(present_keys), "factor": self.factor},
                applied=True,
                random=False,
                invertible=False,
            )
        )
        return bundle
```

### Custom Random Transform

Use `RandomTransform` when the transform should be applied with probability
`prob`.

```python
import tensorflow as tf

from medicai.transforms import RandomTransform, TensorBundle


class RandomAddBias(RandomTransform):
    def __init__(self, prob=0.5, bias=0.1):
        super().__init__(prob=prob)
        self.bias = bias

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        should_apply = self.sample_should_apply()
        if "image" in bundle.data:
            image = bundle["image"]
            bundle.data["image"] = tf.cond(
                should_apply,
                lambda: image + tf.cast(self.bias, image.dtype),
                lambda: image,
            )
        self.record_random_transform(
            bundle,
            params={"keys": ["image"], "bias": self.bias},
            applied=should_apply,
        )
        return bundle
```

### Custom Invertible Transform

Use `InvertibleTransform` when the transform can restore the original sample
from trace metadata.

```python
import tensorflow as tf

from medicai.transforms import InvertibleTransform, KeyedTransform, TensorBundle


class AddConstant(KeyedTransform, InvertibleTransform):
    def __init__(self, keys, value, allow_missing_keys=False):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.value = value

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present_keys = self.apply_to_present_keys(
            bundle,
            lambda tensor, _: tensor + tf.cast(self.value, tensor.dtype),
        )
        self.record_transform(bundle, {"keys": list(present_keys), "value": self.value})
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        self.apply_to_present_keys(
            bundle,
            lambda tensor, _: tensor - tf.cast(self.value, tensor.dtype),
        )
        return bundle
```

### Design Guidelines

- keep tensors in channel-last layout
- prefer TensorFlow ops throughout the transform implementation
- make 2D/3D-capable transforms rank-agnostic instead of encoding 2D or 3D
  defaults
- reserve explicit rank checks for transforms that are intentionally 3D-only
- record trace entries for debugging, auditability, and inversion support when
  applicable

## Pipeline API

```{eval-rst}
.. autoclass:: medicai.transforms.Compose
```

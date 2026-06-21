# Transformations

`medicai.transforms` provides TensorFlow-native preprocessing and augmentation
utilities for medical imaging workflows. The transforms are designed to work
cleanly inside `tf.data` pipelines, and most of them are rank-agnostic across
2D and 3D channel-last tensors.

Most transforms accept either:

- a plain sample mapping such as `{"image": image, "label": label}`
- an existing `TensorBundle`

In both cases, the output is a `TensorBundle`, and transformed tensors remain
available under the same keys.

The package is organized into three practical groups:

- `medicai.transforms.spatial`
- `medicai.transforms.intensity`
- `medicai.transforms.random`

## Input Conventions

Medic-AI transforms operate on channel-last tensors:

- 2D tensors use `(H, W, C)`
- 3D tensors use `(D, H, W, C)`

Most transforms are intentionally 2D/3D agnostic, so callers should provide
rank-appropriate spatial arguments explicitly instead of relying on implicit
defaults.

Two spatial transforms are intentionally 3D-only:

- `Spacing`
- `Orientation`

These expect volumetric inputs and should reject 2D usage with a clear error.

## Spatial Transforms

Spatial transforms change geometry, layout, orientation, or spatial extent.
Most of them are designed to work for both 2D and 3D tensors as long as the
caller provides spatial arguments with the correct rank.

Common examples:

- `SpatialCrop` for extracting a fixed region
- `Flip` and `Rotate90` for deterministic spatial reordering
- `Resize` for resampling to a target spatial shape
- `CropForeground` for foreground-aware cropping
- `Spacing` and `Orientation` for 3D spatial metadata-aware transforms

```{eval-rst}
.. autoclass:: medicai.transforms.SpatialCrop

.. autoclass:: medicai.transforms.Flip

.. autoclass:: medicai.transforms.Rotate90

.. autoclass:: medicai.transforms.Resize

.. autoclass:: medicai.transforms.Spacing

.. autoclass:: medicai.transforms.Orientation

.. autoclass:: medicai.transforms.CropForeground
```

## Intensity Transforms

Intensity transforms adjust voxel or pixel values without changing spatial
layout.

Common examples:

- `NormalizeIntensity` for mean/std normalization
- `ScaleIntensityRange` for mapping one range into another
- `ShiftIntensity` for additive offsets
- `SignalFillEmpty` for handling invalid values such as `NaN` and `Inf`

```{eval-rst}
.. autoclass:: medicai.transforms.NormalizeIntensity

.. autoclass:: medicai.transforms.ScaleIntensityRange

.. autoclass:: medicai.transforms.ShiftIntensity

.. autoclass:: medicai.transforms.SignalFillEmpty
```

## Random Transforms

Random transforms introduce stochastic augmentation. In the current API, these
are public transform classes built on top of TensorFlow random ops and, where
appropriate, deterministic kernels such as `Flip`, `Rotate90`, or
`ShiftIntensity`.

Common examples:

- `RandomFlip`
- `RandomRotate90`
- `RandomRotate`
- `RandomShiftIntensity`
- `RandomSpatialCrop`
- `RandomCropByPosNegLabel`
- `RandomCutOut`

```{eval-rst}
.. autoclass:: medicai.transforms.RandomFlip

.. autoclass:: medicai.transforms.RandomRotate90

.. autoclass:: medicai.transforms.RandomSpatialCrop

.. autoclass:: medicai.transforms.RandomCropByPosNegLabel

.. autoclass:: medicai.transforms.RandomRotate

.. autoclass:: medicai.transforms.RandomCutOut

.. autoclass:: medicai.transforms.RandomShiftIntensity
```

## LambdaTransform

`LambdaTransform` is the lightest way to create custom transforms without
writing a full subclass. It is useful when a simple callable is enough, but we
still want to keep Medic-AI features such as:

- keyed execution
- optional probability-based application
- optional inverse behavior
- metadata update hooks
- transform trace recording

Example:

```python
import tensorflow as tf

from medicai.transforms import LambdaTransform, TensorBundle

transform = LambdaTransform(
    keys=["image"],
    fn=lambda tensor: tensor + 1.0,
    inverse_fn=lambda tensor: tensor - 1.0,
    trace_params={"description": "add constant"},
    name="AddOne",
)

image = tf.ones((64, 64, 1), dtype=tf.float32)
bundle = TensorBundle({"image": image})

forward = transform(bundle)
restored = transform.inverse(forward)
```

```{eval-rst}
.. autoclass:: medicai.transforms.LambdaTransform
```

## TensorBundle

`TensorBundle` is the internal execution container used by
`medicai.transforms`. It keeps transform inputs and metadata together in one
object:

- `bundle.data` stores tensors such as `image`, `label`, or `mask`
- `bundle.meta` stores side information such as `affine` or applied-transform
  trace entries

Users do not always need to instantiate it directly, because `Transform.__call__`
will convert a plain mapping into a `TensorBundle`. Still, it is useful when we
want to pass metadata explicitly or inspect trace history between steps.

Example:

```python
import tensorflow as tf

from medicai.transforms import TensorBundle

image = tf.random.normal((32, 32, 1))
label = tf.zeros((32, 32, 1), dtype=tf.float32)

bundle = TensorBundle(
    {"image": image, "label": label},
    {"affine": tf.eye(4)},
)
```

```{eval-rst}
.. autoclass:: medicai.transforms.TensorBundle
```

## Compose

`Compose` builds transform pipelines by chaining transforms sequentially. It
accepts either a plain mapping or a `TensorBundle`, normalizes the input, and
passes the same container through each transform.

Example:

```python
from medicai.transforms import (
    Compose,
    NormalizeIntensity,
    RandomFlip,
    Resize,
)

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

```{eval-rst}
.. autoclass:: medicai.transforms.Compose
```

## Transform

`Transform` is the root abstraction. It is the right base class when a custom
transform needs full control over the whole `TensorBundle`, including both
tensor data and metadata.

Typical use cases:

- transforms that need to inspect or update `bundle.meta`
- transforms that do not naturally map to a fixed list of tensor keys
- wrapper transforms that orchestrate more than one internal operation

Example custom transform:

```python
import tensorflow as tf

from medicai.transforms import TensorBundle, Transform


class AttachSpacingFlag(Transform):
    def apply(self, bundle: TensorBundle) -> TensorBundle:
        bundle.meta["has_affine"] = "affine" in bundle.meta
        if "image" in bundle.data:
            image = bundle["image"]
            bundle.data["image"] = tf.identity(image)
        return bundle
```

```{eval-rst}
.. autoclass:: medicai.transforms.Transform
```

## KeyedTransform

`KeyedTransform` is the most common base class for deterministic user-defined
transforms. It is intended for transforms that operate on a known list of keys
such as `image`, `label`, or `mask`.

It provides:

- `self.keys`
- `allow_missing_keys`
- `iter_present_keys()`
- `apply_to_present_keys()`

Example custom transform:

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

```{eval-rst}
.. autoclass:: medicai.transforms.KeyedTransform
```

## RandomTransform

`RandomTransform` is the base class for transforms with probabilistic
application. It mainly provides `prob`, `sample_should_apply()`, and
`record_random_transform()`.

Use it when:

- the transform should apply only with some probability
- the decision should stay inside TensorFlow ops for `tf.data` compatibility
- we want standard random-trace recording

Example custom transform:

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

```{eval-rst}
.. autoclass:: medicai.transforms.RandomTransform
```

## InvertibleTransform

`InvertibleTransform` marks transforms that can reverse their forward effect.
This is useful for workflows where preprocessing must later be undone, such as
post-processing predictions back into the original spatial frame.

In practice, invertible custom transforms usually:

- inherit from `KeyedTransform` and `InvertibleTransform`
- record enough information during `apply()`
- implement `inverse()` using either stored state or bundle trace metadata

Example custom transform:

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

```{eval-rst}
.. autoclass:: medicai.transforms.InvertibleTransform
```

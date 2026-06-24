# Transformations

`medicai.transforms` provides TensorFlow-native preprocessing and augmentation
utilities for medical imaging workflows. The transforms are designed to work
cleanly inside `tf.data.Dataset` pipelines, or, ``keras.utils.PyDataset`` and, ``torch.utils.data.Dataset`` by converting samples to ``numpy``. Most of transformation are rank-agnostic across 2D and 3D channel-last tensors.

Most transforms accept either:

- a plain sample mapping such as `{"image": image, "label": label}`
- an existing :class:`~medicai.transforms.TensorBundle`

In both cases, the output is a `TensorBundle`, and transformed tensors remain
available under the same keys.

## Input Conventions

``medicai`` transforms operate on channel-last tensors:

- 2D tensors use `(H, W, C)`
- 3D tensors use `(D, H, W, C)`

Most transforms are intentionally 2D/3D agnostic, so callers should provide
rank-appropriate spatial arguments explicitly instead of relying on implicit
defaults.

```{note}
Two spatial transforms are intentionally 3D-only:

- `Spacing`
- `Orientation`
```

## Spatial

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

## Intensity

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

Random transforms introduce stochastic augmentation.

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

## Compose

```{eval-rst}
.. autoclass:: medicai.transforms.Compose
```

## Custom Transforms

The base abstractions below are most useful when implementing custom
transforms or understanding how ``medicai`` pipelines are structured internally.

## LambdaTransform

```{eval-rst}
.. autoclass:: medicai.transforms.LambdaTransform
```

## Transform

```{eval-rst}
.. autoclass:: medicai.transforms.Transform
   :members: apply, inverse, build_trace_entry
```

## KeyedTransform

```{eval-rst}
.. autoclass:: medicai.transforms.KeyedTransform
   :members: apply_to_present_keys, iter_present_keys
```

## RandomTransform

```{eval-rst}
.. autoclass:: medicai.transforms.RandomTransform
   :members: sample_should_apply, record_random_transform
```

## InvertibleTransform

```{eval-rst}
.. autoclass:: medicai.transforms.InvertibleTransform
   :members: record_transform, inverse
```

## Advanced: TensorBundle

`TensorBundle` is the internal execution container used by
`medicai.transforms`. You usually do not need to create it directly unless
you are working with metadata, inversion, or custom transforms.

```{eval-rst}
.. autoclass:: medicai.transforms.TensorBundle
```

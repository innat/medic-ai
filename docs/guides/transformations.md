# Transformations

`medicai.transforms` provides TensorFlow-native preprocessing and augmentation
utilities for medical imaging workflows. The transforms are designed to work
cleanly inside `tf.data.Dataset` pipelines. They can also be used from
`keras.utils.PyDataset` or `torch.utils.data.Dataset` by converting returned
tensors to `numpy` when needed. Most transforms are rank-agnostic across 2D
and 3D channel-last tensors.

Most transforms accept either:

- a plain sample mapping such as `{"image": image, "label": label}`
- an existing `TensorBundle`

In both cases, the output is a `TensorBundle`, and transformed tensors remain
available under the same keys.

## Input Conventions

``medicai`` transforms operate on channel-last tensors:

- 2D tensors use `(H, W, C)`
- 3D tensors use `(D, H, W, C)`
- batched 2D tensors use `(B, H, W, C)`
- batched 3D tensors use `(B, D, H, W, C)`

The new transform API makes execution mode explicit through `input_mode`:

- `input_mode="sample"` means the transform expects one sample tensor at a time
- `input_mode="batch"` means the transform expects one batched tensor bundle

Not every transform supports both modes.

## Capability Split

### Sample-only transforms

These transforms operate on one sample at a time because they depend on
sample-specific metadata or sample-specific spatial decisions:

- `CropForeground`
- `Spacing`
- `Orientation`

They accept `input_mode="sample"` only and raise a clear error if
`input_mode="batch"` is requested.

### Dual-mode transforms

These transforms can operate on either one sample or one already-batched
tensor bundle, depending on `input_mode`:

- `Flip`
- `Rotate90`
- `Resize`
- `SpatialCrop`
- `RandomFlip`
- `RandomRotate90`
- `RandomShiftIntensity`
- `RandomSpatialCrop`
- `RandomCropByPosNegLabel`

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

For dual-mode spatial transforms, the same class can be used either:

- in dataloaders with `input_mode="sample"`
- on already batched tensors with `input_mode="batch"`

For metadata-aware transforms such as `Spacing` and `Orientation`, keep them in
sample pipelines because affine metadata is tracked per sample.

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

## Random

Random transforms introduce stochastic augmentation.

All random public transforms inherit the shared `RandomTransform` seed
contract. The `seed` argument accepts:

- `None` for ordinary non-deterministic randomness
- an integer seed for reproducible replay
- `keras.random.SeedGenerator` for stateful seeded sampling

For the currently migrated dual-mode random transforms, when
`input_mode="batch"` is used, one random decision or one sampled parameter set
is shared across the whole incoming batch tensor. This keeps inversion and
trace behavior simple and predictable.

Common examples:

- `RandomFlip`
- `RandomRotate90`
- `RandomRotate`
- `RandomShiftIntensity`
- `RandomSpatialCrop`
- `RandomCropByPosNegLabel`
- `RandomCutOut`
- `RandomChoice`

```{eval-rst}
.. autoclass:: medicai.transforms.RandomFlip

.. autoclass:: medicai.transforms.RandomRotate90

.. autoclass:: medicai.transforms.RandomSpatialCrop

.. autoclass:: medicai.transforms.RandomCropByPosNegLabel

.. autoclass:: medicai.transforms.RandomRotate

.. autoclass:: medicai.transforms.RandomCutOut

.. autoclass:: medicai.transforms.RandomShiftIntensity

.. autoclass:: medicai.transforms.RandomChoice
```

## Compose

`Compose` chains transforms in order and returns one final `TensorBundle`.
This is the usual way to define a preprocessing or augmentation pipeline for a
dataset loader.

```{eval-rst}
.. autoclass:: medicai.transforms.Compose
```

## Custom Transforms

The APIs below are most useful when building custom transforms or
understanding how `medicai.transforms` pipelines are structured internally.

### LambdaTransform

```{eval-rst}
.. autoclass:: medicai.transforms.LambdaTransform
```

### Transform

```{eval-rst}
.. autoclass:: medicai.transforms.Transform
   :members: apply, inverse, build_trace_entry
```

### KeyedTransform

```{eval-rst}
.. autoclass:: medicai.transforms.KeyedTransform
   :members: apply_to_present_keys, iter_present_keys
```

### RandomTransform

```{eval-rst}
.. autoclass:: medicai.transforms.RandomTransform
   :members: sample_should_apply, record_random_transform
```

### InvertibleTransform

```{eval-rst}
.. autoclass:: medicai.transforms.InvertibleTransform
   :members: record_transform, inverse
```

### Advanced: TensorBundle

`TensorBundle` is the internal execution container used by
`medicai.transforms`. You usually do not need to create it directly unless
you are working with metadata, inversion, or custom transforms.

```{eval-rst}
.. autoclass:: medicai.transforms.TensorBundle
```

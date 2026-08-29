# Transformations

`medicai.transforms` provides native, multi-backend preprocessing and
augmentation utilities for medical imaging workflows. The transforms are
designed to integrate cleanly with `pygrain`, `torch.utils.data.Dataset`,
`tf.data.Dataset`, and `keras.utils.PyDataset`.

**Input Conventions**

``medicai`` transforms use **channel-last** tensors:

- 2D tensors use `(H, W, C)`
- 3D tensors use `(D, H, W, C)`
- batched 2D tensors use `(B, H, W, C)`
- batched 3D tensors use `(B, D, H, W, C)`

The transform API uses `input_layout` to make the execution mode explicit:

- sample layouts (`"HWC"`, `"DHWC"`) indicate that the transform expects one sample tensor at a time
- batch layouts (`"BHWC"`, `"BDHWC"`) indicate that the transform expects one batched tensor bundle


## Capability Split

**Sample-only transforms**

These transforms process one sample at a time because they depend on
sample-specific metadata or spatial decisions:

- `CropForeground`
- `Spacing`
- `Orientation`

They support only sample layouts (`"HWC"` or `"DHWC"`).

**Dual-mode transforms**

These transforms can process either one sample or an already-batched tensor,
depending on `input_layout`:

- `Flip`
- `Rotate90`
- `Resize`
- `SpatialCrop`
- `RandomFlip`
- `RandomRotate90`
- `RandomRotate`
- `RandomShiftIntensity`
- `RandomSpatialCrop`
- `RandomCropByPosNegLabel`
- `RandomCutOut`

They support both sample layouts (`"HWC"` or `"DHWC"`) and batch layout (`"BHWC"` or `"BDHWC"`). Callers should therefore provide spatial arguments appropriate to the input rank instead of relying on implicit defaults.

```{note}
Two spatial transforms are intentionally sample-level 3D-only. The do not support 2D input or batch support:

- `Spacing`
- `Orientation`
```

## Spatial

Spatial transforms modify geometry, layout, orientation, or spatial extent.
Most support both 2D and 3D tensors when the caller provides arguments with the
appropriate rank. So, the same class can be used in either of these
contexts:

- in dataloaders with sample layouts such as `"HWC"` or `"DHWC"`
- on already batched tensors with batch layouts such as `"BHWC"` or `"BDHWC"`


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

Intensity transforms modify voxel or pixel values without changing the spatial
layout. Most support both 2D and 3D tensors when the caller provides arguments with the appropriate rank. So, the same class can be used in either of these
contexts:

- in dataloaders with sample layouts such as `"HWC"` or `"DHWC"`
- on already batched tensors with batch layouts such as `"BHWC"` or `"BDHWC"`

```{eval-rst}
.. autoclass:: medicai.transforms.NormalizeIntensity

.. autoclass:: medicai.transforms.ScaleIntensityRange

.. autoclass:: medicai.transforms.ShiftIntensity

.. autoclass:: medicai.transforms.SignalFillEmpty
```

## Random

Random transforms provide stochastic augmentation. Most support both 2D and 3D tensors when the caller provides arguments with the appropriate rank. So, the same class can be used in either of these contexts:

- in dataloaders with sample layouts such as `"HWC"` or `"DHWC"`
- on already batched tensors with batch layouts such as `"BHWC"` or `"BDHWC"`

All public random transforms inherit the shared `RandomTransform` seed contract. The `seed` argument accepts:

- `None` for ordinary non-deterministic randomness
- an integer seed for reproducible replay
- `keras.random.SeedGenerator` for stateful seeded sampling

For currently dual-mode random transforms, a batch layout causes one random
decision or sampled parameter set to be shared across the entire incoming batch
tensor. This keeps inversion and trace behavior simple and predictable.


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

`Compose` applies transforms sequentially. It is the usual way to define a
preprocessing or augmentation pipeline for a dataset loader.

```{eval-rst}
.. autoclass:: medicai.transforms.Compose
```

## Custom Transforms

The APIs below are primarily useful when creating custom transforms or learning
how `medicai.transforms` pipelines are structured internally.

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

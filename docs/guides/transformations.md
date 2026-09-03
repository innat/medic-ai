# Transformations

`medicai.transforms` provides native, multi-backend preprocessing and
augmentation utilities for medical imaging workflows. The transforms are
designed to integrate cleanly with `pygrain`, `torch.utils.data.Dataset`,
`tf.data.Dataset`, and `keras.utils.PyDataset`.

Choose the right dataloader for the target Keras backend.

| Keras backend | PyGrain | `torch.utils.data` | `tf.data` | `keras.utils.PyDataset` |
| :--- | :---: | :---: | :---: | :---: |
| TensorFlow | ✓ | ✗ | ✓ | ✓ |
| Torch | ✓ | ✓ | ✗ | ✓ |
| JAX | ✓ | ✗ | ✗ | ✓ |

When `torch` is the active backend, `medicai.transforms` use Torch-backed
Keras operations. The same applies to the `tensorflow` and `jax` backends.

If you want a common dataloader that supports all backends out of the box,
the recommended option is **PyGrain**, which provides efficient parallel data
loading and multithreading and multiprocessing worker support.


**Input Conventions**

``medicai`` transforms use **channel-last** tensors and provide the
`input_layout` argument to make the execution mode explicit:

- single 2D tensors use: `input_layout="HWC"`
- single 3D tensors use: `input_layout="DHWC"`
- batched 2D tensors use: `input_layout="BHWC"`
- batched 3D tensors use: `input_layout="BDHWC"`


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

They support both sample layouts (`"HWC"` or `"DHWC"`) and batch layouts
(`"BHWC"` or `"BDHWC"`). Callers should therefore provide spatial arguments
appropriate to the input rank instead of relying on implicit defaults.

**Sample-only transforms**

These transforms process one sample at a time because they depend on
sample-specific metadata or spatial decisions:

- `CropForeground`
- `Spacing`
- `Orientation`

They support only sample layouts (`"HWC"` or `"DHWC"`).

```{note}
Two spatial transforms are intentionally sample-level and 3D-only. They do not
support 2D input or batched input:

- `Spacing`
- `Orientation`
```

## Spatial

Spatial transforms modify geometry, layout, orientation, or spatial extent.
Most support both `2D` and `3D` tensors when the caller provides arguments with the
appropriate rank. Therefore, the same class can be used in either of these
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
layout. Most support both `2D` and `3D` tensors when the caller provides arguments
with the appropriate rank. Therefore, the same class can be used in either of these
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

Random transforms provide stochastic augmentation. Most support both `2D` and `3D`
tensors when the caller provides arguments with the appropriate rank.
Therefore, the same class can be used in either of these contexts:

- in dataloaders with sample layouts such as `"HWC"` or `"DHWC"`
- on already batched tensors with batch layouts such as `"BHWC"` or `"BDHWC"`

All public random transforms inherit the shared `RandomTransform` seed contract. The `seed` argument accepts:

- `None` for ordinary non-deterministic randomness
- an integer seed for reproducible replay
- `keras.random.SeedGenerator` for stateful seeded sampling

```{note}

For currently dual-mode random transforms, using a batch layout causes one
random decision or sampled parameter set to be shared across the entire
incoming batch tensor. This keeps inversion and trace behavior simple and
predictable.
```


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

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
the recommended option is [**PyGrain**](https://google-grain.readthedocs.io/en/latest/), which provides efficient parallel data
loading and multithreading and multiprocessing worker support.


**Overview**

The ``medicai`` transforms use **channel-last** tensors and provide the
`input_layout` argument to make the execution mode explicit:

- single 2D tensors use: `input_layout="HWC"`
- single 3D tensors use: `input_layout="DHWC"`
- batched 2D tensors use: `input_layout="BHWC"`
- batched 3D tensors use: `input_layout="BDHWC"`

The transforms below support eager execution on both CPU and GPU when the
selected Keras backend and device provide the required operations. This table
describes device support, not XLA or compiled-mode compatibility. Refer to
the transform docstrings and the [recorded benchmark report](../misc/transform.md)
for backend-specific compilation limitations and measurements.

| Transform | Supported Layout | CPU | GPU |
| :--- | :--- | :---: | :---: |
| `CropForeground` | `HWC`, `DHWC` | ✓ | ✓ |
| `Orientation` | `DHWC` | ✓ | ✓ |
| `Spacing` | `DHWC` | ✓ | ✓ |
| `NormalizeIntensity` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `ScaleIntensityRange` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `ShiftIntensity` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `SignalFillEmpty` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `Flip` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `Rotate90` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `Resize` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `SpatialCrop` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `RandomCropByPosNegLabel` | `HWC`, `DHWC` | ✓ | ✓ |
| `RandomFlip` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `RandomRotate90` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `RandomRotate` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `RandomShiftIntensity` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `RandomSpatialCrop` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `RandomTranslate` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `RandomScale` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `RandomShear` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `RandomAffine` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `RandomCutOut` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |
| `RandomElasticTransform` | `HWC`, `DHWC`, `BHWC`, `BDHWC` | ✓ | ✓ |

`Spacing` and `Orientation` are intentionally restricted to 3D sample
layouts because they require sample-specific spatial metadata. The other
sample-only transforms in the table do not currently support batched layouts.
Callers should provide spatial arguments appropriate to the input rank instead
of relying on implicit defaults.

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

.. autoclass:: medicai.transforms.Pad

.. autoclass:: medicai.transforms.PadIfNeeded
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

```{eval-rst}
.. autoclass:: medicai.transforms.RandomFlip

.. autoclass:: medicai.transforms.RandomRotate

.. autoclass:: medicai.transforms.RandomRotate90

.. autoclass:: medicai.transforms.RandomTranslate

.. autoclass:: medicai.transforms.RandomScale

.. autoclass:: medicai.transforms.RandomShear

.. autoclass:: medicai.transforms.RandomAffine

.. autoclass:: medicai.transforms.RandomSpatialCrop

.. autoclass:: medicai.transforms.RandomCropByPosNegLabel

.. autoclass:: medicai.transforms.RandomElasticTransform

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

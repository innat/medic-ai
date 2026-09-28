# JIT Compiled Dataloader

This example measures whether compiling a batch-level augmentation pipeline improves
steady-state dataloader throughput. It uses a small custom Python generator rather
than a framework-specific input API, so the comparison focuses on ``medicai`` transforms.

The example uses:

- Dummy 2D segmentation samples.
- `BHWC` batches with shape `(16, 256, 256, 1)`.
- `RandomFlip`, `RandomAffine`, and `RandomElasticTransform`.
- A comparison of eager and compiled modes under the same transform configuration.

```{eval-rst}
.. note::

Before choosing transforms, check the benchmark documentation for
your Keras backend. It reports which transforms support compiled execution and shows the measured eager-versus-compiled performance, so you can select a pipeline that is compatible with your backend and worthwhile for your workload.
```

## Why Enable JIT in the Dataloader?

Dataloader-side JIT is intended to reduce augmentation and preprocessing time by
compiling the transform pipeline. Its purpose is throughput, not a change to the
augmentation semantics. Compilation is not guaranteed to improve every backend,
shape, device, or transform combination, so eager and compiled execution should
be benchmarked for the intended workload.

This example uses 2D `BHWC` batches, but the same approach applies to 3D
`BDHWC` batches when the selected transforms support the 3D layout and backend.
It also applies to sample-level `HWC` and `DHWC` inputs when transforms run before
batching. Use the layout that matches where the pipeline executes: sample-level
preprocessing before batching, or batch-level augmentation after batching.

Before enabling compilation, keep the following in mind:

- Keep rank, batch shape, spatial shape, dtype, and device stable to avoid repeated compilation.
- Use tensor-only inputs with empty metadata in a compiled `Compose` pipeline.
- Use persistent random generators for random transforms, and verify JIT behavior with the selected backend.
- Call `warmup()` with a representative batch before measuring steady-state throughput.
- Compilation may fail for unsupported operations, and `inverse()` is unavailable for compiled pipelines.
- A compiled pipeline can be slower than eager execution for small or inexpensive transforms.

## Dataloader Choice

The custom Python generator below keeps the example short and makes the timing
focus clear. It is not a required dataloader implementation. In a real project,
use the input API that best fits your selected backend, such as `tf.data` for
TensorFlow, `torch.utils.data` for Torch, or PyGrain, or
`keras.utils.PyDataset` for backend-agnostic workflows.

## Imports and Configuration

```python
import sys
import time

import keras
import numpy as np
from keras import ops
from tqdm import tqdm

from medicai.transforms import (
    Compose,
    RandomAffine,
    RandomElasticTransform,
    RandomFlip,
)

IMAGE_SIZE = 256
BATCH_SIZE = 16
NUM_SAMPLES = BATCH_SIZE * 4
WARMUP_BATCHES = 3
MEASURED_BATCHES = 20
```

## Dummy Segmentation Data

The labels contain a simple foreground square so the geometric transforms operate
on an image/mask pair instead of empty tensors.

```python
rng = np.random.default_rng(7)
images = rng.random(
    (NUM_SAMPLES, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32
)
labels = np.zeros(
    (NUM_SAMPLES, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32
)
labels[:, 64:192, 64:192, 0] = 1.0
```

## Transform Pipelines

The random transforms use persistent generators so repeated calls receive new
random parameters. `Compose(jit_compile=True)` compiles the complete forward
pipeline for the selected backend.

```python
def build_pipeline(jit_compile):
    return Compose(
        [
            RandomFlip(
                keys=["image", "label"],
                spatial_axis=[1, 2],
                prob=1.0,
                input_layout="BHWC",
                seed=keras.random.SeedGenerator(11),
            ),
            RandomAffine(
                keys=["image", "label"],
                rotation_factor=0.08,
                scale_factor=0.05,
                translation_factor=0.04,
                shear_factor=0.02,
                interpolation={"image": "bilinear", "label": "nearest"},
                fill_mode="constant",
                fill_value={"image": 0.0, "label": 0.0},
                prob=1.0,
                input_layout="BHWC",
                seed=keras.random.SeedGenerator(13),
            ),
            RandomElasticTransform(
                keys=["image", "label"],
                alpha=2.0,
                sigma=4.0,
                control_grid_spacing=(16, 16),
                interpolation={"image": "bilinear", "label": "nearest"},
                fill_mode="constant",
                fill_value=0.0,
                prob=1.0,
                input_layout="BHWC",
                seed=keras.random.SeedGenerator(17),
            ),
        ],
        jit_compile=jit_compile,
    )
```

## Custom Batch Generator

The generator yields fixed-shape batches indefinitely. Keeping the shape stable
avoids measuring repeated backend recompilation caused by changing input signatures.

```python
def batch_generator():
    while True:
        for start in range(0, NUM_SAMPLES, BATCH_SIZE):
            end = start + BATCH_SIZE
            yield images[start:end], labels[start:end]


def apply_pipeline(batch, pipeline):
    image_batch, label_batch = batch
    result = pipeline(
        {
            "image": ops.convert_to_tensor(image_batch),
            "label": ops.convert_to_tensor(label_batch),
        }
    )
    # Materialize outputs so asynchronous device work is included in timing.
    return (
        ops.convert_to_numpy(result["image"]),
        ops.convert_to_numpy(result["label"]),
    )
```

## Benchmark Function

The JIT pipeline is warmed up before timing begins. Compilation time is therefore not
included in the reported steady-state measurement.

```python
PROGRESS_FORMAT = "{desc}: {n_fmt}/{total_fmt} |{bar:12}| elapsed={elapsed} | {rate_fmt}"

def benchmark(jit_compile):
    pipeline = build_pipeline(jit_compile)

    if jit_compile:
        pipeline.warmup(
            {
                "image": ops.convert_to_tensor(images[:BATCH_SIZE]),
                "label": ops.convert_to_tensor(labels[:BATCH_SIZE]),
            }
        )

    iterator = batch_generator()

    for _ in tqdm(
        range(WARMUP_BATCHES),
        desc=f"warmup ({'jit' if jit_compile else 'eager'})",
        bar_format=PROGRESS_FORMAT,
        file=sys.stdout,
        colour="green",
    ):
        apply_pipeline(next(iterator), pipeline)

    start = time.perf_counter()
    for _ in tqdm(
        range(MEASURED_BATCHES),
        desc=f"measure ({'jit' if jit_compile else 'eager'})",
        bar_format=PROGRESS_FORMAT,
        file=sys.stdout,
        colour="green",
    ):
        apply_pipeline(next(iterator), pipeline)

    return (time.perf_counter() - start) * 1000.0 / MEASURED_BATCHES
```

## Run and Compare

```python
backend = keras.config.backend()
print(f"backend={backend}")
print(f"batch_shape=({BATCH_SIZE}, {IMAGE_SIZE}, {IMAGE_SIZE}, 1)")

eager_ms = benchmark(jit_compile=False)
print(f"eager_ms_per_batch={eager_ms:.2f}")

jit_ms = benchmark(jit_compile=True)
print(f"jit_ms_per_batch={jit_ms:.2f}")

print(f"speedup={eager_ms / jit_ms:.2f}x")
```
```bash
backend=jax
batch_shape=(16, 256, 256, 1)
warmup (eager): 3/3 |████████████| elapsed=00:00 |  6.00it/s
measure (eager): 20/20 |████████████| elapsed=00:03 |  5.82it/s
eager_ms_per_batch=171.88
warmup (jit): 3/3 |████████████| elapsed=00:00 | 75.26it/s
measure (jit): 20/20 |████████████| elapsed=00:00 | 74.26it/s
jit_ms_per_batch=13.59
speedup=12.65x
```

This run shows an approximately `13x` steady-state speedup after compilation.

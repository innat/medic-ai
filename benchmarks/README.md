# Transform Benchmarks

These scripts measure `medicai` transforms outside the test suite. They compare
dataloader-style CPU execution with tensor-only transforms that can also run
inside a model or GPU training step.

## Organization

The benchmark currently focuses only on transforms, but its internal layout is
kept extensible for future benchmark suites:

```text
benchmarks/
├── transforms.py                 # Stable CLI entry point
├── common/
│   ├── devices.py                # Backend-specific device discovery
│   ├── timing.py                 # Synchronization helpers
│   ├── compilation.py            # Backend compilation adapters
│   └── reporting.py              # Console result formatting
└── transform_benchmark/
    ├── cases.py                  # Synthetic 2D/3D input generation
    ├── runner.py                 # Transform profiling workflow
    └── specs.py                  # Transform benchmark definitions
```

## How to run

Set the Keras backend before starting Python:

```bash
KERAS_BACKEND=tensorflow python benchmarks/transforms.py --device cpu
KERAS_BACKEND=tensorflow python benchmarks/transforms.py --device both
KERAS_BACKEND=torch python benchmarks/transforms.py --device gpu

# Compare eager execution with the active backend's compiled path.
KERAS_BACKEND=tensorflow python benchmarks/transforms.py --device gpu --compile xla
```

The registry uses two execution groups:

- `cpu`: transforms that depend on medical metadata or are normally applied
  before batching, such as `CropForeground`, `Orientation`, and `Spacing`.
- `cpu+gpu`: tensor-only transforms such as intensity, flip, resize, crop, and
  random augmentation transforms.

The runner separates warm-up from measured iterations, reuses one prebuilt
tensor case while creating a fresh bundle for every call, synchronizes backend
work before stopping the timer, and reports forward timings. Input-case setup
is reported separately as `case_setup_ms`; it is not included in transform
timings. The benchmark is a timing tool, not a correctness replacement for
`test/transforms/`.

`--compile none` is the default and measures eager transform calls. With
`--compile xla`, TensorFlow uses `tf.function(jit_compile=True)`, JAX uses
`jax.jit`, and Torch uses `torch.compile` with the Keras-standard `inductor`
backend with its default graph-break behavior. 

Compilation time is reported separately as `compile_time_ms`. Metadata-dependent transforms
are skipped because their Python-side metadata and dynamic geometry are not
part of this compiled tensor-only benchmark. If a transform uses an operation unsupported by the active backend's XLA compiler, it is recorded with `compile_status=not-xla-compatible`
and the remaining benchmark continues.

Example:

```bash
python benchmarks/transforms.py --group cpu+gpu --device both \
  --layout BDHWC --sizes 64 96 128 160 --batch-size 1 \
  --iterations 50 --warmup 10 --json /tmp/medicai.json
```

Common image-size profiles:

```bash
# 2D: (B, H, W, C), with H=W in each run.
python benchmarks/transforms.py --layout BHWC --sizes 224 512 1024 --batch-size 1

# 3D: (B, D, H, W, C), with D=H=W in each run.
python benchmarks/transforms.py --layout BDHWC --sizes 64 96 128 160 --batch-size 1
```

Use a smaller `--batch-size` or fewer sizes when measuring large 3D volumes;
memory use grows cubically with the 3D size.


## Recorded Results

The following results report forward median execution time in **milliseconds**. Each transformation has its own section, with separate CPU, GPU, and compiled GPU tables. Every row represents one concrete input shape and batch configuration. The fastest backend in each row is shown in **bold**; `--` means no matching result or unsupported execution.

Also the following benchmark is performed on Kaggle-Tesla T4 GPU environment.

### CropForeground

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| DHWC | (96, 96, 96, 1) | CropForeground | 17.62 | **8.39** | 312.19 |
| DHWC | (160, 160, 160, 1) | CropForeground | 25.38 | **24.59** | 379.00 |
| DHWC | (256, 256, 256, 1) | CropForeground | **120.97** | 191.29 | 610.93 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| DHWC | (96, 96, 96, 1) | CropForeground | 17.67 | **6.68** | 310.04 |
| DHWC | (160, 160, 160, 1) | CropForeground | 25.07 | **17.39** | 359.07 |
| DHWC | (256, 256, 256, 1) | CropForeground | **121.47** | 163.57 | 412.34 |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| DHWC | (96, 96, 96, 1) | CropForeground | -- | -- | -- |
| DHWC | (160, 160, 160, 1) | CropForeground | -- | -- | -- |
| DHWC | (256, 256, 256, 1) | CropForeground | -- | -- | -- |
### Flip

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | Flip | **1.24** | 1.41 | 2.24 |
| BHWC | (8, 224, 224, 1) | Flip | **1.54** | 2.24 | 2.98 |
| BHWC | (16, 224, 224, 1) | Flip | **2.35** | 4.30 | 4.76 |
| BHWC | (32, 224, 224, 1) | Flip | **4.36** | 8.67 | 8.96 |
| BHWC | (4, 512, 512, 1) | Flip | **2.89** | 5.38 | 5.58 |
| BHWC | (8, 512, 512, 1) | Flip | **6.25** | 10.87 | 15.59 |
| BHWC | (16, 512, 512, 1) | Flip | **12.04** | 21.48 | 30.87 |
| BHWC | (4, 1280, 1280, 1) | Flip | **17.85** | 33.27 | 46.95 |
| DHWC | (96, 96, 96, 1) | Flip | **2.58** | 4.59 | 5.08 |
| DHWC | (160, 160, 160, 1) | Flip | **11.65** | 21.00 | 30.02 |
| DHWC | (256, 256, 256, 1) | Flip | **128.54** | 188.40 | 240.11 |
| BDHWC | (1, 96, 96, 96, 1) | Flip | **2.55** | 4.57 | 5.05 |
| BDHWC | (2, 96, 96, 96, 1) | Flip | **4.68** | 9.54 | 10.23 |
| BDHWC | (1, 160, 160, 160, 1) | Flip | **11.83** | 20.82 | 29.62 |
| BDHWC | (1, 256, 256, 256, 1) | Flip | **129.87** | 187.19 | 242.28 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | Flip | 1.24 | **0.84** | 1.46 |
| BHWC | (8, 224, 224, 1) | Flip | 1.56 | **1.42** | 2.03 |
| BHWC | (16, 224, 224, 1) | Flip | 2.36 | **2.33** | 2.99 |
| BHWC | (32, 224, 224, 1) | Flip | **4.22** | 4.73 | 5.01 |
| BHWC | (4, 512, 512, 1) | Flip | **2.94** | 3.12 | 3.85 |
| BHWC | (8, 512, 512, 1) | Flip | **6.30** | 7.17 | 8.33 |
| BHWC | (16, 512, 512, 1) | Flip | **11.90** | 13.76 | 24.11 |
| BHWC | (4, 1280, 1280, 1) | Flip | **17.95** | 21.42 | 41.77 |
| DHWC | (96, 96, 96, 1) | Flip | **2.50** | 2.97 | 3.41 |
| DHWC | (160, 160, 160, 1) | Flip | **11.49** | 13.74 | 16.41 |
| DHWC | (256, 256, 256, 1) | Flip | 126.52 | 159.20 | **106.65** |
| BDHWC | (1, 96, 96, 96, 1) | Flip | **2.37** | 2.88 | 3.36 |
| BDHWC | (2, 96, 96, 96, 1) | Flip | **4.47** | 5.59 | 5.34 |
| BDHWC | (1, 160, 160, 160, 1) | Flip | **12.01** | 13.41 | 16.22 |
| BDHWC | (1, 256, 256, 256, 1) | Flip | 127.52 | 159.66 | **106.23** |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | Flip | 1.34 | **0.97** | 1.18 |
| BHWC | (8, 224, 224, 1) | Flip | 1.90 | **1.46** | 1.95 |
| BHWC | (16, 224, 224, 1) | Flip | 2.84 | **2.66** | 3.05 |
| BHWC | (32, 224, 224, 1) | Flip | 4.86 | 5.07 | **4.79** |
| BHWC | (4, 512, 512, 1) | Flip | 3.58 | **3.17** | 3.38 |
| BHWC | (8, 512, 512, 1) | Flip | 7.43 | **6.99** | 8.45 |
| BHWC | (16, 512, 512, 1) | Flip | **14.02** | 14.32 | 16.48 |
| BHWC | (4, 1280, 1280, 1) | Flip | **22.11** | 22.14 | 35.95 |
| DHWC | (96, 96, 96, 1) | Flip | 3.27 | **2.90** | 3.08 |
| DHWC | (160, 160, 160, 1) | Flip | 14.12 | **13.85** | 16.83 |
| DHWC | (256, 256, 256, 1) | Flip | 164.47 | 158.16 | **106.68** |
| BDHWC | (1, 96, 96, 96, 1) | Flip | 3.10 | **2.81** | 3.03 |
| BDHWC | (2, 96, 96, 96, 1) | Flip | 5.87 | 4.99 | **4.88** |
| BDHWC | (1, 160, 160, 160, 1) | Flip | 13.51 | **13.27** | 15.99 |
| BDHWC | (1, 256, 256, 256, 1) | Flip | 164.98 | 164.76 | **105.94** |
### NormalizeIntensity

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | NormalizeIntensity | 3.67 | **1.98** | 7.93 |
| BHWC | (8, 224, 224, 1) | NormalizeIntensity | 3.99 | **2.91** | 8.89 |
| BHWC | (16, 224, 224, 1) | NormalizeIntensity | **4.43** | 4.55 | 10.87 |
| BHWC | (32, 224, 224, 1) | NormalizeIntensity | **6.40** | 8.29 | 16.10 |
| BHWC | (4, 512, 512, 1) | NormalizeIntensity | **4.85** | 5.70 | 12.25 |
| BHWC | (8, 512, 512, 1) | NormalizeIntensity | **8.16** | 11.47 | 24.97 |
| BHWC | (16, 512, 512, 1) | NormalizeIntensity | **13.31** | 23.07 | 55.71 |
| BHWC | (4, 1280, 1280, 1) | NormalizeIntensity | **19.10** | 36.22 | 99.57 |
| DHWC | (96, 96, 96, 1) | NormalizeIntensity | **4.68** | 5.13 | 11.05 |
| DHWC | (160, 160, 160, 1) | NormalizeIntensity | **12.91** | 22.67 | 43.88 |
| DHWC | (256, 256, 256, 1) | NormalizeIntensity | **107.74** | 195.39 | 349.02 |
| BDHWC | (1, 96, 96, 96, 1) | NormalizeIntensity | 4.90 | **4.73** | 11.33 |
| BDHWC | (2, 96, 96, 96, 1) | NormalizeIntensity | **5.95** | 8.90 | 17.33 |
| BDHWC | (1, 160, 160, 160, 1) | NormalizeIntensity | **13.00** | 22.02 | 43.51 |
| BDHWC | (1, 256, 256, 256, 1) | NormalizeIntensity | **106.99** | 193.01 | 350.32 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | NormalizeIntensity | 3.81 | **1.45** | 6.58 |
| BHWC | (8, 224, 224, 1) | NormalizeIntensity | 4.14 | **1.96** | 6.87 |
| BHWC | (16, 224, 224, 1) | NormalizeIntensity | 4.60 | **2.96** | 7.78 |
| BHWC | (32, 224, 224, 1) | NormalizeIntensity | 6.17 | **5.28** | 8.98 |
| BHWC | (4, 512, 512, 1) | NormalizeIntensity | 5.00 | **3.47** | 7.98 |
| BHWC | (8, 512, 512, 1) | NormalizeIntensity | 8.21 | **7.42** | 11.43 |
| BHWC | (16, 512, 512, 1) | NormalizeIntensity | **12.72** | 15.43 | 16.93 |
| BHWC | (4, 1280, 1280, 1) | NormalizeIntensity | **19.08** | 24.02 | 23.67 |
| DHWC | (96, 96, 96, 1) | NormalizeIntensity | 4.66 | **3.26** | 7.79 |
| DHWC | (160, 160, 160, 1) | NormalizeIntensity | **12.96** | 14.97 | 16.54 |
| DHWC | (256, 256, 256, 1) | NormalizeIntensity | 109.61 | 165.52 | **83.46** |
| BDHWC | (1, 96, 96, 96, 1) | NormalizeIntensity | 4.66 | **3.04** | 7.89 |
| BDHWC | (2, 96, 96, 96, 1) | NormalizeIntensity | 5.97 | **5.48** | 8.87 |
| BDHWC | (1, 160, 160, 160, 1) | NormalizeIntensity | **12.69** | 14.69 | 16.76 |
| BDHWC | (1, 256, 256, 256, 1) | NormalizeIntensity | 109.41 | 165.35 | **83.06** |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | NormalizeIntensity | 1.41 | **1.02** | 1.26 |
| BHWC | (8, 224, 224, 1) | NormalizeIntensity | 1.93 | **1.62** | 1.82 |
| BHWC | (16, 224, 224, 1) | NormalizeIntensity | 3.03 | 2.86 | **2.70** |
| BHWC | (32, 224, 224, 1) | NormalizeIntensity | 5.51 | 4.67 | **4.61** |
| BHWC | (4, 512, 512, 1) | NormalizeIntensity | 3.55 | **3.21** | 3.53 |
| BHWC | (8, 512, 512, 1) | NormalizeIntensity | 7.46 | **7.10** | 8.40 |
| BHWC | (16, 512, 512, 1) | NormalizeIntensity | 14.37 | **14.16** | 23.69 |
| BHWC | (4, 1280, 1280, 1) | NormalizeIntensity | 22.41 | **20.81** | 35.93 |
| DHWC | (96, 96, 96, 1) | NormalizeIntensity | 3.19 | **3.08** | 3.23 |
| DHWC | (160, 160, 160, 1) | NormalizeIntensity | **13.80** | 13.82 | 16.86 |
| DHWC | (256, 256, 256, 1) | NormalizeIntensity | 163.54 | 164.32 | **107.04** |
| BDHWC | (1, 96, 96, 96, 1) | NormalizeIntensity | 3.10 | **3.04** | 3.07 |
| BDHWC | (2, 96, 96, 96, 1) | NormalizeIntensity | 5.34 | **5.01** | 5.22 |
| BDHWC | (1, 160, 160, 160, 1) | NormalizeIntensity | 13.69 | **13.68** | 16.00 |
| BDHWC | (1, 256, 256, 256, 1) | NormalizeIntensity | 164.72 | 162.04 | **107.46** |
### Orientation

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| DHWC | (96, 96, 96, 1) | Orientation | 26.19 | **7.87** | 43.59 |
| DHWC | (160, 160, 160, 1) | Orientation | 36.81 | **24.71** | 68.23 |
| DHWC | (256, 256, 256, 1) | Orientation | 196.68 | **190.81** | 272.66 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| DHWC | (96, 96, 96, 1) | Orientation | 26.47 | **5.99** | 32.12 |
| DHWC | (160, 160, 160, 1) | Orientation | 37.05 | **16.86** | 43.96 |
| DHWC | (256, 256, 256, 1) | Orientation | 196.84 | 162.45 | **139.79** |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| DHWC | (96, 96, 96, 1) | Orientation | -- | -- | -- |
| DHWC | (160, 160, 160, 1) | Orientation | -- | -- | -- |
| DHWC | (256, 256, 256, 1) | Orientation | -- | -- | -- |
### RandomCutOut

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomCutOut | 50.65 | **5.03** | 151.70 |
| BHWC | (8, 224, 224, 1) | RandomCutOut | 85.48 | **7.51** | 157.99 |
| BHWC | (16, 224, 224, 1) | RandomCutOut | 157.24 | **13.26** | 159.87 |
| BHWC | (32, 224, 224, 1) | RandomCutOut | 302.41 | **25.84** | 154.66 |
| BHWC | (4, 512, 512, 1) | RandomCutOut | 51.76 | **8.35** | 159.45 |
| BHWC | (8, 512, 512, 1) | RandomCutOut | 88.86 | **16.35** | 159.49 |
| BHWC | (16, 512, 512, 1) | RandomCutOut | 164.82 | **30.68** | 163.56 |
| BHWC | (4, 1280, 1280, 1) | RandomCutOut | 65.75 | **36.59** | 175.64 |
| DHWC | (96, 96, 96, 1) | RandomCutOut | 13.99 | **5.74** | 128.33 |
| DHWC | (160, 160, 160, 1) | RandomCutOut | **21.58** | 22.24 | 127.27 |
| DHWC | (256, 256, 256, 1) | RandomCutOut | **119.78** | 190.53 | 281.93 |
| BDHWC | (1, 96, 96, 96, 1) | RandomCutOut | 26.27 | **5.95** | 145.15 |
| BDHWC | (2, 96, 96, 96, 1) | RandomCutOut | 37.99 | **10.39** | 161.76 |
| BDHWC | (1, 160, 160, 160, 1) | RandomCutOut | 34.44 | **22.64** | 146.35 |
| BDHWC | (1, 256, 256, 256, 1) | RandomCutOut | **134.55** | 190.36 | 297.50 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomCutOut | 49.67 | **4.17** | 152.88 |
| BHWC | (8, 224, 224, 1) | RandomCutOut | 85.11 | **6.61** | 158.28 |
| BHWC | (16, 224, 224, 1) | RandomCutOut | 155.00 | **11.77** | 158.92 |
| BHWC | (32, 224, 224, 1) | RandomCutOut | 300.72 | **22.04** | 152.63 |
| BHWC | (4, 512, 512, 1) | RandomCutOut | 51.44 | **6.22** | 157.48 |
| BHWC | (8, 512, 512, 1) | RandomCutOut | 88.59 | **12.55** | 156.65 |
| BHWC | (16, 512, 512, 1) | RandomCutOut | 164.83 | **22.41** | 158.65 |
| BHWC | (4, 1280, 1280, 1) | RandomCutOut | 66.17 | **25.02** | 169.18 |
| DHWC | (96, 96, 96, 1) | RandomCutOut | 13.86 | **4.09** | 127.68 |
| DHWC | (160, 160, 160, 1) | RandomCutOut | 21.85 | **14.99** | 120.60 |
| DHWC | (256, 256, 256, 1) | RandomCutOut | **121.28** | 162.15 | 206.82 |
| BDHWC | (1, 96, 96, 96, 1) | RandomCutOut | 25.75 | **4.10** | 143.49 |
| BDHWC | (2, 96, 96, 96, 1) | RandomCutOut | 37.65 | **7.25** | 159.95 |
| BDHWC | (1, 160, 160, 160, 1) | RandomCutOut | 34.56 | **15.51** | 139.62 |
| BDHWC | (1, 256, 256, 256, 1) | RandomCutOut | **134.87** | 163.27 | 226.54 |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomCutOut | 1.40 | 5.22 | **1.24** |
| BHWC | (8, 224, 224, 1) | RandomCutOut | 2.04 | 8.68 | **1.79** |
| BHWC | (16, 224, 224, 1) | RandomCutOut | 3.07 | 14.32 | **2.97** |
| BHWC | (32, 224, 224, 1) | RandomCutOut | 5.11 | 27.62 | **4.92** |
| BHWC | (4, 512, 512, 1) | RandomCutOut | 3.50 | 7.69 | **3.43** |
| BHWC | (8, 512, 512, 1) | RandomCutOut | **7.82** | 13.77 | 8.50 |
| BHWC | (16, 512, 512, 1) | RandomCutOut | **14.47** | 26.20 | 16.84 |
| BHWC | (4, 1280, 1280, 1) | RandomCutOut | 32.60 | 25.88 | **25.45** |
| DHWC | (96, 96, 96, 1) | RandomCutOut | 3.10 | 4.66 | **2.97** |
| DHWC | (160, 160, 160, 1) | RandomCutOut | **13.87** | 15.40 | 16.06 |
| DHWC | (256, 256, 256, 1) | RandomCutOut | 165.21 | 166.92 | **104.95** |
| BDHWC | (1, 96, 96, 96, 1) | RandomCutOut | 3.24 | 4.90 | **3.00** |
| BDHWC | (2, 96, 96, 96, 1) | RandomCutOut | 5.60 | 7.89 | **5.29** |
| BDHWC | (1, 160, 160, 160, 1) | RandomCutOut | **14.06** | 15.89 | 16.13 |
| BDHWC | (1, 256, 256, 256, 1) | RandomCutOut | 163.09 | 164.16 | **105.97** |
### RandomFlip

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomFlip | 3.87 | **2.07** | 108.22 |
| BHWC | (8, 224, 224, 1) | RandomFlip | 4.24 | **2.82** | 108.81 |
| BHWC | (16, 224, 224, 1) | RandomFlip | 5.11 | **4.67** | 111.70 |
| BHWC | (32, 224, 224, 1) | RandomFlip | **6.92** | 8.48 | 116.98 |
| BHWC | (4, 512, 512, 1) | RandomFlip | **5.67** | 5.99 | 110.17 |
| BHWC | (8, 512, 512, 1) | RandomFlip | **8.64** | 11.08 | 118.35 |
| BHWC | (16, 512, 512, 1) | RandomFlip | **14.86** | 22.01 | 137.09 |
| BHWC | (4, 1280, 1280, 1) | RandomFlip | **21.41** | 33.81 | 164.39 |
| DHWC | (96, 96, 96, 1) | RandomFlip | 5.17 | **5.16** | 116.18 |
| DHWC | (160, 160, 160, 1) | RandomFlip | **14.35** | 21.61 | 138.66 |
| DHWC | (256, 256, 256, 1) | RandomFlip | **131.85** | 189.65 | 367.35 |
| BDHWC | (1, 96, 96, 96, 1) | RandomFlip | 5.17 | **4.95** | 114.29 |
| BDHWC | (2, 96, 96, 96, 1) | RandomFlip | **7.56** | 9.36 | 120.01 |
| BDHWC | (1, 160, 160, 160, 1) | RandomFlip | **14.40** | 21.15 | 139.31 |
| BDHWC | (1, 256, 256, 256, 1) | RandomFlip | **133.62** | 188.54 | 369.68 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomFlip | 3.78 | **1.53** | 113.68 |
| BHWC | (8, 224, 224, 1) | RandomFlip | 4.20 | **2.09** | 115.14 |
| BHWC | (16, 224, 224, 1) | RandomFlip | 5.02 | **3.03** | 115.84 |
| BHWC | (32, 224, 224, 1) | RandomFlip | 7.04 | **5.76** | 116.11 |
| BHWC | (4, 512, 512, 1) | RandomFlip | 5.64 | **3.81** | 111.04 |
| BHWC | (8, 512, 512, 1) | RandomFlip | 8.98 | **7.90** | 113.66 |
| BHWC | (16, 512, 512, 1) | RandomFlip | 15.44 | **14.60** | 126.75 |
| BHWC | (4, 1280, 1280, 1) | RandomFlip | **22.02** | 22.87 | 143.41 |
| DHWC | (96, 96, 96, 1) | RandomFlip | 5.17 | **3.38** | 119.19 |
| DHWC | (160, 160, 160, 1) | RandomFlip | **14.18** | 14.54 | 125.83 |
| DHWC | (256, 256, 256, 1) | RandomFlip | **131.75** | 160.48 | 225.60 |
| BDHWC | (1, 96, 96, 96, 1) | RandomFlip | 5.22 | **3.22** | 117.67 |
| BDHWC | (2, 96, 96, 96, 1) | RandomFlip | 7.43 | **5.98** | 118.52 |
| BDHWC | (1, 160, 160, 160, 1) | RandomFlip | 14.32 | **14.22** | 127.60 |
| BDHWC | (1, 256, 256, 256, 1) | RandomFlip | **133.84** | 160.40 | 224.09 |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomFlip | 1.40 | 1.77 | **1.26** |
| BHWC | (8, 224, 224, 1) | RandomFlip | 1.95 | 2.34 | **1.65** |
| BHWC | (16, 224, 224, 1) | RandomFlip | 3.00 | 3.42 | **2.86** |
| BHWC | (32, 224, 224, 1) | RandomFlip | 5.00 | 5.64 | **4.71** |
| BHWC | (4, 512, 512, 1) | RandomFlip | 3.59 | 4.36 | **3.44** |
| BHWC | (8, 512, 512, 1) | RandomFlip | **7.54** | 7.99 | 8.47 |
| BHWC | (16, 512, 512, 1) | RandomFlip | **14.43** | 15.03 | 16.19 |
| BHWC | (4, 1280, 1280, 1) | RandomFlip | 23.27 | **21.91** | 24.70 |
| DHWC | (96, 96, 96, 1) | RandomFlip | 3.22 | 3.59 | **2.99** |
| DHWC | (160, 160, 160, 1) | RandomFlip | **14.19** | 14.70 | 16.62 |
| DHWC | (256, 256, 256, 1) | RandomFlip | 165.14 | 158.64 | **106.00** |
| BDHWC | (1, 96, 96, 96, 1) | RandomFlip | 3.24 | 3.75 | **3.14** |
| BDHWC | (2, 96, 96, 96, 1) | RandomFlip | **5.31** | 6.61 | 5.39 |
| BDHWC | (1, 160, 160, 160, 1) | RandomFlip | **13.81** | 13.90 | 15.86 |
| BDHWC | (1, 256, 256, 256, 1) | RandomFlip | 166.35 | 159.81 | **106.07** |
### RandomRotate

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomRotate | **11.08** | 13.03 | 47.09 |
| BHWC | (8, 224, 224, 1) | RandomRotate | **11.41** | 22.56 | 48.30 |
| BHWC | (16, 224, 224, 1) | RandomRotate | **12.20** | 42.70 | 56.35 |
| BHWC | (32, 224, 224, 1) | RandomRotate | **15.10** | 81.72 | 54.30 |
| BHWC | (4, 512, 512, 1) | RandomRotate | **12.87** | 16.76 | 53.20 |
| BHWC | (8, 512, 512, 1) | RandomRotate | **17.06** | 31.67 | 61.33 |
| BHWC | (16, 512, 512, 1) | RandomRotate | **44.13** | 58.55 | 97.27 |
| BHWC | (4, 1280, 1280, 1) | RandomRotate | **47.06** | 90.78 | 122.62 |
| DHWC | (96, 96, 96, 1) | RandomRotate | 19.11 | **14.51** | 54.73 |
| DHWC | (160, 160, 160, 1) | RandomRotate | **33.87** | 57.78 | 77.36 |
| DHWC | (256, 256, 256, 1) | RandomRotate | **200.17** | 336.15 | 294.58 |
| BDHWC | (1, 96, 96, 96, 1) | RandomRotate | 17.87 | **14.43** | 53.61 |
| BDHWC | (2, 96, 96, 96, 1) | RandomRotate | **20.41** | 26.00 | 57.01 |
| BDHWC | (1, 160, 160, 160, 1) | RandomRotate | **28.33** | 58.93 | 75.32 |
| BDHWC | (1, 256, 256, 256, 1) | RandomRotate | **181.02** | 333.95 | 294.49 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomRotate | **11.05** | 12.66 | 46.21 |
| BHWC | (8, 224, 224, 1) | RandomRotate | **11.32** | 21.89 | 46.76 |
| BHWC | (16, 224, 224, 1) | RandomRotate | **12.59** | 40.50 | 52.64 |
| BHWC | (32, 224, 224, 1) | RandomRotate | **14.88** | 76.86 | 49.79 |
| BHWC | (4, 512, 512, 1) | RandomRotate | **12.85** | 14.46 | 50.33 |
| BHWC | (8, 512, 512, 1) | RandomRotate | **17.35** | 27.90 | 53.34 |
| BHWC | (16, 512, 512, 1) | RandomRotate | **24.45** | 50.54 | 70.39 |
| BHWC | (4, 1280, 1280, 1) | RandomRotate | **50.60** | 79.55 | 88.13 |
| DHWC | (96, 96, 96, 1) | RandomRotate | 19.66 | **12.94** | 51.99 |
| DHWC | (160, 160, 160, 1) | RandomRotate | **34.03** | 50.77 | 61.57 |
| DHWC | (256, 256, 256, 1) | RandomRotate | 199.05 | 306.35 | **171.77** |
| BDHWC | (1, 96, 96, 96, 1) | RandomRotate | 17.64 | **12.68** | 49.94 |
| BDHWC | (2, 96, 96, 96, 1) | RandomRotate | **20.71** | 22.49 | 51.87 |
| BDHWC | (1, 160, 160, 160, 1) | RandomRotate | **28.79** | 51.00 | 61.13 |
| BDHWC | (1, 256, 256, 256, 1) | RandomRotate | 182.08 | 305.35 | **170.48** |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomRotate | -- | 13.45 | **1.31** |
| BHWC | (8, 224, 224, 1) | RandomRotate | -- | 25.80 | **2.10** |
| BHWC | (16, 224, 224, 1) | RandomRotate | -- | 41.84 | **3.68** |
| BHWC | (32, 224, 224, 1) | RandomRotate | -- | 82.52 | **6.61** |
| BHWC | (4, 512, 512, 1) | RandomRotate | -- | 16.20 | **4.67** |
| BHWC | (8, 512, 512, 1) | RandomRotate | -- | 28.35 | **10.40** |
| BHWC | (16, 512, 512, 1) | RandomRotate | -- | 54.48 | **21.43** |
| BHWC | (4, 1280, 1280, 1) | RandomRotate | -- | 80.97 | **33.28** |
| DHWC | (96, 96, 96, 1) | RandomRotate | -- | 15.11 | **3.95** |
| DHWC | (160, 160, 160, 1) | RandomRotate | -- | 53.16 | **20.00** |
| DHWC | (256, 256, 256, 1) | RandomRotate | -- | 308.46 | **123.45** |
| BDHWC | (1, 96, 96, 96, 1) | RandomRotate | -- | 14.96 | **3.88** |
| BDHWC | (2, 96, 96, 96, 1) | RandomRotate | -- | 25.18 | **6.54** |
| BDHWC | (1, 160, 160, 160, 1) | RandomRotate | -- | 52.03 | **19.42** |
| BDHWC | (1, 256, 256, 256, 1) | RandomRotate | -- | 309.75 | **123.63** |
### RandomRotate90

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomRotate90 | 11.35 | **3.37** | 352.18 |
| BHWC | (8, 224, 224, 1) | RandomRotate90 | 12.20 | **4.31** | 358.17 |
| BHWC | (16, 224, 224, 1) | RandomRotate90 | 14.34 | **6.00** | 358.36 |
| BHWC | (32, 224, 224, 1) | RandomRotate90 | 19.20 | **9.56** | 356.14 |
| BHWC | (4, 512, 512, 1) | RandomRotate90 | 15.60 | **7.00** | 348.53 |
| BHWC | (8, 512, 512, 1) | RandomRotate90 | 21.90 | **12.73** | 351.51 |
| BHWC | (16, 512, 512, 1) | RandomRotate90 | 37.33 | **23.40** | 360.90 |
| BHWC | (4, 1280, 1280, 1) | RandomRotate90 | 59.62 | **35.96** | 389.64 |
| DHWC | (96, 96, 96, 1) | RandomRotate90 | 14.77 | **6.64** | 364.81 |
| DHWC | (160, 160, 160, 1) | RandomRotate90 | 33.62 | **22.52** | 375.42 |
| DHWC | (256, 256, 256, 1) | RandomRotate90 | 273.98 | **192.16** | 609.85 |
| BDHWC | (1, 96, 96, 96, 1) | RandomRotate90 | 15.69 | **6.11** | 361.83 |
| BDHWC | (2, 96, 96, 96, 1) | RandomRotate90 | 20.65 | **10.15** | 352.50 |
| BDHWC | (1, 160, 160, 160, 1) | RandomRotate90 | 32.89 | **22.41** | 373.69 |
| BDHWC | (1, 256, 256, 256, 1) | RandomRotate90 | 275.08 | **191.23** | 612.95 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomRotate90 | 11.39 | **3.02** | 357.86 |
| BHWC | (8, 224, 224, 1) | RandomRotate90 | 12.33 | **3.53** | 370.10 |
| BHWC | (16, 224, 224, 1) | RandomRotate90 | 14.08 | **4.28** | 370.39 |
| BHWC | (32, 224, 224, 1) | RandomRotate90 | 20.12 | **6.28** | 353.66 |
| BHWC | (4, 512, 512, 1) | RandomRotate90 | 16.07 | **4.99** | 355.73 |
| BHWC | (8, 512, 512, 1) | RandomRotate90 | 22.08 | **8.84** | 351.01 |
| BHWC | (16, 512, 512, 1) | RandomRotate90 | 37.23 | **15.72** | 358.68 |
| BHWC | (4, 1280, 1280, 1) | RandomRotate90 | 57.02 | **23.84** | 378.00 |
| DHWC | (96, 96, 96, 1) | RandomRotate90 | 14.69 | **4.75** | 372.19 |
| DHWC | (160, 160, 160, 1) | RandomRotate90 | 33.27 | **14.86** | 361.35 |
| DHWC | (256, 256, 256, 1) | RandomRotate90 | 272.82 | **162.62** | 476.01 |
| BDHWC | (1, 96, 96, 96, 1) | RandomRotate90 | 14.57 | **4.46** | 370.12 |
| BDHWC | (2, 96, 96, 96, 1) | RandomRotate90 | 20.05 | **6.56** | 351.62 |
| BDHWC | (1, 160, 160, 160, 1) | RandomRotate90 | 33.25 | **15.43** | 363.23 |
| BDHWC | (1, 256, 256, 256, 1) | RandomRotate90 | 275.13 | **161.74** | 478.97 |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomRotate90 | 1.50 | 3.61 | **1.43** |
| BHWC | (8, 224, 224, 1) | RandomRotate90 | 2.13 | 4.04 | **1.79** |
| BHWC | (16, 224, 224, 1) | RandomRotate90 | 3.02 | 5.06 | **3.01** |
| BHWC | (32, 224, 224, 1) | RandomRotate90 | 5.53 | 7.55 | **4.71** |
| BHWC | (4, 512, 512, 1) | RandomRotate90 | 3.84 | 5.62 | **3.70** |
| BHWC | (8, 512, 512, 1) | RandomRotate90 | **7.54** | 9.34 | 8.75 |
| BHWC | (16, 512, 512, 1) | RandomRotate90 | **14.69** | 16.53 | 16.87 |
| BHWC | (4, 1280, 1280, 1) | RandomRotate90 | **23.68** | 23.85 | 35.38 |
| DHWC | (96, 96, 96, 1) | RandomRotate90 | **3.26** | 5.27 | **3.26** |
| DHWC | (160, 160, 160, 1) | RandomRotate90 | **14.47** | 16.22 | 16.89 |
| DHWC | (256, 256, 256, 1) | RandomRotate90 | 164.94 | 165.14 | **107.33** |
| BDHWC | (1, 96, 96, 96, 1) | RandomRotate90 | 3.42 | 5.27 | **3.04** |
| BDHWC | (2, 96, 96, 96, 1) | RandomRotate90 | 6.01 | 7.96 | **5.13** |
| BDHWC | (1, 160, 160, 160, 1) | RandomRotate90 | **14.11** | 15.58 | 16.27 |
| BDHWC | (1, 256, 256, 256, 1) | RandomRotate90 | 166.75 | 165.25 | **107.13** |
### RandomShiftIntensity

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomShiftIntensity | 5.11 | **2.23** | 62.32 |
| BHWC | (8, 224, 224, 1) | RandomShiftIntensity | 5.48 | **3.04** | 67.23 |
| BHWC | (16, 224, 224, 1) | RandomShiftIntensity | 6.03 | **4.83** | 68.94 |
| BHWC | (32, 224, 224, 1) | RandomShiftIntensity | **7.47** | 8.74 | 65.83 |
| BHWC | (4, 512, 512, 1) | RandomShiftIntensity | 6.50 | **5.87** | 65.80 |
| BHWC | (8, 512, 512, 1) | RandomShiftIntensity | **9.78** | 11.45 | 67.91 |
| BHWC | (16, 512, 512, 1) | RandomShiftIntensity | **14.67** | 21.67 | 74.88 |
| BHWC | (4, 1280, 1280, 1) | RandomShiftIntensity | **20.43** | 35.09 | 86.02 |
| DHWC | (96, 96, 96, 1) | RandomShiftIntensity | 6.29 | **4.94** | 70.71 |
| DHWC | (160, 160, 160, 1) | RandomShiftIntensity | **14.75** | 21.07 | 76.60 |
| DHWC | (256, 256, 256, 1) | RandomShiftIntensity | **108.58** | 188.40 | 220.47 |
| BDHWC | (1, 96, 96, 96, 1) | RandomShiftIntensity | 6.10 | **5.12** | 69.40 |
| BDHWC | (2, 96, 96, 96, 1) | RandomShiftIntensity | **8.04** | 9.36 | 65.76 |
| BDHWC | (1, 160, 160, 160, 1) | RandomShiftIntensity | **14.55** | 21.65 | 77.69 |
| BDHWC | (1, 256, 256, 256, 1) | RandomShiftIntensity | **110.19** | 187.30 | 221.83 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomShiftIntensity | 5.06 | **1.70** | 63.52 |
| BHWC | (8, 224, 224, 1) | RandomShiftIntensity | 5.46 | **2.15** | 68.05 |
| BHWC | (16, 224, 224, 1) | RandomShiftIntensity | 6.50 | **3.06** | 68.84 |
| BHWC | (32, 224, 224, 1) | RandomShiftIntensity | 7.88 | **5.51** | 64.44 |
| BHWC | (4, 512, 512, 1) | RandomShiftIntensity | 6.54 | **3.78** | 65.28 |
| BHWC | (8, 512, 512, 1) | RandomShiftIntensity | 9.59 | **8.07** | 65.13 |
| BHWC | (16, 512, 512, 1) | RandomShiftIntensity | 14.43 | **13.90** | 68.39 |
| BHWC | (4, 1280, 1280, 1) | RandomShiftIntensity | **20.68** | 22.73 | 75.49 |
| DHWC | (96, 96, 96, 1) | RandomShiftIntensity | 6.28 | **3.48** | 70.22 |
| DHWC | (160, 160, 160, 1) | RandomShiftIntensity | 14.68 | **13.79** | 69.68 |
| DHWC | (256, 256, 256, 1) | RandomShiftIntensity | **108.86** | 159.03 | 149.11 |
| BDHWC | (1, 96, 96, 96, 1) | RandomShiftIntensity | 6.15 | **3.32** | 69.38 |
| BDHWC | (2, 96, 96, 96, 1) | RandomShiftIntensity | 8.15 | **5.98** | 64.22 |
| BDHWC | (1, 160, 160, 160, 1) | RandomShiftIntensity | 14.44 | **14.35** | 70.98 |
| BDHWC | (1, 256, 256, 256, 1) | RandomShiftIntensity | **110.64** | 160.32 | 147.73 |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomShiftIntensity | 1.46 | 1.77 | **1.26** |
| BHWC | (8, 224, 224, 1) | RandomShiftIntensity | 1.99 | 2.36 | **1.69** |
| BHWC | (16, 224, 224, 1) | RandomShiftIntensity | 3.03 | 3.39 | **2.94** |
| BHWC | (32, 224, 224, 1) | RandomShiftIntensity | 5.36 | 5.21 | **4.92** |
| BHWC | (4, 512, 512, 1) | RandomShiftIntensity | 3.75 | 4.21 | **3.51** |
| BHWC | (8, 512, 512, 1) | RandomShiftIntensity | **7.64** | 8.08 | 8.34 |
| BHWC | (16, 512, 512, 1) | RandomShiftIntensity | **14.04** | 14.67 | 16.26 |
| BHWC | (4, 1280, 1280, 1) | RandomShiftIntensity | **22.74** | 23.45 | 25.27 |
| DHWC | (96, 96, 96, 1) | RandomShiftIntensity | 3.24 | 3.49 | **3.02** |
| DHWC | (160, 160, 160, 1) | RandomShiftIntensity | **14.04** | 14.25 | 16.06 |
| DHWC | (256, 256, 256, 1) | RandomShiftIntensity | 164.67 | 161.93 | **106.18** |
| BDHWC | (1, 96, 96, 96, 1) | RandomShiftIntensity | 3.25 | 3.50 | **3.19** |
| BDHWC | (2, 96, 96, 96, 1) | RandomShiftIntensity | 5.77 | 6.09 | **5.31** |
| BDHWC | (1, 160, 160, 160, 1) | RandomShiftIntensity | 13.63 | **13.42** | 15.85 |
| BDHWC | (1, 256, 256, 256, 1) | RandomShiftIntensity | 165.16 | 160.70 | **105.78** |
### RandomSpatialCrop

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomSpatialCrop | 12.76 | **4.47** | 19.29 |
| BHWC | (8, 224, 224, 1) | RandomSpatialCrop | 13.05 | **5.23** | 20.79 |
| BHWC | (16, 224, 224, 1) | RandomSpatialCrop | 13.74 | **6.87** | 22.37 |
| BHWC | (32, 224, 224, 1) | RandomSpatialCrop | 15.80 | **10.15** | 25.95 |
| BHWC | (4, 512, 512, 1) | RandomSpatialCrop | 14.09 | **7.93** | 22.71 |
| BHWC | (8, 512, 512, 1) | RandomSpatialCrop | 15.96 | **12.62** | 30.84 |
| BHWC | (16, 512, 512, 1) | RandomSpatialCrop | 21.30 | **21.16** | 44.15 |
| BHWC | (4, 1280, 1280, 1) | RandomSpatialCrop | **26.48** | 31.69 | 65.83 |
| DHWC | (96, 96, 96, 1) | RandomSpatialCrop | 13.38 | **6.78** | 21.80 |
| DHWC | (160, 160, 160, 1) | RandomSpatialCrop | 19.97 | **19.66** | 40.90 |
| DHWC | (256, 256, 256, 1) | RandomSpatialCrop | **101.33** | 138.92 | 226.75 |
| BDHWC | (1, 96, 96, 96, 1) | RandomSpatialCrop | 13.59 | **6.89** | 24.05 |
| BDHWC | (2, 96, 96, 96, 1) | RandomSpatialCrop | 15.22 | **10.56** | 27.32 |
| BDHWC | (1, 160, 160, 160, 1) | RandomSpatialCrop | **20.14** | 20.33 | 43.98 |
| BDHWC | (1, 256, 256, 256, 1) | RandomSpatialCrop | **103.41** | 138.56 | 230.92 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomSpatialCrop | 12.78 | **4.02** | 17.47 |
| BHWC | (8, 224, 224, 1) | RandomSpatialCrop | 13.03 | **4.29** | 17.58 |
| BHWC | (16, 224, 224, 1) | RandomSpatialCrop | 13.82 | **5.19** | 18.97 |
| BHWC | (32, 224, 224, 1) | RandomSpatialCrop | 15.87 | **7.70** | 20.31 |
| BHWC | (4, 512, 512, 1) | RandomSpatialCrop | 14.18 | **5.71** | 19.35 |
| BHWC | (8, 512, 512, 1) | RandomSpatialCrop | 15.74 | **8.65** | 21.85 |
| BHWC | (16, 512, 512, 1) | RandomSpatialCrop | 22.74 | **13.76** | 32.79 |
| BHWC | (4, 1280, 1280, 1) | RandomSpatialCrop | 26.86 | **20.58** | 36.29 |
| DHWC | (96, 96, 96, 1) | RandomSpatialCrop | 13.56 | **5.23** | 18.25 |
| DHWC | (160, 160, 160, 1) | RandomSpatialCrop | 19.64 | **12.07** | 26.69 |
| DHWC | (256, 256, 256, 1) | RandomSpatialCrop | 101.34 | 108.99 | **91.85** |
| BDHWC | (1, 96, 96, 96, 1) | RandomSpatialCrop | 13.43 | **5.34** | 20.06 |
| BDHWC | (2, 96, 96, 96, 1) | RandomSpatialCrop | 15.38 | **6.99** | 21.64 |
| BDHWC | (1, 160, 160, 160, 1) | RandomSpatialCrop | 20.04 | **13.21** | 29.20 |
| BDHWC | (1, 256, 256, 256, 1) | RandomSpatialCrop | 103.17 | 110.28 | **93.15** |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | RandomSpatialCrop | 92.12 | 6.62 | **1.10** |
| BHWC | (8, 224, 224, 1) | RandomSpatialCrop | 91.79 | 7.23 | **1.51** |
| BHWC | (16, 224, 224, 1) | RandomSpatialCrop | 91.73 | 7.85 | **2.33** |
| BHWC | (32, 224, 224, 1) | RandomSpatialCrop | 90.08 | 9.44 | **3.85** |
| BHWC | (4, 512, 512, 1) | RandomSpatialCrop | 90.02 | 8.51 | **2.92** |
| BHWC | (8, 512, 512, 1) | RandomSpatialCrop | 92.86 | 10.55 | **4.79** |
| BHWC | (16, 512, 512, 1) | RandomSpatialCrop | 95.62 | 17.19 | **12.35** |
| BHWC | (4, 1280, 1280, 1) | RandomSpatialCrop | 103.55 | 22.82 | **19.63** |
| DHWC | (96, 96, 96, 1) | RandomSpatialCrop | 91.02 | 7.94 | **2.30** |
| DHWC | (160, 160, 160, 1) | RandomSpatialCrop | 98.74 | 15.96 | **10.90** |
| DHWC | (256, 256, 256, 1) | RandomSpatialCrop | 200.69 | 114.42 | **74.84** |
| BDHWC | (1, 96, 96, 96, 1) | RandomSpatialCrop | 90.83 | 8.39 | **2.34** |
| BDHWC | (2, 96, 96, 96, 1) | RandomSpatialCrop | 94.92 | 9.77 | **3.77** |
| BDHWC | (1, 160, 160, 160, 1) | RandomSpatialCrop | 98.71 | 15.10 | **10.76** |
| BDHWC | (1, 256, 256, 256, 1) | RandomSpatialCrop | 201.56 | 115.73 | **74.25** |
### Resize

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | Resize | 6.34 | **2.24** | 3.92 |
| BHWC | (8, 224, 224, 1) | Resize | 7.49 | **3.04** | 4.57 |
| BHWC | (16, 224, 224, 1) | Resize | 10.18 | **4.44** | 6.09 |
| BHWC | (32, 224, 224, 1) | Resize | 15.58 | **7.85** | 9.92 |
| BHWC | (4, 512, 512, 1) | Resize | 11.80 | **5.51** | 9.35 |
| BHWC | (8, 512, 512, 1) | Resize | 19.00 | **10.01** | 14.74 |
| BHWC | (16, 512, 512, 1) | Resize | 34.97 | **19.70** | 43.35 |
| BHWC | (4, 1280, 1280, 1) | Resize | 51.87 | **30.20** | 67.63 |
| DHWC | (96, 96, 96, 1) | Resize | 67.38 | **8.91** | 33.17 |
| DHWC | (160, 160, 160, 1) | Resize | 115.11 | **24.35** | 51.53 |
| DHWC | (256, 256, 256, 1) | Resize | 628.46 | **159.35** | 237.54 |
| BDHWC | (1, 96, 96, 96, 1) | Resize | 68.14 | **8.79** | 31.12 |
| BDHWC | (2, 96, 96, 96, 1) | Resize | 79.36 | **12.92** | 37.43 |
| BDHWC | (1, 160, 160, 160, 1) | Resize | 111.43 | **24.46** | 73.81 |
| BDHWC | (1, 256, 256, 256, 1) | Resize | 635.82 | **158.81** | 303.55 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | Resize | 6.11 | **1.69** | 3.23 |
| BHWC | (8, 224, 224, 1) | Resize | 7.54 | **2.12** | 3.56 |
| BHWC | (16, 224, 224, 1) | Resize | 10.01 | **2.75** | 4.32 |
| BHWC | (32, 224, 224, 1) | Resize | 15.48 | **4.61** | 6.01 |
| BHWC | (4, 512, 512, 1) | Resize | 11.95 | **3.30** | 4.82 |
| BHWC | (8, 512, 512, 1) | Resize | 19.27 | **5.74** | 7.31 |
| BHWC | (16, 512, 512, 1) | Resize | 35.36 | **12.42** | 16.22 |
| BHWC | (4, 1280, 1280, 1) | Resize | 51.97 | **18.40** | 29.68 |
| DHWC | (96, 96, 96, 1) | Resize | 66.88 | **7.23** | 30.03 |
| DHWC | (160, 160, 160, 1) | Resize | 113.68 | **17.39** | 37.29 |
| DHWC | (256, 256, 256, 1) | Resize | 628.25 | 129.82 | **101.99** |
| BDHWC | (1, 96, 96, 96, 1) | Resize | 69.48 | **7.36** | 28.10 |
| BDHWC | (2, 96, 96, 96, 1) | Resize | 78.94 | **9.26** | 29.90 |
| BDHWC | (1, 160, 160, 160, 1) | Resize | 110.66 | **17.33** | 36.64 |
| BDHWC | (1, 256, 256, 256, 1) | Resize | 637.24 | 128.67 | **99.51** |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | Resize | 1.14 | 3.07 | **1.06** |
| BHWC | (8, 224, 224, 1) | Resize | 1.86 | 3.79 | **1.47** |
| BHWC | (16, 224, 224, 1) | Resize | 2.84 | 4.37 | **2.50** |
| BHWC | (32, 224, 224, 1) | Resize | 4.76 | 5.70 | **4.44** |
| BHWC | (4, 512, 512, 1) | Resize | **3.00** | 4.46 | 3.32 |
| BHWC | (8, 512, 512, 1) | Resize | 6.28 | 6.66 | **6.17** |
| BHWC | (16, 512, 512, 1) | Resize | 13.53 | **12.82** | 24.45 |
| BHWC | (4, 1280, 1280, 1) | Resize | **18.12** | 18.39 | 27.20 |
| DHWC | (96, 96, 96, 1) | Resize | 2.45 | 8.89 | **2.05** |
| DHWC | (160, 160, 160, 1) | Resize | **9.34** | 16.35 | 11.40 |
| DHWC | (256, 256, 256, 1) | Resize | 82.87 | 122.56 | **74.63** |
| BDHWC | (1, 96, 96, 96, 1) | Resize | 2.44 | 8.96 | **2.34** |
| BDHWC | (2, 96, 96, 96, 1) | Resize | **3.68** | 10.16 | 3.79 |
| BDHWC | (1, 160, 160, 160, 1) | Resize | **9.58** | 16.41 | 10.85 |
| BDHWC | (1, 256, 256, 256, 1) | Resize | 83.45 | 121.92 | **73.94** |
### Rotate90

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | Rotate90 | 3.74 | **1.42** | 3.40 |
| BHWC | (8, 224, 224, 1) | Rotate90 | 4.46 | **2.34** | 4.08 |
| BHWC | (16, 224, 224, 1) | Rotate90 | 6.08 | **3.98** | 6.02 |
| BHWC | (32, 224, 224, 1) | Rotate90 | 10.43 | **8.56** | 10.09 |
| BHWC | (4, 512, 512, 1) | Rotate90 | 7.65 | **5.51** | 6.78 |
| BHWC | (8, 512, 512, 1) | Rotate90 | 14.02 | **10.98** | 16.57 |
| BHWC | (16, 512, 512, 1) | Rotate90 | 27.23 | **21.34** | 41.06 |
| BHWC | (4, 1280, 1280, 1) | Rotate90 | 39.10 | **33.17** | 63.20 |
| DHWC | (96, 96, 96, 1) | Rotate90 | 6.47 | **4.73** | 6.88 |
| DHWC | (160, 160, 160, 1) | Rotate90 | 24.79 | **21.02** | 30.88 |
| DHWC | (256, 256, 256, 1) | Rotate90 | 264.41 | **191.10** | 240.72 |
| BDHWC | (1, 96, 96, 96, 1) | Rotate90 | 6.78 | **4.67** | 6.16 |
| BDHWC | (2, 96, 96, 96, 1) | Rotate90 | 11.98 | **9.00** | 11.11 |
| BDHWC | (1, 160, 160, 160, 1) | Rotate90 | 25.30 | **20.79** | 30.51 |
| BDHWC | (1, 256, 256, 256, 1) | Rotate90 | 266.87 | **188.52** | 242.40 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | Rotate90 | 3.56 | **0.90** | 2.82 |
| BHWC | (8, 224, 224, 1) | Rotate90 | 4.49 | **1.43** | 3.20 |
| BHWC | (16, 224, 224, 1) | Rotate90 | 6.09 | **2.41** | 4.18 |
| BHWC | (32, 224, 224, 1) | Rotate90 | 10.72 | **4.58** | 6.28 |
| BHWC | (4, 512, 512, 1) | Rotate90 | 7.66 | **3.12** | 4.85 |
| BHWC | (8, 512, 512, 1) | Rotate90 | 13.64 | **7.14** | 9.83 |
| BHWC | (16, 512, 512, 1) | Rotate90 | 28.03 | **13.88** | 25.41 |
| BHWC | (4, 1280, 1280, 1) | Rotate90 | 38.91 | **21.83** | 41.34 |
| DHWC | (96, 96, 96, 1) | Rotate90 | 6.80 | **2.92** | 4.79 |
| DHWC | (160, 160, 160, 1) | Rotate90 | 25.45 | **12.87** | 17.70 |
| DHWC | (256, 256, 256, 1) | Rotate90 | 263.50 | 160.07 | **108.44** |
| BDHWC | (1, 96, 96, 96, 1) | Rotate90 | 6.71 | **2.95** | 4.59 |
| BDHWC | (2, 96, 96, 96, 1) | Rotate90 | 11.78 | **5.77** | 6.57 |
| BDHWC | (1, 160, 160, 160, 1) | Rotate90 | 25.32 | **13.64** | 17.10 |
| BDHWC | (1, 256, 256, 256, 1) | Rotate90 | 266.93 | 158.05 | **107.67** |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | Rotate90 | 1.34 | **0.93** | 1.30 |
| BHWC | (8, 224, 224, 1) | Rotate90 | 1.86 | **1.54** | 1.77 |
| BHWC | (16, 224, 224, 1) | Rotate90 | 2.83 | **2.51** | 2.91 |
| BHWC | (32, 224, 224, 1) | Rotate90 | 5.35 | 5.10 | **4.44** |
| BHWC | (4, 512, 512, 1) | Rotate90 | 3.62 | **2.92** | 3.45 |
| BHWC | (8, 512, 512, 1) | Rotate90 | 7.63 | **7.22** | 8.48 |
| BHWC | (16, 512, 512, 1) | Rotate90 | 14.89 | **13.84** | 23.25 |
| BHWC | (4, 1280, 1280, 1) | Rotate90 | 22.82 | **21.74** | 36.02 |
| DHWC | (96, 96, 96, 1) | Rotate90 | 3.03 | **2.82** | 3.11 |
| DHWC | (160, 160, 160, 1) | Rotate90 | 14.18 | **13.82** | 16.89 |
| DHWC | (256, 256, 256, 1) | Rotate90 | 164.41 | 160.42 | **106.70** |
| BDHWC | (1, 96, 96, 96, 1) | Rotate90 | 2.99 | 2.99 | **2.80** |
| BDHWC | (2, 96, 96, 96, 1) | Rotate90 | 5.98 | 5.23 | **4.75** |
| BDHWC | (1, 160, 160, 160, 1) | Rotate90 | 13.99 | **13.90** | 16.19 |
| BDHWC | (1, 256, 256, 256, 1) | Rotate90 | 163.19 | 161.36 | **108.09** |
### ScaleIntensityRange

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | ScaleIntensityRange | **1.19** | 1.48 | 2.09 |
| BHWC | (8, 224, 224, 1) | ScaleIntensityRange | **1.46** | 2.36 | 2.50 |
| BHWC | (16, 224, 224, 1) | ScaleIntensityRange | **1.97** | 3.89 | 3.59 |
| BHWC | (32, 224, 224, 1) | ScaleIntensityRange | **2.97** | 7.39 | 5.68 |
| BHWC | (4, 512, 512, 1) | ScaleIntensityRange | **2.32** | 4.96 | 4.13 |
| BHWC | (8, 512, 512, 1) | ScaleIntensityRange | **5.19** | 10.98 | 9.77 |
| BHWC | (16, 512, 512, 1) | ScaleIntensityRange | **10.02** | 22.04 | 19.51 |
| BHWC | (4, 1280, 1280, 1) | ScaleIntensityRange | **16.48** | 34.17 | 31.09 |
| DHWC | (96, 96, 96, 1) | ScaleIntensityRange | **2.15** | 4.49 | 3.75 |
| DHWC | (160, 160, 160, 1) | ScaleIntensityRange | **9.68** | 21.07 | 18.99 |
| DHWC | (256, 256, 256, 1) | ScaleIntensityRange | **106.92** | 188.83 | 149.93 |
| BDHWC | (1, 96, 96, 96, 1) | ScaleIntensityRange | **2.05** | 4.15 | 3.74 |
| BDHWC | (2, 96, 96, 96, 1) | ScaleIntensityRange | **3.70** | 8.49 | 6.25 |
| BDHWC | (1, 160, 160, 160, 1) | ScaleIntensityRange | **9.55** | 20.97 | 18.91 |
| BDHWC | (1, 256, 256, 256, 1) | ScaleIntensityRange | **106.17** | 189.45 | 151.27 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | ScaleIntensityRange | 1.32 | **0.95** | 1.80 |
| BHWC | (8, 224, 224, 1) | ScaleIntensityRange | 1.56 | **1.48** | 2.16 |
| BHWC | (16, 224, 224, 1) | ScaleIntensityRange | **2.00** | 2.35 | 2.86 |
| BHWC | (32, 224, 224, 1) | ScaleIntensityRange | **3.16** | 4.68 | 3.97 |
| BHWC | (4, 512, 512, 1) | ScaleIntensityRange | **2.45** | 3.19 | 3.33 |
| BHWC | (8, 512, 512, 1) | ScaleIntensityRange | **5.17** | 6.90 | 6.49 |
| BHWC | (16, 512, 512, 1) | ScaleIntensityRange | **10.97** | 14.46 | 11.51 |
| BHWC | (4, 1280, 1280, 1) | ScaleIntensityRange | **17.11** | 22.49 | 18.58 |
| DHWC | (96, 96, 96, 1) | ScaleIntensityRange | **2.12** | 2.74 | 2.84 |
| DHWC | (160, 160, 160, 1) | ScaleIntensityRange | **9.73** | 13.94 | 11.45 |
| DHWC | (256, 256, 256, 1) | ScaleIntensityRange | 106.31 | 160.50 | **79.26** |
| BDHWC | (1, 96, 96, 96, 1) | ScaleIntensityRange | **2.10** | 2.59 | 2.84 |
| BDHWC | (2, 96, 96, 96, 1) | ScaleIntensityRange | **3.42** | 4.81 | 4.45 |
| BDHWC | (1, 160, 160, 160, 1) | ScaleIntensityRange | **9.86** | 13.83 | 11.21 |
| BDHWC | (1, 256, 256, 256, 1) | ScaleIntensityRange | 107.23 | 159.67 | **78.02** |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | ScaleIntensityRange | 1.44 | **0.88** | 1.23 |
| BHWC | (8, 224, 224, 1) | ScaleIntensityRange | 1.90 | **1.47** | 1.91 |
| BHWC | (16, 224, 224, 1) | ScaleIntensityRange | 3.01 | **2.61** | 2.94 |
| BHWC | (32, 224, 224, 1) | ScaleIntensityRange | 5.26 | **4.25** | 4.61 |
| BHWC | (4, 512, 512, 1) | ScaleIntensityRange | 3.75 | **3.30** | 3.45 |
| BHWC | (8, 512, 512, 1) | ScaleIntensityRange | 7.36 | **7.06** | 8.45 |
| BHWC | (16, 512, 512, 1) | ScaleIntensityRange | 14.24 | **13.74** | 22.92 |
| BHWC | (4, 1280, 1280, 1) | ScaleIntensityRange | 22.25 | **21.50** | 36.18 |
| DHWC | (96, 96, 96, 1) | ScaleIntensityRange | 3.12 | **2.70** | 2.99 |
| DHWC | (160, 160, 160, 1) | ScaleIntensityRange | 13.96 | **13.66** | 16.71 |
| DHWC | (256, 256, 256, 1) | ScaleIntensityRange | 165.41 | 158.41 | **106.26** |
| BDHWC | (1, 96, 96, 96, 1) | ScaleIntensityRange | 3.03 | **2.90** | 3.01 |
| BDHWC | (2, 96, 96, 96, 1) | ScaleIntensityRange | 5.60 | 5.73 | **4.70** |
| BDHWC | (1, 160, 160, 160, 1) | ScaleIntensityRange | 13.65 | **12.92** | 15.95 |
| BDHWC | (1, 256, 256, 256, 1) | ScaleIntensityRange | 165.18 | 157.80 | **105.48** |
### ShiftIntensity

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | ShiftIntensity | **0.73** | 1.29 | 1.32 |
| BHWC | (8, 224, 224, 1) | ShiftIntensity | **0.99** | 2.17 | 1.73 |
| BHWC | (16, 224, 224, 1) | ShiftIntensity | **1.65** | 4.08 | 2.68 |
| BHWC | (32, 224, 224, 1) | ShiftIntensity | **2.80** | 8.07 | 5.07 |
| BHWC | (4, 512, 512, 1) | ShiftIntensity | **2.05** | 4.83 | 3.26 |
| BHWC | (8, 512, 512, 1) | ShiftIntensity | **5.24** | 10.57 | 9.75 |
| BHWC | (16, 512, 512, 1) | ShiftIntensity | **10.27** | 21.33 | 19.45 |
| BHWC | (4, 1280, 1280, 1) | ShiftIntensity | **15.76** | 33.27 | 30.40 |
| DHWC | (96, 96, 96, 1) | ShiftIntensity | **1.83** | 4.25 | 2.88 |
| DHWC | (160, 160, 160, 1) | ShiftIntensity | **9.83** | 20.66 | 18.96 |
| DHWC | (256, 256, 256, 1) | ShiftIntensity | **105.07** | 189.10 | 149.06 |
| BDHWC | (1, 96, 96, 96, 1) | ShiftIntensity | **1.75** | 4.65 | 2.89 |
| BDHWC | (2, 96, 96, 96, 1) | ShiftIntensity | **3.08** | 8.25 | 5.51 |
| BDHWC | (1, 160, 160, 160, 1) | ShiftIntensity | **9.91** | 20.70 | 18.71 |
| BDHWC | (1, 256, 256, 256, 1) | ShiftIntensity | **105.75** | 186.46 | 149.50 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | ShiftIntensity | **0.66** | 0.80 | 0.97 |
| BHWC | (8, 224, 224, 1) | ShiftIntensity | **1.02** | 1.32 | 1.27 |
| BHWC | (16, 224, 224, 1) | ShiftIntensity | **1.56** | 2.52 | 2.03 |
| BHWC | (32, 224, 224, 1) | ShiftIntensity | **2.84** | 5.22 | 3.09 |
| BHWC | (4, 512, 512, 1) | ShiftIntensity | **2.04** | 2.93 | 2.44 |
| BHWC | (8, 512, 512, 1) | ShiftIntensity | **5.06** | 6.80 | 5.81 |
| BHWC | (16, 512, 512, 1) | ShiftIntensity | **10.02** | 13.62 | 11.54 |
| BHWC | (4, 1280, 1280, 1) | ShiftIntensity | **15.84** | 21.55 | 17.84 |
| DHWC | (96, 96, 96, 1) | ShiftIntensity | **1.86** | 2.50 | 2.07 |
| DHWC | (160, 160, 160, 1) | ShiftIntensity | **9.67** | 13.49 | 11.30 |
| DHWC | (256, 256, 256, 1) | ShiftIntensity | 105.08 | 158.86 | **77.35** |
| BDHWC | (1, 96, 96, 96, 1) | ShiftIntensity | **1.77** | 2.75 | 2.03 |
| BDHWC | (2, 96, 96, 96, 1) | ShiftIntensity | **3.18** | 5.38 | 3.41 |
| BDHWC | (1, 160, 160, 160, 1) | ShiftIntensity | **9.75** | 13.20 | 11.12 |
| BDHWC | (1, 256, 256, 256, 1) | ShiftIntensity | 104.17 | 157.50 | **76.23** |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | ShiftIntensity | 1.33 | **0.93** | 1.22 |
| BHWC | (8, 224, 224, 1) | ShiftIntensity | 1.81 | **1.47** | 1.75 |
| BHWC | (16, 224, 224, 1) | ShiftIntensity | 2.93 | **2.64** | 2.85 |
| BHWC | (32, 224, 224, 1) | ShiftIntensity | 4.95 | 4.82 | **4.75** |
| BHWC | (4, 512, 512, 1) | ShiftIntensity | 3.52 | **3.35** | 3.51 |
| BHWC | (8, 512, 512, 1) | ShiftIntensity | 7.29 | **7.25** | 8.30 |
| BHWC | (16, 512, 512, 1) | ShiftIntensity | 13.73 | **13.67** | 16.47 |
| BHWC | (4, 1280, 1280, 1) | ShiftIntensity | 22.15 | **21.75** | 35.78 |
| DHWC | (96, 96, 96, 1) | ShiftIntensity | 3.22 | **2.60** | 3.11 |
| DHWC | (160, 160, 160, 1) | ShiftIntensity | 13.98 | **13.57** | 16.86 |
| DHWC | (256, 256, 256, 1) | ShiftIntensity | 164.59 | 155.81 | **107.85** |
| BDHWC | (1, 96, 96, 96, 1) | ShiftIntensity | 3.03 | **2.87** | 2.93 |
| BDHWC | (2, 96, 96, 96, 1) | ShiftIntensity | 5.72 | 5.67 | **5.07** |
| BDHWC | (1, 160, 160, 160, 1) | ShiftIntensity | 13.60 | **13.16** | 16.04 |
| BDHWC | (1, 256, 256, 256, 1) | ShiftIntensity | 164.68 | 158.56 | **106.02** |
### SignalFillEmpty

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | SignalFillEmpty | **1.57** | 1.75 | 2.95 |
| BHWC | (8, 224, 224, 1) | SignalFillEmpty | **2.01** | 2.60 | 3.54 |
| BHWC | (16, 224, 224, 1) | SignalFillEmpty | **2.55** | 4.22 | 5.23 |
| BHWC | (32, 224, 224, 1) | SignalFillEmpty | **4.16** | 7.87 | 8.04 |
| BHWC | (4, 512, 512, 1) | SignalFillEmpty | **2.99** | 5.35 | 5.67 |
| BHWC | (8, 512, 512, 1) | SignalFillEmpty | **5.97** | 11.43 | 14.08 |
| BHWC | (16, 512, 512, 1) | SignalFillEmpty | **11.54** | 22.54 | 27.14 |
| BHWC | (4, 1280, 1280, 1) | SignalFillEmpty | **17.85** | 34.88 | 41.91 |
| DHWC | (96, 96, 96, 1) | SignalFillEmpty | **2.69** | 4.81 | 5.23 |
| DHWC | (160, 160, 160, 1) | SignalFillEmpty | **11.32** | 21.92 | 26.43 |
| DHWC | (256, 256, 256, 1) | SignalFillEmpty | **105.28** | 194.40 | 217.25 |
| BDHWC | (1, 96, 96, 96, 1) | SignalFillEmpty | **2.92** | 4.94 | 5.27 |
| BDHWC | (2, 96, 96, 96, 1) | SignalFillEmpty | **4.23** | 8.68 | 8.74 |
| BDHWC | (1, 160, 160, 160, 1) | SignalFillEmpty | **11.16** | 21.81 | 25.96 |
| BDHWC | (1, 256, 256, 256, 1) | SignalFillEmpty | **105.60** | 191.60 | 216.81 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | SignalFillEmpty | 1.68 | **1.15** | 2.38 |
| BHWC | (8, 224, 224, 1) | SignalFillEmpty | 1.99 | **1.69** | 2.55 |
| BHWC | (16, 224, 224, 1) | SignalFillEmpty | **2.56** | 2.67 | 3.47 |
| BHWC | (32, 224, 224, 1) | SignalFillEmpty | **4.20** | 4.98 | 4.62 |
| BHWC | (4, 512, 512, 1) | SignalFillEmpty | **2.85** | 3.12 | 3.77 |
| BHWC | (8, 512, 512, 1) | SignalFillEmpty | **6.05** | 7.61 | 7.23 |
| BHWC | (16, 512, 512, 1) | SignalFillEmpty | **11.54** | 14.62 | 13.08 |
| BHWC | (4, 1280, 1280, 1) | SignalFillEmpty | **17.89** | 23.22 | 19.32 |
| DHWC | (96, 96, 96, 1) | SignalFillEmpty | **2.73** | 3.03 | 3.66 |
| DHWC | (160, 160, 160, 1) | SignalFillEmpty | **10.89** | 14.57 | 12.76 |
| DHWC | (256, 256, 256, 1) | SignalFillEmpty | 107.15 | 163.67 | **78.32** |
| BDHWC | (1, 96, 96, 96, 1) | SignalFillEmpty | **2.81** | 3.19 | 3.74 |
| BDHWC | (2, 96, 96, 96, 1) | SignalFillEmpty | **4.41** | 5.43 | 4.87 |
| BDHWC | (1, 160, 160, 160, 1) | SignalFillEmpty | **11.34** | 14.43 | 12.45 |
| BDHWC | (1, 256, 256, 256, 1) | SignalFillEmpty | 108.10 | 161.55 | **78.67** |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | SignalFillEmpty | 1.33 | **0.95** | 1.26 |
| BHWC | (8, 224, 224, 1) | SignalFillEmpty | 1.78 | **1.50** | 1.88 |
| BHWC | (16, 224, 224, 1) | SignalFillEmpty | 3.30 | **2.60** | 2.65 |
| BHWC | (32, 224, 224, 1) | SignalFillEmpty | 5.23 | **4.59** | 4.61 |
| BHWC | (4, 512, 512, 1) | SignalFillEmpty | 3.58 | **3.05** | 3.44 |
| BHWC | (8, 512, 512, 1) | SignalFillEmpty | 7.50 | **7.24** | 8.34 |
| BHWC | (16, 512, 512, 1) | SignalFillEmpty | **13.93** | 13.98 | 23.14 |
| BHWC | (4, 1280, 1280, 1) | SignalFillEmpty | 21.79 | **20.25** | 35.79 |
| DHWC | (96, 96, 96, 1) | SignalFillEmpty | 3.06 | **2.83** | 3.00 |
| DHWC | (160, 160, 160, 1) | SignalFillEmpty | **13.54** | 13.70 | 16.76 |
| DHWC | (256, 256, 256, 1) | SignalFillEmpty | 164.28 | 155.59 | **106.14** |
| BDHWC | (1, 96, 96, 96, 1) | SignalFillEmpty | 3.18 | **2.80** | 3.08 |
| BDHWC | (2, 96, 96, 96, 1) | SignalFillEmpty | 5.76 | **4.83** | 4.96 |
| BDHWC | (1, 160, 160, 160, 1) | SignalFillEmpty | 13.60 | **12.60** | 15.93 |
| BDHWC | (1, 256, 256, 256, 1) | SignalFillEmpty | 164.02 | 162.03 | **105.83** |
### Spacing

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| DHWC | (96, 96, 96, 1) | Spacing | 73.37 | **8.72** | 47.63 |
| DHWC | (160, 160, 160, 1) | Spacing | 85.48 | **15.63** | 59.19 |
| DHWC | (256, 256, 256, 1) | Spacing | 159.59 | **47.58** | 191.49 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| DHWC | (96, 96, 96, 1) | Spacing | 73.01 | **6.89** | 44.81 |
| DHWC | (160, 160, 160, 1) | Spacing | 88.66 | **8.32** | 47.23 |
| DHWC | (256, 256, 256, 1) | Spacing | 160.28 | **17.88** | 54.16 |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| DHWC | (96, 96, 96, 1) | Spacing | -- | -- | -- |
| DHWC | (160, 160, 160, 1) | Spacing | -- | -- | -- |
| DHWC | (256, 256, 256, 1) | Spacing | -- | -- | -- |
### SpatialCrop

#### CPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | SpatialCrop | 13.18 | **4.49** | 18.01 |
| BHWC | (8, 224, 224, 1) | SpatialCrop | 13.21 | **5.20** | 18.05 |
| BHWC | (16, 224, 224, 1) | SpatialCrop | 13.74 | **6.99** | 20.00 |
| BHWC | (32, 224, 224, 1) | SpatialCrop | 15.30 | **10.05** | 25.71 |
| BHWC | (4, 512, 512, 1) | SpatialCrop | 14.34 | **7.83** | 21.96 |
| BHWC | (8, 512, 512, 1) | SpatialCrop | 16.20 | **11.76** | 28.45 |
| BHWC | (16, 512, 512, 1) | SpatialCrop | **21.92** | 22.24 | 42.76 |
| BHWC | (4, 1280, 1280, 1) | SpatialCrop | **27.17** | 31.50 | 56.74 |
| DHWC | (96, 96, 96, 1) | SpatialCrop | 13.31 | **7.26** | 20.13 |
| DHWC | (160, 160, 160, 1) | SpatialCrop | 19.71 | **19.62** | 39.60 |
| DHWC | (256, 256, 256, 1) | SpatialCrop | **100.74** | 139.22 | 229.86 |
| BDHWC | (1, 96, 96, 96, 1) | SpatialCrop | 14.10 | **6.88** | 21.87 |
| BDHWC | (2, 96, 96, 96, 1) | SpatialCrop | 15.22 | **10.30** | 26.63 |
| BDHWC | (1, 160, 160, 160, 1) | SpatialCrop | 19.95 | **19.91** | 41.88 |
| BDHWC | (1, 256, 256, 256, 1) | SpatialCrop | **103.08** | 138.86 | 233.33 |

#### GPU

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | SpatialCrop | 13.12 | **4.05** | 17.44 |
| BHWC | (8, 224, 224, 1) | SpatialCrop | 13.38 | **4.31** | 16.95 |
| BHWC | (16, 224, 224, 1) | SpatialCrop | 13.68 | **5.33** | 18.56 |
| BHWC | (32, 224, 224, 1) | SpatialCrop | 15.36 | **6.91** | 19.99 |
| BHWC | (4, 512, 512, 1) | SpatialCrop | 14.51 | **5.92** | 19.10 |
| BHWC | (8, 512, 512, 1) | SpatialCrop | 16.19 | **7.95** | 21.49 |
| BHWC | (16, 512, 512, 1) | SpatialCrop | 21.70 | **14.56** | 32.69 |
| BHWC | (4, 1280, 1280, 1) | SpatialCrop | 26.93 | **20.50** | 41.60 |
| DHWC | (96, 96, 96, 1) | SpatialCrop | 13.67 | **5.62** | 17.93 |
| DHWC | (160, 160, 160, 1) | SpatialCrop | 19.89 | **12.72** | 26.17 |
| DHWC | (256, 256, 256, 1) | SpatialCrop | 100.51 | 109.61 | **92.33** |
| BDHWC | (1, 96, 96, 96, 1) | SpatialCrop | 14.00 | **5.42** | 19.56 |
| BDHWC | (2, 96, 96, 96, 1) | SpatialCrop | 15.30 | **7.09** | 21.59 |
| BDHWC | (1, 160, 160, 160, 1) | SpatialCrop | 20.10 | **12.81** | 28.46 |
| BDHWC | (1, 256, 256, 256, 1) | SpatialCrop | 102.26 | 109.00 | **93.46** |

#### GPU (compiled)

| Layout | Shape | Transform | TensorFlow (ms) | Torch (ms) | JAX (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| BHWC | (4, 224, 224, 1) | SpatialCrop | **1.17** | 7.46 | 1.18 |
| BHWC | (8, 224, 224, 1) | SpatialCrop | **1.63** | 7.85 | 1.67 |
| BHWC | (16, 224, 224, 1) | SpatialCrop | **2.31** | 8.60 | 2.36 |
| BHWC | (32, 224, 224, 1) | SpatialCrop | 3.87 | 10.16 | **3.74** |
| BHWC | (4, 512, 512, 1) | SpatialCrop | 2.78 | 8.80 | **2.68** |
| BHWC | (8, 512, 512, 1) | SpatialCrop | 4.83 | 11.01 | **4.39** |
| BHWC | (16, 512, 512, 1) | SpatialCrop | **10.71** | 17.32 | 12.74 |
| BHWC | (4, 1280, 1280, 1) | SpatialCrop | **17.05** | 23.02 | 19.30 |
| DHWC | (96, 96, 96, 1) | SpatialCrop | 2.54 | 8.93 | **2.13** |
| DHWC | (160, 160, 160, 1) | SpatialCrop | **9.39** | 16.46 | 11.31 |
| DHWC | (256, 256, 256, 1) | SpatialCrop | 81.83 | 114.97 | **74.80** |
| BDHWC | (1, 96, 96, 96, 1) | SpatialCrop | 2.44 | 9.00 | **2.37** |
| BDHWC | (2, 96, 96, 96, 1) | SpatialCrop | **3.76** | 10.98 | 3.79 |
| BDHWC | (1, 160, 160, 160, 1) | SpatialCrop | **9.27** | 16.27 | 10.86 |
| BDHWC | (1, 256, 256, 256, 1) | SpatialCrop | 82.99 | 116.76 | **74.06** |

TensorFlow `RandomRotate` is not XLA-compatible because its affine image kernel is unsupported by the TensorFlow XLA GPU compiler. Metadata-aware sample-level transforms `CropForeground`, `Orientation`, and `Spacing` have no compiled records.

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

## Reports

- [Transforms](REPORT.md)

## How to run

**Option 1**: `cli`

The main `cli` arguments are:

- `--device {cpu,gpu}` selects the isolated execution device.
- `--target-transforms NAME [NAME ...]` selects transforms; the default is
  `all`.
- `--layout {HWC,DHWC,BHWC,BDHWC}` selects sample or batch channel-last input.
- `--sizes SIZE [SIZE ...]` selects square 2D or cubic 3D spatial sizes.
- `--batch-size N` selects the batch size; use `1` for `HWC` and `DHWC`.
- `--channels N` selects the channel count.
- `--iterations N` and `--warmup N` control timing iterations.
- `--compile` enables TensorFlow XLA, JAX JIT, or Torch Inductor.
- `--seed N` controls transform randomness, and `--json PATH` saves results.

Examples covering the backend, device, and compilation combinations:

```bash
# Replace the backend with tensorflow, torch, or jax.
KERAS_BACKEND=tensorflow python benchmarks/transforms.py \
  --device cpu --layout BDHWC --sizes 96 --batch-size 1
KERAS_BACKEND=tensorflow python benchmarks/transforms.py \
  --device gpu --layout BDHWC --sizes 96 --batch-size 1
KERAS_BACKEND=tensorflow python benchmarks/transforms.py \
  --device cpu --layout BDHWC --sizes 96 --batch-size 1 --compile
KERAS_BACKEND=tensorflow python benchmarks/transforms.py \
  --device gpu --layout BDHWC --sizes 96 --batch-size 1 --compile

# 2D batch input.
KERAS_BACKEND=torch python benchmarks/transforms.py \
  --device gpu --layout BHWC --sizes 224 --batch-size 8
KERAS_BACKEND=torch python benchmarks/transforms.py \
  --device gpu --layout BHWC --sizes 224 --batch-size 8 --compile

# 3D sample input. Batch size is explicitly one.
KERAS_BACKEND=jax python benchmarks/transforms.py \
  --device cpu --layout DHWC --sizes 96 --batch-size 1
KERAS_BACKEND=jax python benchmarks/transforms.py \
  --device gpu --layout DHWC --sizes 96 --batch-size 1 --compile

# Target one or more transforms and optionally save JSON output.
KERAS_BACKEND=tensorflow python benchmarks/transforms.py \
  --device gpu --layout BDHWC --sizes 96 --batch-size 1 \
  --target-transforms RandomElasticTransform RandomRotate \
  --iterations 50 --warmup 10 --json /tmp/transform_results.json
```

**Option 2**: `Python`

For a `Python` matrix launcher, pass the selected device profile through each
subprocess rather than changing `os.environ` after importing Keras. The
benchmark applies the CPU/GPU visibility rule before importing Keras:

```python
import os
import subprocess

BENCHMARK = "benchmarks/transforms.py"

def run(
    backend,
    layout,
    size,
    batch,
    compile_enabled=False,
    device="cpu",
    transforms=("all",),
):
    transform_label = "-".join(transforms)
    json_path = (
        f"/tmp/{backend}_{layout}_SIZE{size}_BATCH{batch}_"
        f"TRANSFORM_{transform_label}_on_{device.upper()}_COMPILE_{compile_enabled}.json"
    )
    command = [
        "python",
        "-u",
        BENCHMARK,
        "--device",
        device,
        "--iterations",
        "50",
        "--warmup",
        "10",
        "--layout",
        layout,
        "--sizes",
        str(size),
        "--batch-size",
        str(batch),
        "--target-transforms",
        *transforms,
        "--json",
        json_path,
    ]
    if compile_enabled:
        command.append("--compile")

    environment = {**os.environ, "KERAS_BACKEND": backend}
    print(
        f"\n=== {backend} {layout} size={size} batch={batch} "
        f"compile={compile_enabled} ===",
        flush=True,
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    if process.returncode != 0:
        print(
            f"!! FAILED: {backend} {layout} size={size} batch={batch} "
            f"compile={compile_enabled} (rc={process.returncode})",
            flush=True,
        )

def run_transform_matrix(
    backend,
    compile_enabled=False,
    layout="BDHWC",
    transforms=("all",),
    device="cpu",
):
    if layout == "BHWC":
        profiles = [(224, [4, 8, 16, 32]), (512, [4, 8, 16]), (1024, [4, 8])]
    elif layout == "DHWC":
        profiles = [(96, [1]), (160, [1]), (256, [1])]
    elif layout == "BDHWC":
        profiles = [(96, [1, 2]), (160, [1]), (256, [1])]
    else:
        raise ValueError(f"Unsupported benchmark layout: {layout}")

    for size, batches in profiles:
        for batch in batches:
            run(
                backend,
                layout,
                size,
                batch,
                compile_enabled,
                device=device,
                transforms=transforms,
            )
```

The Python matrix launcher can cover all backend, device, compilation, and
layout combinations. `BDHWC` is the default 3D batch layout; `BHWC` covers 2D
batch inputs; and `DHWC` covers 3D sample inputs with batch size one.

```python
backends = ["tensorflow", "torch", "jax"]

for backend in backends:
    for device in ["cpu", "gpu"]:
        for compile_enabled in [False, True]:
            # Default 3D batch layout.
            run_transform_matrix(
                backend,
                compile_enabled=compile_enabled,
                layout="BDHWC",
                device=device,
            )

            # 2D batch layout.
            run_transform_matrix(
                backend,
                compile_enabled=compile_enabled,
                layout="BHWC",
                device=device,
            )

            # 3D sample layout; the launcher uses batch size one.
            run_transform_matrix(
                backend,
                compile_enabled=compile_enabled,
                layout="DHWC",
                device=device,
            )
```

To benchmark only selected transforms, pass a tuple of names. Names are
case-insensitive. TensorFlow uses `tf.function(jit_compile=True)`, JAX uses
`jax.jit`, and Torch uses `torch.compile` with the `inductor`
backend when `compile_enabled=True`.

```python
selected_transforms = ("RandomElasticTransform", "RandomRotate")
for backend in backends:
    for device in ["cpu", "gpu"]:
        for compile_enabled in [False, True]:
            run_transform_matrix(
                backend,
                compile_enabled=compile_enabled,
                layout="BDHWC",
                device=device,
                transforms=selected_transforms,
            )
```

Compilation time is reported separately as `compile_time_ms`. Metadata-aware
transforms are included in the CPU/GPU benchmark profiles, but their
Python-side metadata or dynamic geometry may not be supported by a backend
compiler. If compilation or the first compiled call is unsupported, it is
recorded with `compile_status=not-compile-compatible`, and the remaining
benchmark continues.

The runner separates warm-up from measured iterations, reuses one prebuilt
tensor case while creating a fresh bundle for every call, synchronizes backend
work before stopping the timer, and reports forward timings. Input-case setup
is reported separately as `case_setup_ms`; it is not included in transform
timings. The benchmark is a timing tool, not a correctness replacement for
`test/transforms/`.

# Transform Benchmarks

These scripts measure MedicAI transforms outside the test suite. They compare
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

Shared benchmark mechanics belong in `common/`; transform-specific cases belong
in `transform_benchmark/`. If model or training benchmarks are added later,
they can reuse device, timing, compilation, and reporting conventions without
expanding this transform CLI module.

Set the Keras backend before starting Python:

```bash
KERAS_BACKEND=tensorflow python benchmarks/transforms.py --device cpu
KERAS_BACKEND=tensorflow python benchmarks/transforms.py --device both
KERAS_BACKEND=torch python benchmarks/transforms.py --device gpu

# Compare eager execution with the active backend's XLA path.
KERAS_BACKEND=tensorflow python benchmarks/transforms.py --device gpu --compile xla
```

The registry uses two execution groups:

- `cpu`: transforms that depend on medical metadata or are normally applied
  before batching, such as `CropForeground`, `Orientation`, and `Spacing`.
- `cpu+gpu`: tensor-only transforms such as intensity, flip, resize, crop, and
  random augmentation transforms.

The runner separates warm-up from measured iterations, reuses one prebuilt
tensor case while creating a fresh bundle for every call, synchronizes backend
work before stopping the timer, and reports forward and optional inverse
timings. Input-case setup is reported separately as `case_setup_ms`; it is not
included in transform timings. Inverse calls have their own warm-up phase. The
benchmark is a timing tool, not a correctness replacement for
`test/transforms/`.

`--compile none` is the default and measures eager transform calls. With
`--compile xla`, TensorFlow uses `tf.function(jit_compile=True)`, JAX uses
`jax.jit`, and Torch uses the optional `torch_xla` OpenXLA backend. Compilation
time is reported separately as `compile_time_ms`. Metadata-dependent transforms
are skipped because their Python-side metadata and dynamic geometry are not
part of this compiled tensor-only benchmark. Invertible transforms report
`inverse_status=not-compiled` in this mode; their trace-based inverse remains
an eager operation. If a transform uses an operation unsupported by the active
backend's XLA compiler, it is recorded with `compile_status=not-xla-compatible`
and the remaining benchmark continues. If the optional Torch XLA runtime is
not installed, the result is reported as `compile_status=xla-unavailable`.

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

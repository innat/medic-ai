# Transform Benchmarks

These scripts measure MedicAI transforms outside the test suite. They compare
dataloader-style CPU execution with tensor-only transforms that can also run
inside a model or GPU training step.

Set the Keras backend before starting Python:

```bash
KERAS_BACKEND=tensorflow python benchmarks/transforms.py --device cpu
KERAS_BACKEND=tensorflow python benchmarks/transforms.py --device both
KERAS_BACKEND=torch python benchmarks/transforms.py --device gpu
```

The registry uses two execution groups:

- `cpu`: transforms that depend on medical metadata or are normally applied
  before batching, such as `CropForeground`, `Orientation`, and `Spacing`.
- `cpu+gpu`: tensor-only transforms such as intensity, flip, resize, crop, and
  random augmentation transforms.

The runner separates warm-up from measured iterations, creates fresh bundles
for every call, synchronizes backend work before stopping the timer, and
reports forward and optional inverse timings. It is a timing tool, not a
correctness replacement for `test/transforms/`.

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

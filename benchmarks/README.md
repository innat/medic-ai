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
  --layout BDHWC --iterations 50 --warmup 10 --json /tmp/medicai.json
```

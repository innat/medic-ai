# Custom Transformation

This example shows how to build your own transform with
``medicai.transforms`` and, more importantly, when to choose each approach.

In practice, most custom transforms fall into one of these buckets:

- a small tensor-only operation applied to one or more keys
- a random augmentation that should sometimes be skipped
- an invertible transform that must restore predictions back to original space
- a sample-level transform that needs full control over both tensors and metadata

Medic-AI supports all of those, but the recommended entry point depends on the
use case.

Note:
`medicai.transforms` lets you call transforms with plain Python mappings such
as `{"image": image, "label": label}`. Internally, Medic-AI wraps those
inputs into a `TensorBundle`, so examples based on `Transform`,
`KeyedTransform`, `RandomTransform`, and `InvertibleTransform` work on
`bundle.data` and `bundle.meta` even though user-facing code can start from a
regular dictionary.

## Quick Decision Guide

| Option | Use when | Switch when |
| --- | --- | --- | --- |
| ``LambdaTransform`` | One callable per key is enough. | You need cross-key logic, new keys, or metadata updates. |
| ``KeyedTransform`` | You want a reusable deterministic keyed class. | You need randomness, inversion, or cross-key logic. |
| ``RandomTransform`` | You want a reusable random augmentation. | You also need reliable inverse behavior. |
| ``InvertibleTransform`` | The transform must be undone later. | The transform is lossy or not practically reversible. |
| ``Transform`` | The transform is sample-level. | The logic is really just per-key tensor work. |

## `LambdaTransform`

``LambdaTransform`` is the easiest option for most users. It is a good fit for
small per-key transforms that do not need a full class definition.

### Use case: simple deterministic preprocessing

Suppose you want to clamp image values into a safe range before feeding them to
later transforms.

```python
import tensorflow as tf
from medicai.transforms import LambdaTransform

clip_image = LambdaTransform(
    keys=["image"],
    fn=lambda x: tf.clip_by_value(x, 0.0, 1.0),
    name="clip_image",
)

sample = {
    "image": tf.random.normal((64, 64, 1)),
    "label": tf.random.uniform((64, 64, 1), maxval=2, dtype=tf.int32),
}

result = clip_image(sample)
print(result["image"].shape)
```

### Use case: same operation on multiple keys

This is useful when image and label must stay aligned.

```python
import tensorflow as tf
from medicai.transforms import LambdaTransform

flip_lr = LambdaTransform(
    keys=["image", "label"],
    fn=lambda x: tf.reverse(x, axis=[1]),
    name="flip_lr",
)

sample = {
    "image": tf.random.normal((64, 64, 1)),
    "label": tf.random.uniform((64, 64, 1), maxval=2, dtype=tf.int32),
}

result = flip_lr(sample)
print(result["image"].shape, result["label"].shape)
```

### Use case: lightweight random augmentation

If you need random transformation with probability, ``LambdaTransform`` is often enough.

```python
import tensorflow as tf
from medicai.transforms import LambdaTransform

random_shift = LambdaTransform(
    keys=["image"],
    fn=lambda x: x + tf.cast(0.1, x.dtype),
    prob=0.5,
    name="random_shift",
)

sample = {"image": tf.random.normal((32, 32, 1))}
result = random_shift(sample)
```

### Use case: simple invertible transform

This is a good pattern when your transform is mathematically reversible.

```python
import tensorflow as tf
from medicai.transforms import LambdaTransform

add_bias = LambdaTransform(
    keys=["image"],
    fn=lambda x: x + tf.cast(2.0, x.dtype),
    inverse_fn=lambda x: x - tf.cast(2.0, x.dtype),
    name="add_bias",
)

sample = {"image": tf.random.normal((32, 32, 1))}
forward = add_bias(sample)
restored = add_bias.inverse(forward)
```

### When `LambdaTransform` becomes awkward

``LambdaTransform`` is intentionally simple. It starts to feel stretched when
one transform must:

- inspect multiple keys together
- create new output keys
- update metadata alongside tensors

That is the point where a full ``Transform`` subclass is usually the better
fit.

## `Transform`

Subclass ``Transform`` when the logic is really **sample-level** rather than
**per-key**.

### Use case: transform depends on multiple tensors together

This is a more realistic ``Transform`` scenario: combine two labels to define
one foreground region, then use that shared region to update multiple image
keys. That kind of **cross-key** dependency is exactly why ``Transform``
exists.

```python
import tensorflow as tf
from medicai.transforms import Transform

class MaskImagesByCombinedLabels(Transform):
    def _build_foreground_mask(self, label_1, label_2):
        return tf.logical_or(label_1 > 0, label_2 > 0)

    def _apply_mask(self, image, mask):
        return tf.where(mask, image, tf.zeros_like(image))

    def apply(self, bundle):
        label_1 = bundle.data["label_1"]
        label_2 = bundle.data["label_2"]

        foreground = self._build_foreground_mask(label_1, label_2)
        bundle.data["image_1"] = self._apply_mask(bundle.data["image_1"], foreground)
        bundle.data["image_2"] = self._apply_mask(bundle.data["image_2"], foreground)
        return bundle


transform = MaskImagesByCombinedLabels()
sample = {
    "image_1": tf.random.normal((64, 64, 1)),
    "image_2": tf.random.normal((64, 64, 1)),
    "label_1": tf.random.uniform((64, 64, 1), maxval=2, dtype=tf.int32),
    "label_2": tf.random.uniform((64, 64, 1), maxval=2, dtype=tf.int32),
}

result = transform(sample)
```

### Use case: update tensor data and metadata together

This is one of the clearest reasons to use ``Transform`` directly.

```python
import tensorflow as tf
from medicai.transforms import Transform

class MarkPreprocessed(Transform):
    def _clip_image(self, image):
        return tf.clip_by_value(image, -1.0, 1.0)

    def apply(self, bundle):
        bundle.data["image"] = self._clip_image(bundle.data["image"])
        bundle.meta["preprocessed"] = True
        return bundle


transform = MarkPreprocessed()
sample = {"image": tf.random.normal((64, 64, 1))}
meta = {"affine": tf.eye(4)}

result = transform(sample, meta)
print(result.meta["preprocessed"])
```

### Use case: update spatial metadata such as `affine`

Transforms that change geometry often need to update spatial metadata together
with tensor data. This is another strong reason to use `Transform`.

```python
import tensorflow as tf
from medicai.transforms import Transform

class ShiftOriginX(Transform):
    def __init__(self, delta):
        self.delta = delta

    def _shift_affine(self, affine):
        translation = tf.convert_to_tensor(
            [
                [1.0, 0.0, 0.0, self.delta],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=affine.dtype,
        )
        return tf.linalg.matmul(affine, translation)

    def apply(self, bundle):
        bundle.meta["affine"] = self._shift_affine(bundle.meta["affine"])
        return bundle
```

Choose this path when the transform is fundamentally about the whole sample,
not just one tensor at a time.

### Use case: create a new output key

``Transform`` is also the cleanest base when a transform produces additional
outputs rather than only replacing existing ones.

```python
import tensorflow as tf
from medicai.transforms import Transform

class AddForegroundMask(Transform):
    def _foreground_mask(self, image):
        return tf.cast(image > 0.0, tf.float32)

    def apply(self, bundle):
        bundle.data["foreground_mask"] = self._foreground_mask(bundle.data["image"])
        return bundle
```

### Use case: custom trace entry with `build_trace_entry`

``build_trace_entry(...)`` is the helper that creates a standardized transform
record inside ``bundle.meta["applied_transforms"]``. It is useful when a custom
``Transform`` wants trace history even though it is not subclassing
``InvertibleTransform``.

```python
import tensorflow as tf
from medicai.transforms import Transform

class WindowAndTrace(Transform):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def _window(self, image):
        return tf.clip_by_value(image, self.lower, self.upper)

    def apply(self, bundle):
        bundle.data["image"] = self._window(bundle.data["image"])
        bundle.push_transform(
            self.build_trace_entry(
                params={"keys": ["image"], "lower": self.lower, "upper": self.upper},
                applied=True,
                random=False,
                invertible=False,
                kernel="window_clip",
            )
        )
        return bundle
```

### Use case: full custom inverse on `Transform`

You can also implement inverse behavior directly on ``Transform`` when the
logic is sample-level.

```python
import tensorflow as tf
from medicai.transforms import Transform

class AddPredictionKey(Transform):
    def apply(self, bundle):
        bundle.data["prediction"] = tf.zeros_like(bundle.data["image"])
        bundle.push_transform(
            self.build_trace_entry(
                params={"created_key": "prediction"},
                applied=True,
                random=False,
                invertible=True,
            )
        )
        return bundle

    @property
    def invertible(self):
        return True

    def inverse(self, bundle):
        applied = bundle.get_applied_transforms()
        if not applied:
            return bundle
        last = applied.pop()
        created_key = last["params"]["created_key"]
        bundle.data.pop(created_key, None)
        return bundle
```

### When `Transform` becomes awkward

If the real logic is applying some tensor operation to multiple keys at a time, then ``Transform`` is often too heavy. For example:

```python
class ClipImages(Transform):
    def apply(self, bundle):
        for key in ["image_1", "image_2"]:
            bundle.data[key] = tf.clip_by_value(bundle.data[key], 0.0, 1.0)
        return bundle
```

That works, but it is boilerplate-heavy compared with ``LambdaTransform`` or
``KeyedTransform``.

## `KeyedTransform`

Subclass ``KeyedTransform`` when the transform is deterministic and applies
independently to selected keys.

### Use case: reusable per-key deterministic transform

```python
import tensorflow as tf
from medicai.transforms import KeyedTransform

class AddConstant(KeyedTransform):
    def __init__(self, keys, value, allow_missing_keys=False):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.value = value

    def apply(self, bundle):
        self.apply_to_present_keys(bundle, lambda tensor, _: tensor + self.value)
        return bundle


transform = AddConstant(keys=["image"], value=2.0)
sample = {"image": tf.random.normal((64, 64, 1))}
result = transform(sample)
```

### Use case: inspect which requested keys are present with `iter_present_keys`

``iter_present_keys(...)`` is useful when a transform should branch based on
which keys actually exist in the current sample.

```python
import tensorflow as tf
from medicai.transforms import KeyedTransform

class ReportPresentImageKeys(KeyedTransform):
    def apply(self, bundle):
        present_keys = self.iter_present_keys(bundle)
        bundle.meta["present_image_keys"] = list(present_keys)
        return bundle


transform = ReportPresentImageKeys(
    keys=["image_1", "image_2", "image_3"],
    allow_missing_keys=True,
)
```

Use it when:

- some keys may be optional
- you need to know which keys are available before applying logic
- you want metadata or control flow based on available keys

### Use case: update present keys in one place with `apply_to_present_keys`

``apply_to_present_keys(...)`` is the main convenience method of
``KeyedTransform``. It applies one tensor function to whichever requested keys
are present and returns the list of updated keys.

```python
import tensorflow as tf
from medicai.transforms import KeyedTransform

class CastImagesToFloat32(KeyedTransform):
    def apply(self, bundle):
        updated_keys = self.apply_to_present_keys(
            bundle,
            lambda tensor, _: tf.cast(tensor, tf.float32),
        )
        bundle.meta["cast_keys"] = list(updated_keys)
        return bundle
```

Use it when:

- each selected key should receive the same tensor-level operation
- you want missing-key policy handled consistently
- you want to record exactly which keys were updated

### When `KeyedTransform` becomes awkward

``KeyedTransform`` starts to become awkward when the transform is no longer
truly **per-key**.

```python
class UseImage1ToRewriteImage2(KeyedTransform):
    def apply(self, bundle):
        # Awkward: now the transform is really cross-key, not key-wise.
        reference = bundle.data["image_1"]
        self.apply_to_present_keys(
            bundle,
            lambda tensor, key: tf.where(reference > 0.0, tensor, tf.zeros_like(tensor)),
            keys=["image_2"],
        )
        return bundle
```

At that point, plain ``Transform`` is usually the clearer choice.

## `RandomTransform`

Subclass ``RandomTransform`` when you want to build a real reusable random
augmentation class instead of just a one-line random callable.

### Use case: custom random noise

```python
import tensorflow as tf
from medicai.transforms import RandomTransform

class RandomGaussianOffset(RandomTransform):
    def __init__(self, keys, stddev=0.1, prob=0.5):
        super().__init__(prob=prob)
        self.keys = list(keys)
        self.stddev = stddev

    def apply(self, bundle):
        should_apply = self.sample_should_apply()

        for key in self.keys:
            tensor = bundle.data[key]
            bundle.data[key] = tf.cond(
                should_apply,
                lambda t=tensor: t
                + tf.random.normal(tf.shape(t), stddev=self.stddev, dtype=t.dtype),
                lambda t=tensor: t,
            )

        self.record_random_transform(
            bundle,
            {
                "keys": self.keys,
                "stddev": self.stddev,
            },
            applied=should_apply,
        )
        return bundle
```

``record_random_transform(...)``: is the helper that appends a standard random
trace entry into the bundle metadata. In practice, it records things like:

- which keys were targeted
- the random transform parameters you want to keep
- whether the transform was actually applied for this sample
- that the trace came from a random transform rather than a deterministic one

This matters because a random transform can be skipped. Without
``record_random_transform(...)``, downstream debugging or inverse-related logic
would have no reliable record of whether the transform ran.

If the transform also samples extra parameters such as a random offset,
sigma, or kernel size, include those in the recorded `params` so they remain
available for debugging or inverse-aware logic.

Use this path when you are building a proper augmentation primitive that should
look and behave like the rest of ``medicai``'s random transform API.

### When `RandomTransform` becomes awkward

If the transform must later be undone, ``RandomTransform`` alone is not enough.

```python
class RandomShiftButNeedInverse(RandomTransform):
    ...
    # Problem: we also need to know the exact sampled offset later
    # to restore predictions to the original space.
```

That is the point where the transform should also carry invertible behavior.

## `InvertibleTransform`

Subclass ``InvertibleTransform`` when the transform must be undone later.

### Use case: restore model outputs back to original scale

```python
import tensorflow as tf
from medicai.transforms import InvertibleTransform, KeyedTransform

class AddValue(KeyedTransform, InvertibleTransform):
    def __init__(self, keys, value):
        KeyedTransform.__init__(self, keys=keys)
        InvertibleTransform.__init__(self)
        self.value = value

    def apply(self, bundle):
        present_keys = self.apply_to_present_keys(
            bundle,
            lambda tensor, _: tensor + tf.cast(self.value, tensor.dtype),
        )
        self.record_transform(
            bundle,
            {
                "keys": list(present_keys),
                "value": self.value,
            },
        )
        return bundle

    def inverse(self, bundle):
        trace = self._get_last_trace(bundle)
        if trace is None:
            return bundle
        self.apply_to_present_keys(
            bundle,
            lambda tensor, _: tensor - tf.cast(trace["params"]["value"], tensor.dtype),
            keys=trace["params"]["keys"],
        )
        return bundle
```

### Use case: random and invertible transform together

Sometimes a transform is both stochastic and reversible. In that case, the
class usually needs ``RandomTransform`` for probability handling and
``InvertibleTransform`` for trace-backed restoration.

```python
import tensorflow as tf
from medicai.transforms import InvertibleTransform, KeyedTransform, RandomTransform

class RandomAddOffset(KeyedTransform, RandomTransform, InvertibleTransform):
    def __init__(self, keys, offset_range=(-0.2, 0.2), prob=0.5):
        KeyedTransform.__init__(self, keys=keys)
        RandomTransform.__init__(self, prob=prob)
        InvertibleTransform.__init__(self)
        self.offset_range = offset_range

    def _sample_offset(self, dtype):
        return tf.random.uniform(
            shape=(),
            minval=self.offset_range[0],
            maxval=self.offset_range[1],
            dtype=dtype,
        )

    def apply(self, bundle):
        should_apply = self.sample_should_apply()
        offset = self._sample_offset(tf.float32)
        present_keys = self.iter_present_keys(bundle)

        for key in present_keys:
            tensor = bundle.data[key]
            bundle.data[key] = tf.cond(
                should_apply,
                lambda tensor=tensor: tensor + tf.cast(offset, tensor.dtype),
                lambda tensor=tensor: tensor,
            )

        self.record_transform(
            bundle,
            {
                "keys": list(present_keys),
                "offset": offset,
                "applied": should_apply,
            },
        )
        return bundle

    def inverse(self, bundle):
        trace = self._get_last_trace(bundle)
        if trace is None:
            return bundle

        applied = trace["params"]["applied"]
        offset = trace["params"]["offset"]
        keys = trace["params"]["keys"]

        for key in keys:
            if key not in bundle.data:
                continue
            tensor = bundle.data[key]
            bundle.data[key] = tf.cond(
                tf.cast(applied, tf.bool),
                lambda tensor=tensor: tensor - tf.cast(offset, tensor.dtype),
                lambda tensor=tensor: tensor,
            )
        return bundle
```

This pattern is useful when:

- the transform samples a different parameter every call
- the exact sampled value must be remembered
- inversion must use the sampled value from that specific forward pass

In other words, ``RandomTransform`` decides whether and how the transform
samples, while ``InvertibleTransform`` ensures that the sampled state is stored
for later restoration.

Testing tip:
When writing an invertible transform, add a unit test that checks
`forward -> inverse` restores the original tensor within a small tolerance,
for example by asserting `tf.reduce_max(tf.abs(original - restored)) < 1e-5`
for floating-point tensors.

### When `InvertibleTransform` is not the right tool

Not every transform should be made invertible. For example, a destructive
operation such as aggressive clipping or thresholding usually loses
information:

```python
class HardThreshold(KeyedTransform, InvertibleTransform):
    ...
    # Problem: once values are collapsed to 0 or 1, the original values are lost.
```

In that case, forcing inverse support would give a misleading API contract.


## Recommendation

If you are choosing a custom transform strategy today, the simplest guidance
is:

1. Start with ``LambdaTransform``.
2. Move to ``KeyedTransform`` when the transform deserves a reusable class.
3. Move to ``Transform`` when the logic is sample-level or metadata-aware.
4. Add ``RandomTransform`` behavior only when you are building a real random
   augmentation primitive.
5. Add ``InvertibleTransform`` behavior when predictions must come back to the
   original space.

That keeps the beginner path simple without blocking more advanced transform
design when the workflow grows.

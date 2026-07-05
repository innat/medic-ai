from __future__ import annotations

import inspect
import itertools
from typing import Any, Mapping, Sequence

import numpy as np
import tensorflow as tf

from .tensor_bundle import TensorBundle


def _convert_numpy_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    """Convert top-level NumPy arrays in a mapping to TensorFlow tensors."""
    if mapping is None:
        return {}

    converted = dict(mapping)
    for key, value in converted.items():
        if isinstance(value, np.ndarray):
            converted[key] = tf.convert_to_tensor(value)
    return converted


def ensure_tensor_bundle(
    inputs: TensorBundle | Mapping[str, Any], meta: Mapping[str, Any] | None = None
) -> TensorBundle:
    """Normalize transform inputs to a ``TensorBundle``.

    Args:
        inputs: Existing ``TensorBundle`` or a mapping of tensor-like values.
        meta: Optional metadata used only when ``inputs`` is a mapping.

    Returns:
        TensorBundle: A bundle containing tensor data and metadata.

    Raises:
        TypeError: If ``inputs`` is neither a ``TensorBundle`` nor a mapping.
        ValueError: If ``meta`` is provided together with a ``TensorBundle`` input.
    """
    if isinstance(inputs, TensorBundle):
        if meta is not None:
            raise ValueError("`meta` cannot be provided when `inputs` is already a TensorBundle.")
        return inputs

    if not isinstance(inputs, Mapping):
        raise TypeError("`inputs` must be a TensorBundle or a mapping of tensors.")

    return TensorBundle(_convert_numpy_mapping(inputs), _convert_numpy_mapping(meta))


def _trace_applied_to_bool(applied: tf.Tensor | bool) -> bool:
    """Convert a trace `applied` flag into a Python bool when possible."""
    if isinstance(applied, bool):
        return applied
    if tf.is_tensor(applied):
        static_value = tf.get_static_value(tf.cast(applied, tf.bool))
        if static_value is None:
            raise ValueError(
                "Cannot evaluate a symbolic `applied` trace flag outside eager execution."
            )
        return bool(static_value)
    return bool(applied)


def _pop_last_transform_trace(
    bundle: TensorBundle,
    transform_name: str,
    predicate=None,
) -> dict[str, Any] | None:
    """Pop the most recent matching transform trace from a bundle.

    This is important for inverse execution when a pipeline contains multiple
    instances of the same transform class. By consuming the most recent trace
    entry during inversion, each transform instance restores against the trace
    it produced most recently instead of repeatedly reusing the same entry.

    Args:
        bundle: Bundle containing applied transform traces.
        transform_name: Trace ``name`` field to match.
        predicate: Optional callable receiving the trace entry and returning
            ``True`` only for acceptable matches.

    Returns:
        Optional[dict[str, Any]]: The popped trace entry, or ``None`` when no
        matching trace exists.
    """
    applied = bundle.get_applied_transforms()
    for index in range(len(applied) - 1, -1, -1):
        entry = applied[index]
        if entry.get("name") != transform_name:
            continue
        if predicate is not None and not predicate(entry):
            continue
        return applied.pop(index)
    return None


def _normalize_keys(keys: Sequence[str] | str, name: str = "keys") -> tuple[str, ...]:
    """Normalize transform keys to a validated tuple of strings."""
    normalized = (keys,) if isinstance(keys, str) else tuple(keys)
    if not normalized:
        raise ValueError(f"`{name}` must contain at least one key.")
    if any(not isinstance(key, str) for key in normalized):
        raise TypeError(f"All entries in `{name}` must be strings.")
    return normalized


def _require_static_value(value: Any, name: str) -> Any:
    """Convert a TensorFlow scalar/tensor to a Python-visible value when possible."""
    if tf.is_tensor(value):
        static_value = tf.get_static_value(value)
        if static_value is None:
            raise ValueError(
                f"`{name}` must be statically knowable when used in this transform. "
                "This usually means the transform is being executed under graph mode "
                "with symbolic control flow that cannot be resolved to Python."
            )
        value = static_value
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class Transform:
    """Base class for Medic-AI transforms.

    ``Transform`` is the root abstraction of ``medicai.transforms``.
    Subclasses implement :meth:`apply` and receive a normalized
    :class:`~medicai.transforms.TensorBundle`, regardless of whether the user
    called the transform with a raw mapping or an existing bundle.

    This keeps input normalization, trace helpers, and inversion-related
    conventions in one place while allowing concrete transforms to focus on
    their transformation logic.

    When to use this:
        Use ``Transform`` when a custom transform needs to inspect or update
        the whole bundle, especially metadata such as ``affine`` or applied
        transform history. It is the best fit for orchestration-style
        transforms that do not naturally operate on a fixed set of tensor
        keys.

    Example:
        Define a simple metadata-aware transform:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import TensorBundle, Transform

            class MarkSample(Transform):
                def apply(self, bundle: TensorBundle) -> TensorBundle:
                    bundle["processed"] = True
                    bundle["image"] = tf.identity(bundle["image"])
                    return bundle

            image = tf.random.normal((64, 64, 1))
            output = MarkSample()({"image": image})
            print(output["processed"])
    """

    def __call__(
        self, inputs: TensorBundle | Mapping[str, Any], meta: Mapping[str, Any] | None = None
    ) -> TensorBundle:
        return self.apply(ensure_tensor_bundle(inputs, meta))

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        """Apply the transform to a ``TensorBundle``.

        Subclasses override this method with their forward transform logic.

        Args:
            bundle: The normalized bundle containing tensor data and optional
                metadata.

        Returns:
            TensorBundle: The updated bundle after the transform has been
            applied.
        """
        raise NotImplementedError

    @property
    def invertible(self) -> bool:
        """Whether the transform supports inverse execution."""
        return False

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        """Apply the inverse transform to a ``TensorBundle``.

        Invertible subclasses override this method when they can restore a
        previous sample state or geometry.

        Args:
            bundle: The bundle to restore.

        Returns:
            ``TensorBundle``: The bundle after inverse execution.

        Raises:
            NotImplementedError: If the transform does not support inversion.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement inverse transforms.")

    def build_trace_entry(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        applied: tf.Tensor | bool = True,
        random: bool = False,
        invertible: bool | None = None,
        kernel: str | None = None,
    ) -> dict[str, Any]:
        """Build a standardized transform trace entry.

        This helper centralizes the metadata format stored in
        ``bundle.meta["applied_transforms"]`` so transforms can record a
        consistent trace schema.

        Args:
            params: Optional transform-specific metadata to store.
            applied: Whether the transform was actually applied. Random
                transforms may store this as a TensorFlow boolean tensor.
            random: Whether the transform is stochastic.
            invertible: Optional override for the invertibility flag. When
                omitted, the transform's ``invertible`` property is used.
            kernel: Optional underlying kernel name, useful when a random
                transform wraps a deterministic implementation.

        Returns:
            dict[str, Any]: A standardized trace entry ready to be appended to
            the bundle metadata.
        """
        trace_entry = {
            "name": type(self).__name__,
            "params": dict(params or {}),
            "applied": applied,
            "random": random,
            "invertible": self.invertible if invertible is None else invertible,
        }
        if kernel is not None:
            trace_entry["kernel"] = kernel
        return trace_entry


class RandomTransform(Transform):
    """Base class for random TensorFlow-native transforms.

    ``RandomTransform`` adds probability-driven behavior on top of
    :class:`~medicai.transforms.Transform`. It is intended for transforms that
    sample whether to apply an operation using TensorFlow ops so the transform
    remains compatible with eager execution, ``tf.function``, and
    ``tf.data`` pipelines.

    Args:
        prob: Probability of applying the random transform. Must be in
            ``[0, 1]``.

    When to use this:
        Use ``RandomTransform`` when a transform needs probabilistic behavior
        implemented with TensorFlow ops so it stays compatible with
        ``tf.function`` and ``tf.data``. It is most useful as a base for
        random augmentations that decide whether to apply themselves per
        sample.

    Example:
        Build a tiny random transform that adds a bias to ``"image"``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomTransform, TensorBundle

            class RandomAddOne(RandomTransform):
                def apply(self, bundle: TensorBundle) -> TensorBundle:
                    should_apply = self.sample_should_apply()
                    image = bundle["image"]
                    bundle.data["image"] = tf.cond(
                        should_apply,
                        lambda: image + 1.0,
                        lambda: image,
                    )
                    self.record_random_transform(
                        bundle,
                        params={"keys": ["image"]},
                        applied=should_apply,
                    )
                    return bundle

            image = tf.zeros((32, 32, 1), dtype=tf.float32)
            output = RandomAddOne(prob=0.5)({"image": image})
            result = output['image']
    """

    def __init__(self, prob: float = 0.1):
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"`prob` must be in the range [0, 1]. Received {prob}.")
        self.prob = prob

    def sample_should_apply(self) -> tf.Tensor:
        """Sample whether the random transform should be applied.

        Returns:
            tf.Tensor: A scalar boolean tensor indicating whether the random
            transform should execute for the current sample.
        """
        return tf.random.uniform(shape=(), dtype=tf.float32) < self.prob

    def record_random_transform(
        self,
        bundle: TensorBundle,
        params: Mapping[str, Any] | None = None,
        applied: tf.Tensor | bool | None = None,
        kernel: str | None = None,
    ) -> TensorBundle:
        """Append a random transform trace entry to bundle metadata.

        Args:
            bundle: Bundle whose metadata should record the random transform.
            params: Optional transform-specific metadata to attach.
            applied: Whether the transform was applied. If omitted, ``True`` is
                recorded.
            kernel: Optional deterministic kernel name used internally.

        Returns:
            ``TensorBundle``: The same bundle, updated in place with one new trace
            entry.
        """
        bundle.push_transform(
            self.build_trace_entry(
                params=params,
                applied=True if applied is None else applied,
                random=True,
                kernel=kernel,
            )
        )
        return bundle


class RandomChoice(RandomTransform):
    """Randomly choose and apply a subset of transforms without replacement.

    ``RandomChoice`` is a transform orchestration utility similar in spirit to
    Albumentations ``OneOf`` and ``SomeOf``. On each call it optionally selects
    one or more child transforms, applies them sequentially in sampled order,
    and records which transforms were chosen so ``inverse()`` can walk back
    through the selected invertible transforms in reverse order.

    Selection is always performed without replacement. If users need the same
    transform to be eligible multiple times, they can include multiple
    instances of that transform in ``transforms``.

    For eager execution, ``RandomChoice`` supports the full API including
    multi-transform sampling and inverse bookkeeping. Under symbolic graph
    execution such as ``tf.data`` pipelines, ``RandomChoice`` uses a
    graph-safe forward path that statically unrolls up to ``max_choices``
    sequential dispatch steps with ``tf.switch_case``.

    This graph-safe path focuses on forward tensor transformation and assumes
    that every candidate transform preserves the same key structure, shape, and
    dtype per key across branches. It does not preserve eager-style wrapper
    trace bookkeeping used for ``inverse()``.

    When to use this:
        Use ``RandomChoice`` when an augmentation pipeline should sample from a
        pool of candidate transforms rather than always applying the same
        sequence. It is a good fit for "pick one of these" or "pick a few of
        these" style augmentation blocks.

    Args:
        transforms: Candidate transform objects to sample from.
        num_choices: Either one integer for an exact number of transforms to
            apply, or a ``(min, max)`` tuple specifying an inclusive range.
        prob: Probability of applying any sampled transforms at all. When the
            probability gate is not passed, the input is returned unchanged and
            no candidate transform is executed.
        weights: Optional relative sampling weights aligned with
            ``transforms``. Larger values increase the chance that a transform
            is selected. Weights are interpreted as relative preferences, not
            normalized probabilities, and sampling is still performed without
            replacement. A weight of ``0`` means that transform is never
            selected, and at least one weight must be positive when
            ``weights`` is provided. For example, ``weights=[0.7, 0.2, 0.1]``
            means the first transform is preferred over the second, and the
            second is preferred over the third; it does not mean the exact
            per-call probabilities are 70%, 20%, and 10%. Weights only affect
            behavior when ``RandomChoice`` has an actual choice to make, such
            as when ``num_choices < len(transforms)``. If all transforms are
            guaranteed to be selected, for example when
            ``num_choices == len(transforms)``, the weights do not materially
            change the outcome.

    Example:
        Pick exactly one geometric transform:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomChoice, RandomRotate, RandomRotate90

            transform = RandomChoice(
                transforms=[
                    RandomRotate90(keys=["image"], prob=1.0),
                    RandomRotate(keys=["image"], factor=0.2, prob=1.0),
                ],
                num_choices=1,
                prob=1.0,
            )

            image = tf.random.normal((32, 64, 64, 1))
            result = transform({"image": image})

        Pick between one and two transforms from a pool:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Flip, RandomChoice, ShiftIntensity

            transform = RandomChoice(
                transforms=[
                    Flip(keys=["image"], spatial_axis=0),
                    Flip(keys=["image"], spatial_axis=1),
                    ShiftIntensity(keys=["image"], offset=0.1),
                ],
                num_choices=(1, 2),
                weights=[1.0, 1.0, 0.5],
            )

            image = tf.random.normal((64, 64, 1))
            result = transform({"image": image})
    """

    def __init__(
        self,
        transforms: Sequence[Transform],
        num_choices: int | tuple[int, int] = 1,
        prob: float = 1.0,
        weights: Sequence[float] | None = None,
    ):
        super().__init__(prob=prob)
        self.transforms = tuple(transforms)
        if not self.transforms:
            raise ValueError("`transforms` must contain at least one transform.")
        if any(not callable(transform) for transform in self.transforms):
            raise TypeError("Every entry in `transforms` must be callable.")

        self.min_choices, self.max_choices = self._normalize_num_choices(num_choices)
        if self.max_choices > len(self.transforms):
            raise ValueError(
                f"`num_choices` cannot request more than {len(self.transforms)} transforms."
            )

        self.weights = self._normalize_weights(weights)
        if self.weights is not None:
            positive_count = sum(weight > 0.0 for weight in self.weights)
            if self.max_choices > positive_count:
                raise ValueError(
                    "`num_choices` cannot exceed the number of transforms with positive weight."
                )

    @property
    def invertible(self) -> bool:
        """Whether any candidate layer supports inverse execution."""
        return any(getattr(transform, "invertible", False) for transform in self.transforms)

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        if not tf.executing_eagerly():
            return self._apply_graph_choice(bundle)

        should_apply = self.sample_should_apply()
        should_apply_bool = _trace_applied_to_bool(should_apply)

        selected_indices: list[int] = []
        selected_names: list[str] = []

        if should_apply_bool:
            num_to_apply = self._sample_num_choices()
            num_to_apply = int(_require_static_value(num_to_apply, "num_choices"))

            if num_to_apply > 0:
                selected_indices = self._sample_indices(num_to_apply)
                selected_names = [
                    type(self.transforms[index]).__name__ for index in selected_indices
                ]
                for index in selected_indices:
                    bundle = self.transforms[index](bundle)

        self.record_random_transform(
            bundle,
            params={
                "selected_indices": list(selected_indices),
                "selected_names": list(selected_names),
                "num_selected": len(selected_indices),
                "num_choices": (self.min_choices, self.max_choices),
            },
            applied=bool(selected_indices),
            kernel="RandomChoice",
        )
        return bundle

    def _apply_graph_choice(self, bundle: TensorBundle) -> TensorBundle:
        data_keys = tuple(bundle.data.keys())
        if not data_keys:
            return bundle

        should_apply = self.sample_should_apply()
        permutation = self._sample_permutation_graph()
        num_to_apply = self._sample_num_choices()

        def apply_step(current_outputs, step: int):
            selected_index = permutation[step]

            def make_branch(index: int, current_outputs=current_outputs):
                def branch():
                    local_data = {
                        key: value for key, value in zip(data_keys, current_outputs, strict=True)
                    }
                    local_bundle = TensorBundle(local_data, dict(bundle.meta))
                    output = self.transforms[index](local_bundle)
                    return tuple(output.data[key] for key in data_keys)

                return branch

            branch_fns = {index: make_branch(index) for index in range(len(self.transforms))}
            return tf.switch_case(selected_index, branch_fns=branch_fns)

        current_outputs = tuple(bundle.data[key] for key in data_keys)
        for step in range(self.max_choices):
            current_outputs = tf.cond(
                tf.logical_and(should_apply, tf.constant(step, dtype=tf.int32) < num_to_apply),
                lambda current_outputs=current_outputs, step=step: apply_step(
                    current_outputs, step
                ),
                lambda current_outputs=current_outputs: current_outputs,
            )

        bundle.data = {key: value for key, value in zip(data_keys, current_outputs, strict=True)}
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        if not self.invertible:
            return bundle

        trace = self._get_last_random_choice_trace(bundle)
        if trace is None:
            return bundle

        for index in reversed(trace["params"].get("selected_indices", [])):
            transform = self.transforms[index]
            if getattr(transform, "invertible", False):
                bundle = transform.inverse(bundle)
        return bundle

    def _sample_num_choices(self) -> tf.Tensor:
        if self.min_choices == self.max_choices:
            return tf.constant(self.min_choices, dtype=tf.int32)
        return tf.random.uniform(
            shape=(),
            minval=self.min_choices,
            maxval=self.max_choices + 1,
            dtype=tf.int32,
        )

    def _sample_indices(self, num_to_apply: int) -> list[int]:
        if num_to_apply == 0:
            return []

        if self.weights is None:
            permutation = tf.random.shuffle(tf.range(len(self.transforms), dtype=tf.int32))
        else:
            weights = tf.convert_to_tensor(self.weights, dtype=tf.float32)
            uniforms = tf.random.uniform(shape=(len(self.transforms),), minval=1e-6, maxval=1.0)
            gumbels = -tf.math.log(-tf.math.log(uniforms))
            scores = tf.math.log(weights) + gumbels
            permutation = tf.argsort(scores, direction="DESCENDING")

        permutation = _require_static_value(permutation[:num_to_apply], "selected_indices")
        return [int(index) for index in permutation]

    def _sample_permutation_graph(self) -> tf.Tensor:
        """Graph-safe unique permutation of transform indices."""
        num_transforms = len(self.transforms)
        if self.weights is None:
            return tf.random.shuffle(tf.range(num_transforms, dtype=tf.int32))

        weights = tf.convert_to_tensor(self.weights, dtype=tf.float32)
        uniforms = tf.random.uniform(shape=(num_transforms,), minval=1e-6, maxval=1.0)
        gumbels = -tf.math.log(-tf.math.log(uniforms))
        valid = weights > 0.0
        safe_weights = tf.where(valid, weights, 1e-9)
        scores = tf.math.log(safe_weights) + gumbels
        scores = tf.where(valid, scores, -1e9)
        return tf.argsort(scores, direction="DESCENDING")

    def _normalize_num_choices(self, num_choices: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(num_choices, int):
            if num_choices < 0:
                raise ValueError("`num_choices` must be >= 0.")
            return num_choices, num_choices

        if not isinstance(num_choices, tuple) or len(num_choices) != 2:
            raise TypeError("`num_choices` must be an int or a `(min, max)` tuple.")

        min_choices, max_choices = num_choices
        if min_choices < 0 or max_choices < 0:
            raise ValueError("`num_choices` bounds must be >= 0.")
        if min_choices > max_choices:
            raise ValueError("`num_choices` requires `min <= max`.")
        return min_choices, max_choices

    def _normalize_weights(self, weights: Sequence[float] | None) -> tuple[float, ...] | None:
        if weights is None:
            return None
        normalized = tuple(float(weight) for weight in weights)
        if len(normalized) != len(self.transforms):
            raise ValueError("`weights` must have the same length as `transforms`.")
        if any(weight < 0.0 for weight in normalized):
            raise ValueError("`weights` must be non-negative.")
        if not any(weight > 0.0 for weight in normalized):
            raise ValueError("`weights` must contain at least one positive value.")
        return normalized

    def _get_last_random_choice_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)


class KeyedTransform(Transform):
    """Base class for transforms operating on a known set of data keys.

    ``KeyedTransform`` is the most common base class for Medic-AI transforms.
    It is designed for transforms that operate on a predefined set of keys
    such as ``"image"``, ``"label"``, or ``"mask"``.

    Args:
        keys: Keys of tensors this transform should process.
        allow_missing_keys: If ``True``, missing keys are skipped. If
            ``False``, missing keys raise ``KeyError``.

    When to use this:
        Use ``KeyedTransform`` when a transform acts on one or more known data
        entries such as ``"image"``, ``"label"``, or ``"mask"``. This is the
        default base class for deterministic per-key transforms because it
        handles missing-key policy and keyed tensor updates for you.

    Example:
        Multiply selected tensors by a constant:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import KeyedTransform, TensorBundle

            class Multiply(KeyedTransform):
                def __init__(self, keys, factor):
                    super().__init__(keys=keys)
                    self.factor = factor

                def apply(self, bundle: TensorBundle) -> TensorBundle:
                    self.apply_to_present_keys(
                        bundle,
                        lambda tensor, _: tensor * tf.cast(self.factor, tensor.dtype),
                    )
                    return bundle

            image = tf.ones((16, 16, 1), dtype=tf.float32)
            output = Multiply(keys=["image"], factor=2.0)({"image": image})
    """

    def __init__(self, keys: Sequence[str] | str, allow_missing_keys: bool = False):
        self.keys = _normalize_keys(keys)
        self.allow_missing_keys = allow_missing_keys

    def iter_present_keys(self, bundle: TensorBundle) -> list[str]:
        """Return the data keys present in ``bundle`` for this transform.

        Args:
            bundle: Bundle whose data mapping should be inspected.

        Returns:
            list[str]: Keys from ``self.keys`` that are present in
            ``bundle.data``.

        Raises:
            KeyError: If a requested key is missing and
                ``allow_missing_keys=False``.
        """
        present_keys = []
        for key in self.keys:
            if key in bundle.data:
                present_keys.append(key)
            elif not self.allow_missing_keys:
                raise KeyError(f"Key '{key}' not found in input data.")
        return present_keys

    def apply_to_present_keys(
        self,
        bundle: TensorBundle,
        fn,
        *,
        keys: Sequence[str] | None = None,
    ) -> list[str]:
        """Apply a tensor transform function to present keys in-place.

        Args:
            bundle: Bundle containing the tensors to update.
            fn: Callable receiving ``(tensor, key)`` and returning the updated tensor.
            keys: Optional subset of keys to process. Defaults to this transform's keys.

        Returns:
            list[str]: The keys that were present and updated.

        Raises:
            KeyError: If a requested key is missing and
                ``allow_missing_keys=False``.
        """
        target_keys = _normalize_keys(keys) if keys is not None else self.keys
        present_keys = []
        for key in target_keys:
            if key in bundle.data:
                present_keys.append(key)
            elif not self.allow_missing_keys:
                raise KeyError(f"Key '{key}' not found in input data.")

        for key in present_keys:
            bundle.data[key] = fn(bundle.data[key], key)
        return present_keys


class InvertibleTransform(Transform):
    """Base class for transforms that can record inversion metadata.

    ``InvertibleTransform`` marks transforms that can restore a previous sample
    state through :meth:`inverse`. In practice, most invertible transforms also
    record enough metadata during forward execution to reconstruct the original
    tensor layout, shape, or geometry later.

    Subclasses usually combine ``InvertibleTransform`` with either
    :class:`~medicai.transforms.KeyedTransform` or
    :class:`~medicai.transforms.Transform`.

    When to use this:
        Use ``InvertibleTransform`` when a transform can meaningfully undo its
        forward effect, such as restoring the original orientation, shape, or
        intensity adjustment. It is especially helpful for preprocessing steps
        that must later be reversed during post-processing.

    Example:
        Define a minimal additive invertible transform:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import (
                InvertibleTransform, KeyedTransform, TensorBundle
            )

            class AddValue(KeyedTransform, InvertibleTransform):
                def __init__(self, keys, value):
                    KeyedTransform.__init__(self, keys=keys)
                    self.value = value

                def apply(self, bundle: TensorBundle) -> TensorBundle:
                    self.apply_to_present_keys(
                        bundle,
                        lambda tensor, _: tensor + tf.cast(self.value, tensor.dtype),
                    )
                    self.record_transform(
                        bundle,
                        {
                            "keys": list(self.keys),
                            "value": self.value
                        }
                    )
                    return bundle

                def inverse(self, bundle: TensorBundle) -> TensorBundle:
                    self.apply_to_present_keys(
                        bundle,
                        lambda tensor, _: tensor - tf.cast(self.value, tensor.dtype),
                    )
                    return bundle

            image = tf.ones((8, 8, 1), dtype=tf.float32)
            transform = AddValue(keys=["image"], value=5.0)
            forward = transform(TensorBundle({"image": image}))
            restored = transform.inverse(forward)
    """

    @property
    def invertible(self) -> bool:
        return True

    def record_transform(
        self, bundle: TensorBundle, params: Mapping[str, Any] | None = None
    ) -> TensorBundle:
        """Append an invertible transform trace entry to bundle metadata.

        Args:
            bundle: Bundle whose metadata should record the transform.
            params: Optional transform-specific metadata needed for debugging
                or inverse execution.

        Returns:
            ``TensorBundle``: The same bundle, updated in place with one new trace
            entry.
        """
        bundle.push_transform(self.build_trace_entry(params=params, applied=True, random=False))
        return bundle


class LambdaTransform(KeyedTransform):
    """Apply callable-based keyed transforms with optional random and inverse behavior.

    ``LambdaTransform`` is a user-friendly transform wrapper for cases where
    defining a full transform subclass would be unnecessary overhead. It keeps
    Medic-AI's internal ``TensorBundle`` execution model while letting users
    provide simple tensor callables for forward and optional inverse execution.

    Args:
        keys: Keys of tensors to transform.
        fn: Callable applied to each selected tensor. It may accept either
            ``tensor`` or ``(tensor, key)``.
        prob: Optional probability of applying the transform. If ``None``, the
            transform is deterministic and always applies.
        inverse_fn: Optional callable used by :meth:`inverse`. It may accept
            either ``tensor`` or ``(tensor, key)``.
        meta_fn: Optional callable that receives a shallow copy of
            ``bundle.meta`` after forward execution and returns updated
            metadata. If it returns ``None``, in-place mutation is assumed.
        inverse_meta_fn: Optional callable mirroring ``meta_fn`` for inverse
            execution.
        allow_missing_keys: If ``True``, missing keys are skipped.
        name: Optional kernel name recorded in the transform trace.
        trace_params: Optional static trace parameters merged into the recorded
            trace entry.

    When to use this:
        Use ``LambdaTransform`` when users want a lightweight custom transform
        without defining a full subclass. It is a good fit for small
        deterministic or random tensor edits, optional inverse behavior, and
        simple metadata hooks.

    Example:
        Apply a callable to one key and optionally invert it later:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import LambdaTransform, TensorBundle

            transform = LambdaTransform(
                keys=["image"],
                fn=lambda tensor: tensor + 2.0,
                inverse_fn=lambda tensor: tensor - 2.0,
                name="add_two",
            )

            image = tf.ones((32, 32, 1), dtype=tf.float32)
            forward = transform({"image": image})
            restored = transform.inverse(forward)
            output = restored['image']

        Apply the same transform wrapper to multiple keys:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import LambdaTransform

            transform = LambdaTransform(
                keys=["image", "label"],
                fn=lambda tensor, key: (
                    tensor / 255.0
                    if key == "image"
                    else tf.cast(tensor, tf.float32)
                ),
                name="prepare_pair",
            )

            image = tf.ones((32, 32, 1), dtype=tf.float32) * 255.0
            label = tf.ones((32, 32, 1), dtype=tf.int32)
            output = transform(
                {
                    "image": image,
                    "label": label
                }
            )

        Apply a probabilistic callable and record its trace:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import LambdaTransform, TensorBundle

            transform = LambdaTransform(
                keys=["image"],
                fn=lambda tensor: tensor * 0.5,
                prob=0.5,
                trace_params={"kind": "scale"},
            )

            image = tf.ones((32, 32, 1), dtype=tf.float32)
            result = transform({"image": image})
            output = result['image']
    """

    _instance_counter = itertools.count()

    def __init__(
        self,
        keys: Sequence[str],
        fn,
        prob: float | None = None,
        inverse_fn=None,
        meta_fn=None,
        inverse_meta_fn=None,
        allow_missing_keys: bool = False,
        name: str | None = None,
        trace_params: Mapping[str, Any] | None = None,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        if prob is not None and not 0.0 <= prob <= 1.0:
            raise ValueError(f"`prob` must be in the range [0, 1]. Received {prob}.")

        self.fn = fn
        self.prob = prob
        self.inverse_fn = inverse_fn
        self.meta_fn = meta_fn
        self.inverse_meta_fn = inverse_meta_fn
        self.name = name
        self.trace_params = dict(trace_params or {})
        self._trace_id = f"lambda_{next(self._instance_counter)}"
        self._fn_takes_key = self._accepts_two_args(fn)
        self._inverse_fn_takes_key = self._accepts_two_args(inverse_fn)

    @property
    def invertible(self) -> bool:
        return self.inverse_fn is not None

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        should_apply: tf.Tensor | bool = True
        if self.prob is not None:
            should_apply = tf.random.uniform(shape=(), dtype=tf.float32) < self.prob

        present_keys = self.iter_present_keys(bundle)
        for key in present_keys:
            tensor = bundle.data[key]
            if self.prob is None:
                bundle.data[key] = self._call_tensor_fn(self.fn, tensor, key)
            else:
                bundle.data[key] = tf.cond(
                    should_apply,
                    lambda tensor=tensor, key=key: self._call_tensor_fn(self.fn, tensor, key),
                    lambda tensor=tensor: tensor,
                )

        if self.meta_fn is not None:
            try:
                should_update_meta = self.prob is None or _trace_applied_to_bool(should_apply)
            except ValueError:
                should_update_meta = False
            if should_update_meta:
                updated_meta = self.meta_fn(dict(bundle.meta))
                if updated_meta is not None:
                    bundle.meta = updated_meta

        bundle.push_transform(
            self.build_trace_entry(
                params={
                    "keys": list(present_keys),
                    "_lambda_id": self._trace_id,
                    **self.trace_params,
                },
                applied=should_apply,
                random=self.prob is not None,
                invertible=self.invertible,
                kernel=self.name,
            )
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        if self.inverse_fn is None:
            return super().inverse(bundle)

        trace = self._get_last_trace(bundle)
        if trace is None:
            return bundle

        applied = trace.get("applied", True)
        present_keys = [key for key in trace["params"].get("keys", []) if key in bundle.data]
        for key in present_keys:
            tensor = bundle.data[key]
            if tf.is_tensor(applied):
                bundle.data[key] = tf.cond(
                    tf.cast(applied, tf.bool),
                    lambda tensor=tensor, key=key: self._call_tensor_fn(
                        self.inverse_fn, tensor, key
                    ),
                    lambda tensor=tensor: tensor,
                )
            elif _trace_applied_to_bool(applied):
                bundle.data[key] = self._call_tensor_fn(self.inverse_fn, tensor, key)

        if self.inverse_meta_fn is not None:
            try:
                should_update_meta = _trace_applied_to_bool(applied)
            except ValueError:
                should_update_meta = False
            if should_update_meta:
                updated_meta = self.inverse_meta_fn(dict(bundle.meta))
                if updated_meta is not None:
                    bundle.meta = updated_meta
        return bundle

    def _get_last_trace(self, bundle: TensorBundle) -> dict[str, Any] | None:
        for entry in reversed(bundle.get_applied_transforms()):
            if entry.get("name") != type(self).__name__:
                continue
            if entry.get("params", {}).get("_lambda_id") == self._trace_id:
                return entry
        return None

    def _call_tensor_fn(self, fn, tensor: tf.Tensor, key: str) -> tf.Tensor:
        takes_key = self._fn_takes_key if fn is self.fn else self._inverse_fn_takes_key
        if takes_key:
            return fn(tensor, key)
        return fn(tensor)

    @staticmethod
    def _accepts_two_args(fn) -> bool:
        if fn is None:
            return False
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        positional = [
            param
            for param in signature.parameters.values()
            if param.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_var_positional = any(
            param.kind == inspect.Parameter.VAR_POSITIONAL
            for param in signature.parameters.values()
        )
        return len(positional) >= 2 or has_var_positional


class Compose(Transform):
    """Compose a sequence of transforms into one pipeline.

    ``Compose`` is the entry point for building a transformation pipeline in
    ``medicai.transforms``. It accepts raw sample dictionaries, converts any
    NumPy arrays into TensorFlow tensors, wraps the result in a
    ``TensorBundle``, and then applies each transform sequentially.

    This gives every transform a consistent container interface:

    - tensors are stored in the ``TensorBundle`` data mapping
    - optional metadata is stored in the ``TensorBundle`` metadata mapping
    - each transform reads from and writes back to the same container

    When to use this:
        Use ``Compose`` when multiple preprocessing or augmentation steps
        should run as a single pipeline. It is the standard way to define a
        reusable transform workflow for training, validation, or inference.

    Args:
        transforms (Sequence[callable]): A list or sequence of callable transform objects.
            Each transform in the list should accept a ``TensorBundle`` as input and
            return a modified ``TensorBundle``.

    Example:
        .. code-block:: python

            import numpy as np
            from medicai.transforms import (
                Compose,
                Resize,
                ScaleIntensityRange,
            )

            transform = Compose([
                ScaleIntensityRange(
                    keys=["image"],
                    input_min=-175,
                    input_max=250,
                    output_min=0.0,
                    output_max=1.0,
                    clip=True,
                ),
                Resize(
                    keys=["image", "label"],
                    target_shape=(96, 96, 96),
                    interpolation=("trilinear", "nearest")
                )
            ])

            image = np.random.randn(
                128, 128, 128, 1
            ).astype(np.float32)
            label = np.random.randint(
                0, 2, (128, 128, 128, 1)
            ).astype(np.float32)

            data = {
                "image": image,
                "label": label
            }
            output = transform(data)
            processed_image, processed_label = output["image"], output["label"]
            processed_image.shape, processed_label.shape
            # (TensorShape([96, 96, 96, 1]), TensorShape([96, 96, 96, 1]))

        Invert an already-applied pipeline when its transforms support
        ``inverse()``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Compose, Flip, Resize, TensorBundle

            pipeline = Compose(
                [
                    Flip(keys=["image"], spatial_axis=1),
                    Resize(
                        keys=["image"],
                        interpolation="bilinear",
                        target_shape=(32, 32)
                    ),
                ]
            )

            image = tf.random.normal((64, 64, 1))
            forward = pipeline(TensorBundle({"image": image}))
            restored = pipeline.inverse(forward)

    Returns:
        ``TensorBundle``: The transformed result, where the outputs are stored
        under the same keys as the input dictionary. For example, if the input
        contains keys such as ``image`` and ``label``, the transformed tensors
        can be retrieved from the returned bundle using those same keys.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    @property
    def invertible(self) -> bool:
        """Whether the composed pipeline contains any invertible transforms."""
        return any(getattr(transform, "invertible", False) for transform in self.transforms)

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        """Apply the composed transform pipeline to the input data.

        Args:
            bundle (TensorBundle): The input bundle to transform.

        Returns:
            TensorBundle: The transformed output. We can retrieve the processed
            tensors using the same keys as the input dictionary.
        """
        for transform in self.transforms:
            bundle = transform(bundle)
        return bundle

    def inverse(
        self, inputs: TensorBundle | Mapping[str, Any], meta: Mapping[str, Any] | None = None
    ) -> TensorBundle:
        """Apply inverse transforms in reverse order when available.

        Non-invertible transforms are skipped in this initial implementation.

        Args:
            inputs: Existing ``TensorBundle`` or a mapping of tensor-like values.
            meta: Optional metadata used only when ``inputs`` is a mapping.

        Returns:
            TensorBundle: The bundle after inverse execution of invertible
            transforms in reverse order.
        """
        bundle = ensure_tensor_bundle(inputs, meta)
        for transform in reversed(self.transforms):
            if getattr(transform, "invertible", False):
                bundle = transform.inverse(bundle)
        return bundle

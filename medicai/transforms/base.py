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


class Transform:
    """Base class for Medic-AI transforms.

    Subclasses implement :meth:`apply` and receive a normalized
    ``TensorBundle``. This keeps container conversion and shared behaviors in a
    single place while allowing concrete transforms to focus on transform
    logic.
    """

    def __call__(
        self, inputs: TensorBundle | Mapping[str, Any], meta: Mapping[str, Any] | None = None
    ) -> TensorBundle:
        return self.apply(ensure_tensor_bundle(inputs, meta))

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        """Apply the transform to a ``TensorBundle``."""
        raise NotImplementedError

    @property
    def invertible(self) -> bool:
        """Whether the transform supports inverse execution."""
        return False

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        """Apply the inverse transform to a ``TensorBundle``."""
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
        """Build a standardized transform trace entry."""
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
    """Base class for random TensorFlow-native transforms."""

    def __init__(self, prob: float = 0.1):
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"`prob` must be in the range [0, 1]. Received {prob}.")
        self.prob = prob

    def sample_should_apply(self) -> tf.Tensor:
        """Sample whether the random transform should be applied."""
        return tf.random.uniform(shape=(), dtype=tf.float32) < self.prob

    def record_random_transform(
        self,
        bundle: TensorBundle,
        params: Mapping[str, Any] | None = None,
        applied: tf.Tensor | bool | None = None,
        kernel: str | None = None,
    ) -> TensorBundle:
        """Append a random transform trace entry to bundle metadata."""
        bundle.push_transform(
            self.build_trace_entry(
                params=params,
                applied=True if applied is None else applied,
                random=True,
                kernel=kernel,
            )
        )
        return bundle


class KeyedTransform(Transform):
    """Base class for transforms operating on a known set of data keys."""

    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False):
        self.keys = tuple(keys)
        self.allow_missing_keys = allow_missing_keys

    def iter_present_keys(self, bundle: TensorBundle) -> list[str]:
        """Return the data keys present in ``bundle`` for this transform."""
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
        """
        original_keys = self.keys
        try:
            if keys is not None:
                self.keys = tuple(keys)
            present_keys = self.iter_present_keys(bundle)
        finally:
            self.keys = original_keys

        for key in present_keys:
            bundle.data[key] = fn(bundle.data[key], key)
        return present_keys


class InvertibleTransform(Transform):
    """Base class for transforms that can record inversion metadata."""

    @property
    def invertible(self) -> bool:
        return True

    def record_transform(
        self, bundle: TensorBundle, params: Mapping[str, Any] | None = None
    ) -> TensorBundle:
        """Append a transform trace entry to bundle metadata."""
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
        if trace is None or not trace.get("applied", True):
            return bundle

        present_keys = [key for key in trace["params"].get("keys", []) if key in bundle.data]
        for key in present_keys:
            bundle.data[key] = self._call_tensor_fn(self.inverse_fn, bundle.data[key], key)

        if self.inverse_meta_fn is not None:
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
        signature = inspect.signature(fn)
        positional = [
            param
            for param in signature.parameters.values()
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) >= 2:
            return fn(tensor, key)
        return fn(tensor)


class Compose(Transform):
    """
    Compose a sequence of transforms into one pipeline.

    ``Compose`` is the entry point for building a transformation pipeline in
    ``medicai.transforms``. It accepts raw sample dictionaries, converts any
    NumPy arrays into TensorFlow tensors, wraps the result in a
    ``TensorBundle``, and then applies each transform sequentially.

    This gives every transform a consistent container interface:

    - tensors are stored in the ``TensorBundle`` data mapping
    - optional metadata is stored in the ``TensorBundle`` metadata mapping
    - each transform reads from and writes back to the same container

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
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                Resize(
                    keys=["image", "label"],
                    spatial_shape=(96, 96, 96),
                    mode=("trilinear", "nearest")
                )
            ])

            image = np.random.randn(128, 128, 128, 1).astype(np.float32)
            label = np.random.randint(0, 2, (128, 128, 128, 1)).astype(np.float32)
            data = {
                "image": image,
                "label": label
            }
            output = transform(data)
            processed_image, processed_label = output["image"], output["label"]
            processed_image.shape, processed_label.shape
            # (TensorShape([96, 96, 96, 1]), TensorShape([96, 96, 96, 1]))

    Returns:
        ``TensorBundle``: The transformed result, where the outputs are stored
        under the same keys as the input dictionary. For example, if the input
        contains keys such as ``image`` and ``label``, the transformed tensors
        can be retrieved from the returned bundle using those same keys.
    """

    def __init__(self, transforms):
        self.transforms = transforms

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

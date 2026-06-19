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

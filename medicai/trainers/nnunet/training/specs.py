from dataclasses import dataclass
from typing import Mapping

import copy

import keras


@dataclass(frozen=True)
class OutputSpec:
    """Describe the output contract shared by training and inference.

    Attributes:
        names: Stable model output names, for example ``("final", "aux_0")``.
        scales: Spatial scale of each output relative to ``final``.
        channels: Number of channels produced by each output.
        activation: Output activation declared by the selected plan.
        target_encoding: Target representation used by the task.
    """

    names: tuple[str, ...]
    scales: Mapping[str, tuple[float, ...]]
    channels: Mapping[str, int]
    activation: str
    target_encoding: str

    def to_dict(self) -> dict:
        """Return a JSON-compatible representation of the output contract."""
        return {
            "names": list(self.names),
            "scales": {name: list(scale) for name, scale in self.scales.items()},
            "channels": dict(self.channels),
            "activation": self.activation,
            "target_encoding": self.target_encoding,
        }


class PerFoldFactory:
    """Create a fresh stateful object whenever a fold is initialized.

    Use this wrapper for custom Keras objects that are not registered with
    Keras serialization, for example ``PerFoldFactory(lambda: MyOptimizer())``.
    """

    def __init__(self, factory):
        if not callable(factory):
            raise TypeError("factory must be callable.")
        self.factory = factory

    def create(self):
        """Create one independent object."""
        return self.factory()


def clone_compile_value(value):
    """Clone a compile object or nested compile structure for one fold."""
    if isinstance(value, PerFoldFactory):
        return value.create()
    if isinstance(value, dict):
        return {key: clone_compile_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_compile_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_compile_value(item) for item in value)

    serializers = (
        (keras.optimizers.Optimizer, keras.optimizers.serialize, keras.optimizers.deserialize),
        (keras.losses.Loss, keras.losses.serialize, keras.losses.deserialize),
        (keras.metrics.Metric, keras.metrics.serialize, keras.metrics.deserialize),
    )
    for object_type, serialize, deserialize in serializers:
        if isinstance(value, object_type):
            try:
                return deserialize(serialize(value))
            except (TypeError, ValueError, KeyError):
                # Custom unregistered objects must use PerFoldFactory. A
                # deepcopy is still safer than sharing mutable state when it
                # is available.
                return copy.deepcopy(value)
    return value


def clone_callbacks(callbacks):
    """Return fresh callback instances for one training run."""
    if callbacks is None:
        return None
    cloned = []
    for callback in callbacks:
        if isinstance(callback, PerFoldFactory):
            cloned.append(callback.create())
            continue
        try:
            serialized = keras.saving.serialize_keras_object(callback)
            cloned.append(keras.saving.deserialize_keras_object(serialized))
        except (TypeError, ValueError, KeyError):
            cloned.append(copy.deepcopy(callback))
    return cloned

from dataclasses import dataclass
from typing import Mapping


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

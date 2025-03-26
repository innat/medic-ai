from typing import *

from medicai.utils.general import hide_warnings

hide_warnings()
import tensorflow as tf


class MetaTensor:
    def __init__(self, data: Dict[str, tf.Tensor], meta: Dict[str, Any] = None):
        self.data = data
        self.meta = meta or {}

    def __getitem__(self, key: str) -> Any:
        if key in self.data:
            return self.data[key]
        return self.meta[key]

    def __setitem__(self, key: str, value: Any):
        if key in self.data:
            self.data[key] = value
        else:
            self.meta[key] = value

    def __repr__(self) -> str:
        return f"MetaTensor(data={ {k: v.shape for k, v in self.data.items()} }, meta={self.meta})"

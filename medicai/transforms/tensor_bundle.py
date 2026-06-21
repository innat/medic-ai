from typing import Any, Dict, Optional

import tensorflow as tf


class TensorBundle:
    """Container for transform tensors and their associated metadata.

    ``TensorBundle`` stores the tensor mapping in ``data`` and any auxiliary
    information in ``meta``. Transforms in ``medicai.transforms`` read from and
    write back to this shared container so tensors and metadata stay aligned
    through the pipeline.

    Args:
        data (Dict[str, tf.Tensor]): Mapping from keys such as ``"image"`` or
            ``"label"`` to TensorFlow tensors.
        meta (Dict[str, Any], optional): Mapping of metadata values associated
            with the tensors. If ``None``, an empty metadata dictionary is
            created.

    Example:
        Create a bundle with image data and affine metadata::

            import tensorflow as tf
            from medicai.transforms import TensorBundle

            bundle = TensorBundle(
                data={"image": tf.random.normal((64, 64, 64, 1))},
                meta={"affine": tf.eye(4)},
            )

            image = bundle["image"]
            affine = bundle["affine"]

            print(image.shape) # (64, 64, 64, 1)
            print(affine.shape) # (4, 4)

    Returns:
        ``TensorBundle``: A container that stores tensors in ``data`` and metadata
        in ``meta``.
    """

    def __init__(self, data: Dict[str, tf.Tensor], meta: Dict[str, Any] = None):
        self.data = data
        self.meta = meta or {}

    def __getitem__(self, key: str) -> Any:
        """Access tensor data or metadata using dictionary-like key access.

        Args:
            key (str): Key to retrieve. If the key exists in ``data``, the
                corresponding tensor is returned. Otherwise, the lookup falls
                back to ``meta``.

        Returns:
            Any: The tensor or metadata value associated with ``key``.

        Raises:
            KeyError: If ``key`` is not found in either ``data`` or ``meta``.
        """
        if key in self.data:
            return self.data[key]
        return self.meta[key]

    def __setitem__(self, key: str, value: Any):
        """Set tensor data or metadata using dictionary-like assignment.

        Args:
            key (str): Key to update. If the key already exists in ``data``,
                the tensor value is replaced there. Otherwise, the key is
                stored in ``meta``.
            value (Any): Tensor or metadata value to store.
        """
        if key in self.data:
            self.data[key] = value
        else:
            self.meta[key] = value

    def get_data(self, key: str) -> Optional[tf.Tensor]:
        """Retrieve a tensor from the data mapping.

        Args:
            key (str): The key of the tensor to retrieve.

        Returns:
            Optional[tf.Tensor]: The tensor associated with ``key``, or
            ``None`` if the key is not present in ``data``.
        """
        return self.data.get(key)

    def get_meta(self, key: str) -> Optional[Any]:
        """Retrieve a metadata value from the metadata mapping.

        Args:
            key (str): The key of the metadata to retrieve.

        Returns:
            Optional[Any]: The metadata value associated with ``key``, or
            ``None`` if the key is not present in ``meta``.
        """
        return self.meta.get(key)

    def set_data(self, key: str, value: tf.Tensor):
        """Store a tensor in the data mapping.

        Args:
            key (str): The key to associate with the tensor.
            value (tf.Tensor): The TensorFlow tensor to store.
        """
        self.data[key] = value

    def set_meta(self, key: str, value: Any):
        """Store a metadata value in the metadata mapping.

        Args:
            key (str): The key to associate with the metadata value.
            value (Any): The metadata value to store.
        """
        self.meta[key] = value

    def get_applied_transforms(self) -> list[dict[str, Any]]:
        """Return the recorded transform trace list, creating it when needed."""
        trace = self.meta.get("applied_transforms")
        if trace is None:
            trace = []
            self.meta["applied_transforms"] = trace
        elif not isinstance(trace, list):
            raise TypeError("`meta['applied_transforms']` must be a list.")
        return trace

    def push_transform(self, trace_entry: dict[str, Any]):
        """Append one transform trace entry to metadata."""
        self.get_applied_transforms().append(trace_entry)

    def __repr__(self) -> str:
        """Provides a string representation of the TensorBundle.

        Returns:
            str: A string showing the shapes of the tensors in the data dictionary
            and the contents of the metadata dictionary.
        """
        return f"MetaTensor(data={ {k: v.shape for k, v in self.data.items()} }, meta={self.meta})"

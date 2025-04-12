from typing import Any, Dict, Optional

from medicai.utils.general import hide_warnings

hide_warnings()
import tensorflow as tf


class TensorBundle:
    """A container for holding a dictionary of TensorFlow tensors and associated metadata.

    This class is designed to bundle together the actual data (tensors) and any
    relevant metadata (e.g., affine transformations, original shapes, spacing)
    related to that data. It provides convenient access and modification methods
    for both the data and the metadata.
    """

    def __init__(self, data: Dict[str, tf.Tensor], meta: Dict[str, Any] = None):
        """Initializes a TensorBundle.

        Args:
            data (Dict[str, tf.Tensor]): A dictionary where keys are strings identifying
                the tensors (e.g., 'image', 'label') and values are the corresponding
                TensorFlow tensors.
            meta (Dict[str, Any], optional): A dictionary to store metadata associated
                with the data. Keys are strings and values can be of any type.
                Defaults to an empty dictionary if None is provided.
        """
        self.data = data
        self.meta = meta or {}

    def __getitem__(self, key: str) -> Any:
        """Allows accessing data or metadata using dictionary-like key access.

        Args:
            key (str): The key to retrieve. If the key exists in the data dictionary,
                the corresponding tensor is returned. Otherwise, if the key exists
                in the metadata dictionary, the corresponding metadata value is returned.

        Returns:
            Any: The tensor or metadata value associated with the given key.

        Raises:
            KeyError: If the key is not found in either the data or metadata dictionaries.
        """
        if key in self.data:
            return self.data[key]
        return self.meta[key]

    def __setitem__(self, key: str, value: Any):
        """Allows setting data or metadata using dictionary-like key assignment.

        Args:
            key (str): The key to set. If the key already exists in the data dictionary,
                the corresponding value is updated. Otherwise, the key-value pair is
                added to the metadata dictionary.
            value (Any): The TensorFlow tensor or metadata value to set.
        """
        if key in self.data:
            self.data[key] = value
        else:
            self.meta[key] = value

    def get_data(self, key: str) -> Optional[tf.Tensor]:
        """Retrieves a tensor from the data dictionary.

        Args:
            key (str): The key of the tensor to retrieve.

        Returns:
            Optional[tf.Tensor]: The TensorFlow tensor associated with the key,
            or None if the key is not found in the data dictionary.
        """
        return self.data.get(key)

    def get_meta(self, key: str) -> Optional[Any]:
        """Retrieves a metadata value from the metadata dictionary.

        Args:
            key (str): The key of the metadata to retrieve.

        Returns:
            Optional[Any]: The metadata value associated with the key,
            or None if the key is not found in the metadata dictionary.
        """
        return self.meta.get(key)

    def set_data(self, key: str, value: tf.Tensor):
        """Sets a tensor in the data dictionary.

        Args:
            key (str): The key to associate with the tensor.
            value (tf.Tensor): The TensorFlow tensor to store.
        """
        self.data[key] = value

    def set_meta(self, key: str, value: Any):
        """Sets a metadata value in the metadata dictionary.

        Args:
            key (str): The key to associate with the metadata value.
            value (Any): The metadata value to store.
        """
        self.meta[key] = value

    def __repr__(self) -> str:
        """Provides a string representation of the TensorBundle.

        Returns:
            str: A string showing the shapes of the tensors in the data dictionary
            and the contents of the metadata dictionary.
        """
        return f"MetaTensor(data={ {k: v.shape for k, v in self.data.items()} }, meta={self.meta})"

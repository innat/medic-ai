import numpy as np
import tensorflow as tf

from .tensor_bundle import TensorBundle


class Compose:
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
        self._to_tensor = lambda x: tf.convert_to_tensor(x)

    def __call__(self, image_data, meta_data=None):
        """Apply the composed transform pipeline to the input data.

        Args:
            image_data (dict): A sample dictionary containing the tensors to
                transform. Any NumPy arrays are converted to TensorFlow tensors
                before the pipeline is applied.
            meta_data (dict, optional): A dictionary containing any metadata associated
                with the image data. Any NumPy arrays in the metadata are also
                converted to TensorFlow tensors. Defaults to ``None``.

        Returns:
            TensorBundle: The transformed output. We can retrieve the processed
            tensors using the same keys as the input dictionary.
        """

        # Automatically convert all NumPy arrays in image_data to TensorFlow tensors
        for key, value in image_data.items():
            if isinstance(value, np.ndarray):
                image_data[key] = self._to_tensor(value)

        # Also convert any NumPy arrays in meta_data
        if meta_data is not None:
            for key, value in meta_data.items():
                if isinstance(value, np.ndarray):
                    meta_data[key] = self._to_tensor(value)

        x = TensorBundle(image_data, meta_data)
        for transform in self.transforms:
            x = transform(x)
        return x

from .tensor_bundle import TensorBundle


class Compose:
    """Composes several transforms together.

    This class allows you to chain multiple transformations to be
    applied sequentially to an input TensorBundle.
    """

    def __init__(self, transforms):
        """Initializes the Compose transform.

        Args:
            transforms (Sequence[callable]): A list or sequence of callable transform objects.
                Each transform in the list should accept a TensorBundle as input and
                return a modified TensorBundle.
        """
        self.transforms = transforms

    def __call__(self, image_data, meta_data=None):
        """Applies the sequence of transforms to the input data and metadata.

        Args:
            image_data (dict): A dictionary containing the image tensors, where keys
                are strings and values are TensorFlow tensors. This will be used to
                create the initial TensorBundle.
            meta_data (dict, optional): A dictionary containing any metadata associated
                with the image data. This will be used to create the initial TensorBundle.
                Defaults to None.

        Returns:
            TensorBundle: The result after applying all the transforms in the sequence.
        """
        x = TensorBundle(image_data, meta_data)
        for transform in self.transforms:
            x = transform(x)
        return x

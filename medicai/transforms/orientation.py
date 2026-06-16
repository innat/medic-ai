from typing import Dict, Sequence, Union

import tensorflow as tf

from .tensor_bundle import TensorBundle


class Orientation:
    """
    Reorient tensors to a target anatomical axis code.

    This transform reads the ``affine`` matrix stored in metadata, infers the
    current orientation, and then applies the required transpositions and flips
    to reach the requested ``axcodes`` such as ``"RAS"``. The transform is
    designed for tensors with three spatial axes and an optional channel-last
    dimension.

    Args:
        keys (Sequence[str]): Keys of the tensors to reorient.
        axcodes (str): The desired axis orientation as a 3-character string (e.g., ``RAS``, ``LPS``).
            The characters correspond to the anatomical directions:
            R: Right, L: Left, A: Anterior, P: Posterior, S: Superior, I: Inferior.
            Default is ``RAS`` (Right-Anterior-Superior).

    Example:
        Reorient an image-label pair to ``RAS``::

            import tensorflow as tf
            from medicai.transforms import Orientation, TensorBundle

            orient = Orientation(keys=["image", "label"], axcodes="RAS")

            image = tf.random.normal((32, 64, 64, 1))
            label = tf.random.uniform((32, 64, 64, 1), maxval=2, dtype=tf.int32)
            affine = tf.eye(4)

            bundle = TensorBundle(
                {"image": image, "label": label},
                meta={"affine": affine},
            )
            result = orient(bundle)
            reoriented_image = result["image"]
            reoriented_label = result["label"]

            print(reoriented_image.shape) # (32, 64, 64, 1)
            print(reoriented_label.shape) # (32, 64, 64, 1)

    Returns:
        ``TensorBundle``: The transformed output. We can retrieve the reoriented
        tensors using the same keys as the input.

    Raises:
        ValueError: If ``axcodes`` is not a 3-character orientation code.
        ValueError: If the input metadata does not contain an ``affine`` matrix.
    """

    def __init__(self, keys: Sequence[str] = ("image", "label"), axcodes: str = "RAS"):
        self.keys = keys
        self.axcodes = axcodes.upper()

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:
        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        affine = inputs.meta.get("affine")
        if affine is None:
            raise ValueError("Affine matrix is required for orientation transformation.")

        oriented_data = inputs.data.copy()
        for key in self.keys:
            if key in oriented_data:
                oriented_data[key] = self.apply_orientation(
                    oriented_data[key], affine, self.axcodes
                )

        return TensorBundle(oriented_data, inputs.meta)

    def apply_orientation(self, image: tf.Tensor, affine: tf.Tensor, axcodes: str) -> tf.Tensor:
        """
        Apply the orientation transformation to one tensor using the affine matrix.

        Args:
            image (tf.Tensor): The input image tensor (shape [..., channels]).
            affine (tf.Tensor): The affine transformation matrix (shape (4, 4)).
            axcodes (str): The desired axis orientation (e.g., "RAS").

        Returns:
            tf.Tensor: The reoriented image tensor.

        Raises:
            ValueError: If `axcodes` is not a 3-character string.
        """
        axis_map = {"R": 0, "A": 1, "S": 2, "L": 0, "P": 1, "I": 2}
        flip_map = {"R": False, "A": False, "S": False, "L": True, "P": True, "I": True}

        if len(axcodes) != 3:
            raise ValueError("axcodes must be a 3-character string.")

        axis_order = [axis_map[c] for c in axcodes]
        flip_flags = [flip_map[c] for c in axcodes]

        # Calculate the current orientation from the affine matrix
        current_orientation = self.get_orientation_from_affine(affine)

        # Calculate the desired permutation and flip flags
        permutation = self.calculate_permutation(current_orientation, axcodes)
        flip_flags = self.calculate_flip_flags(current_orientation, axcodes)

        # Transpose
        transpose_order = [permutation.index(i) for i in range(3)]
        if len(image.shape) == 4:
            transpose_order = transpose_order + [3]  # Add channel dimension back.
        reoriented_image = tf.transpose(image, perm=transpose_order)

        # Flip
        for i, flip in enumerate(flip_flags):
            if flip:
                reoriented_image = tf.reverse(reoriented_image, axis=[i])

        return reoriented_image

    def get_orientation_from_affine(self, affine: tf.Tensor) -> str:
        """Calculates the orientation from the affine matrix."""
        orientation = ""
        for i in range(3):
            direction = affine[:3, i]
            orientation += self.get_axis_code(direction)
        return orientation

    def get_axis_code(self, direction: tf.Tensor) -> str:
        axis = tf.argmax(tf.abs(direction))
        axis_int = tf.cast(axis, tf.int32)

        def get_code(axis_val):
            if tf.gather(direction, [axis_val])[0] > 0:
                return tf.gather(tf.constant(list("RAS")), axis_val)
            else:
                return tf.gather(tf.constant(list("LPI")), axis_val)

        return tf.strings.reduce_join([get_code(axis_int)])

    def calculate_permutation(
        self, current_orientation: str, target_orientation: str
    ) -> Sequence[int]:
        """Calculates the permutation needed to change from current to target orientation."""
        permutation = []
        axis_map = {"R": 0, "A": 1, "S": 2, "L": 0, "P": 1, "I": 2}

        for target_axis in target_orientation:
            permutation.append(axis_map[target_axis])
        return permutation

    def calculate_flip_flags(
        self, current_orientation: str, target_orientation: str
    ) -> Sequence[bool]:
        flip_flags = []
        for i in range(3):
            current_axis = tf.strings.substr(current_orientation, i, 1)
            target_axis = tf.strings.substr(target_orientation, i, 1)
            flip_flags.append(
                tf.logical_not(
                    tf.equal(
                        tf.strings.regex_full_match(current_axis, r"[RAS]"),
                        tf.strings.regex_full_match(target_axis, r"[RAS]"),
                    )
                )
            )
        return flip_flags

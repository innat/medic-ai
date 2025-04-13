from typing import Dict, Sequence, Union

from medicai.utils.general import hide_warnings

hide_warnings()
import tensorflow as tf

from .tensor_bundle import TensorBundle


class Orientation:
    """Reorients a volume to a specified axis orientation using the affine matrix.

    This transform takes a tensor bundle containing an image and its affine
    matrix and reorients the image data to the desired 'axcodes' (e.g., 'RAS').
    It determines the current orientation from the affine matrix and calculates
    the necessary transpositions and flips to achieve the target orientation.
    """

    def __init__(self, keys: Sequence[str] = ("image", "label"), axcodes: str = "RAS"):
        """
        Initializes the Orientation transform.

        Args:
            keys (Sequence[str]): Keys of the tensors to reorient. Default is ("image", "label").
            axcodes (str): The desired axis orientation as a 3-character string (e.g., "RAS", "LPS").
                The characters correspond to the anatomical directions:
                R: Right, L: Left, A: Anterior, P: Posterior, S: Superior, I: Inferior.
                Default is "RAS" (Right-Anterior-Superior).
        """
        self.keys = keys
        self.axcodes = axcodes.upper()

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:
        """
        Apply the orientation transformation to the input TensorBundle.

        Args:
            inputs (TensorBundle): A dictionary containing tensors and metadata,
                where the metadata is expected to contain an 'affine' key
                representing the affine transformation matrix.

        Returns:
            TensorBundle: A dictionary with reoriented tensors and the original metadata.

        Raises:
            ValueError: If the 'affine' matrix is not found in the input metadata.
        """

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
        Applies orientation transformation to an image, considering the affine matrix.

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

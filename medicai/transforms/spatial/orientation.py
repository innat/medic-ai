from typing import Sequence

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import get_spatial_rank


class Orientation(KeyedTransform, InvertibleTransform):
    """Reorient tensors to a target anatomical axis code.

    This transform remains explicitly 3D-only and expects channel-last sample
    tensors shaped like ``(D, H, W, C)``.
    """

    _AXIS_TO_WORLD = {"R": 0, "L": 0, "A": 1, "P": 1, "S": 2, "I": 2}
    _AXIS_TO_SIGN = {"R": 1, "L": -1, "A": 1, "P": -1, "S": 1, "I": -1}
    _POSITIVE_CODES = ("R", "A", "S")
    _NEGATIVE_CODES = ("L", "P", "I")

    def __init__(
        self,
        keys: Sequence[str] = ("image", "label"),
        axcodes: str = "RAS",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        axcodes = axcodes.upper()
        if len(axcodes) != 3:
            raise ValueError("axcodes must be a 3-character string.")
        self.axcodes = axcodes

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        affine = bundle.meta.get("affine")
        if affine is None:
            raise ValueError("Affine matrix is required for orientation transformation.")

        current_orientation = self.get_orientation_from_affine(affine)

        def apply_orientation(tensor: tf.Tensor, _: str) -> tf.Tensor:
            spatial_rank = get_spatial_rank(tensor)
            if spatial_rank != 3:
                raise ValueError(
                    f"Orientation currently supports only 3D tensors; got spatial rank "
                    f"{spatial_rank} for shape {tensor.shape}."
                )
            return self.orient_tensor(tensor, affine, self.axcodes)

        present_keys = self.apply_to_present_keys(bundle, apply_orientation)
        bundle.meta["affine"] = self.reoriented_affine(affine, self.axcodes)
        self.record_transform(
            bundle,
            {
                "keys": list(present_keys),
                "original_axcodes": current_orientation,
                "target_axcodes": self.axcodes,
            },
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = self._get_last_orientation_trace(bundle)
        if trace is None:
            return bundle

        original_axcodes = trace["params"].get("original_axcodes")
        if original_axcodes is None:
            return bundle

        affine = bundle.meta.get("affine")
        if affine is None:
            return bundle

        def apply_inverse_orientation(tensor: tf.Tensor, _: str) -> tf.Tensor:
            return self.orient_tensor(tensor, affine, original_axcodes)

        present_keys = [key for key in trace["params"].get("keys", []) if key in bundle.data]
        self.apply_to_present_keys(bundle, apply_inverse_orientation, keys=present_keys)
        bundle.meta["affine"] = self.reoriented_affine(affine, original_axcodes)
        return bundle

    def orient_tensor(self, tensor: tf.Tensor, affine: tf.Tensor, target_axcodes: str) -> tf.Tensor:
        """Reorient one tensor using affine-derived permutation and flips."""
        matrix = tf.cast(affine[:3, :3], tf.float32)
        current_axes = tf.argmax(tf.abs(matrix), axis=0, output_type=tf.int32)
        gather_indices = tf.stack([current_axes, tf.range(3, dtype=tf.int32)], axis=1)
        current_signs = tf.sign(tf.gather_nd(matrix, gather_indices))
        current_signs = tf.where(current_signs == 0, tf.ones_like(current_signs), current_signs)

        target_axes = tf.constant([self._AXIS_TO_WORLD[c] for c in target_axcodes], dtype=tf.int32)
        target_signs = tf.constant([self._AXIS_TO_SIGN[c] for c in target_axcodes], dtype=tf.float32)

        perm_spatial = tf.stack(
            [
                tf.argmax(tf.cast(tf.equal(current_axes, target_axis), tf.int32), output_type=tf.int32)
                for target_axis in tf.unstack(target_axes)
            ]
        )
        perm = tf.concat([perm_spatial, [3]], axis=0)
        reoriented = tf.transpose(tensor, perm=perm)

        current_signs_for_output = tf.gather(current_signs, perm_spatial)
        flip_axes = tf.cast(tf.reshape(tf.where(tf.not_equal(current_signs_for_output, target_signs)), [-1]), tf.int32)
        return tf.reverse(reoriented, axis=flip_axes)

    def get_orientation_from_affine(self, affine: tf.Tensor) -> str:
        """Infer a three-letter orientation code from a 4x4 affine matrix."""
        matrix = tf.cast(affine[:3, :3], tf.float32)
        codes = []
        for axis_index in range(3):
            direction = matrix[:, axis_index]
            dominant_axis = int(tf.argmax(tf.abs(direction), output_type=tf.int32))
            sign = tf.gather(direction, dominant_axis) >= 0
            if dominant_axis == 0:
                codes.append("R" if bool(sign) else "L")
            elif dominant_axis == 1:
                codes.append("A" if bool(sign) else "P")
            else:
                codes.append("S" if bool(sign) else "I")
        return "".join(codes)

    def reoriented_affine(self, affine: tf.Tensor, target_axcodes: str) -> tf.Tensor:
        """Build a simple target-orientation affine preserving axis spacing."""
        affine = tf.cast(affine, tf.float32)
        spacing = tf.stack([tf.norm(affine[:3, i]) for i in range(3)], axis=0)
        target_signs = tf.constant([self._AXIS_TO_SIGN[c] for c in target_axcodes], dtype=tf.float32)
        diagonal = spacing * target_signs
        new_affine = tf.eye(4, dtype=tf.float32)
        new_affine = tf.tensor_scatter_nd_update(
            new_affine,
            indices=[[0, 0], [1, 1], [2, 2]],
            updates=diagonal,
        )
        return new_affine

    def _get_last_orientation_trace(self, bundle: TensorBundle) -> dict | None:
        for entry in reversed(bundle.get_applied_transforms()):
            if entry.get("name") == type(self).__name__:
                return entry
        return None

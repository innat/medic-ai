from typing import Sequence

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import get_spatial_rank


class Orientation(KeyedTransform, InvertibleTransform):
    """Reorient tensors to a target anatomical axis code.

    This transform remains explicitly 3D-only and expects channel-last sample
    tensors shaped like ``(D, H, W, C)``.

    To preserve the library's fixed volumetric tensor layout contract, this
    implementation applies axis flips but does not permute spatial axes. If an
    affine implies that reaching ``axcodes`` would require swapping spatial
    axes, the transform raises a ``ValueError`` instead of returning a tensor
    whose layout changes from ``(D, H, W, C)`` to another axis order.
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
        bundle.meta["affine"] = self.reoriented_affine(affine, current_orientation, self.axcodes)
        self.record_transform(
            bundle,
            {
                "keys": list(present_keys),
                "original_affine": tf.identity(tf.cast(affine, tf.float32)),
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
        original_affine = trace["params"].get("original_affine")
        if original_axcodes is None or original_affine is None:
            return bundle

        affine = bundle.meta.get("affine")
        if affine is None:
            return bundle

        def apply_inverse_orientation(tensor: tf.Tensor, _: str) -> tf.Tensor:
            return self.orient_tensor(tensor, affine, original_axcodes)

        present_keys = [key for key in trace["params"].get("keys", []) if key in bundle.data]
        self.apply_to_present_keys(bundle, apply_inverse_orientation, keys=present_keys)
        bundle.meta["affine"] = tf.cast(original_affine, tf.float32)
        return bundle

    def orient_tensor(self, tensor: tf.Tensor, affine: tf.Tensor, target_axcodes: str) -> tf.Tensor:
        """Reorient one tensor using affine-derived flips while preserving layout."""
        current_orientation = self.get_orientation_from_affine(affine)
        self._validate_orientation_alignment(current_orientation, target_axcodes)
        flip_axes = self._get_flip_axes(current_orientation, target_axcodes)
        if not flip_axes:
            return tensor
        return tf.reverse(tensor, axis=list(flip_axes))

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

    def reoriented_affine(
        self,
        affine: tf.Tensor,
        current_axcodes: str,
        target_axcodes: str,
    ) -> tf.Tensor:
        """Update affine signs to match a flip-only orientation change."""
        affine = tf.cast(affine, tf.float32)
        self._validate_orientation_alignment(current_axcodes, target_axcodes)
        new_affine = tf.identity(affine)
        flip_axes = self._get_flip_axes(current_axcodes, target_axcodes)
        if not flip_axes:
            return new_affine

        indices = [[row, axis] for axis in flip_axes for row in range(3)]
        updates = [-affine[row, axis] for axis in flip_axes for row in range(3)]
        new_affine = tf.tensor_scatter_nd_update(
            new_affine,
            indices=indices,
            updates=tf.cast(updates, tf.float32),
        )
        return new_affine

    def _validate_orientation_alignment(
        self,
        current_axcodes: str,
        target_axcodes: str,
    ) -> None:
        current_world_axes = [self._AXIS_TO_WORLD[c] for c in current_axcodes]
        target_world_axes = [self._AXIS_TO_WORLD[c] for c in target_axcodes]
        if current_world_axes != target_world_axes:
            raise ValueError(
                "Orientation preserves the (D, H, W, C) layout and currently supports "
                "flip-only reorientation. The provided affine requires spatial-axis "
                f"permutation from '{current_axcodes}' to '{target_axcodes}'."
            )

    def _get_flip_axes(
        self,
        current_axcodes: str,
        target_axcodes: str,
    ) -> tuple[int, ...]:
        return tuple(
            axis
            for axis, (current_code, target_code) in enumerate(zip(current_axcodes, target_axcodes))
            if self._AXIS_TO_SIGN[current_code] != self._AXIS_TO_SIGN[target_code]
        )

    def _get_last_orientation_trace(self, bundle: TensorBundle) -> dict | None:
        for entry in reversed(bundle.get_applied_transforms()):
            if entry.get("name") == type(self).__name__:
                return entry
        return None

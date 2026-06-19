from typing import Sequence

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import get_spatial_rank


class Orientation(KeyedTransform, InvertibleTransform):
    """Reorient tensors to a target anatomical axis code.

    This transform remains explicitly 3D-only and expects channel-last sample
    tensors shaped like ``(D, H, W, C)``.

    User-facing ``axcodes`` follow the usual anatomical convention order
    ``(R/L, A/P, S/I)``. Internally, Medic-AI keeps tensors depth-first, so the
    target tensor-axis orientation is interpreted in ``(D, H, W)`` order,
    corresponding to ``(S/I, A/P, R/L)`` for an ``RAS`` target.
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

        sample_key = None
        for key in self.keys:
            if key in bundle.data:
                sample_key = key
                break
        if sample_key is None:
            if self.allow_missing_keys:
                return bundle
            raise KeyError(f"None of the keys {self.keys} were found in input data.")

        current_orientation = self.get_orientation_from_affine(affine)
        sample_tensor = bundle.data[sample_key]
        spatial_rank = get_spatial_rank(sample_tensor)
        if spatial_rank != 3:
            raise ValueError(
                f"Orientation currently supports only 3D tensors; got spatial rank "
                f"{spatial_rank} for shape {sample_tensor.shape}."
            )
        target_tensor_axcodes = self._target_tensor_axcodes(self.axcodes)
        transform_info = self._compute_orientation_transform(affine, target_tensor_axcodes)

        def apply_orientation(tensor: tf.Tensor, _: str) -> tf.Tensor:
            spatial_rank = get_spatial_rank(tensor)
            if spatial_rank != 3:
                raise ValueError(
                    f"Orientation currently supports only 3D tensors; got spatial rank "
                    f"{spatial_rank} for shape {tensor.shape}."
                )
            return self.orient_tensor(tensor, transform_info["perm_spatial"], transform_info["flip_axes"])

        present_keys = self.apply_to_present_keys(bundle, apply_orientation)
        bundle.meta["affine"] = self.reoriented_affine(
            affine,
            tf.shape(sample_tensor)[:3],
            transform_info["perm_spatial"],
            transform_info["flip_axes"],
        )
        self.record_transform(
            bundle,
            {
                "keys": list(present_keys),
                "original_affine": tf.identity(tf.cast(affine, tf.float32)),
                "original_axcodes": current_orientation,
                "target_axcodes": self.axcodes,
                "target_tensor_axcodes": target_tensor_axcodes,
                "perm_spatial": list(transform_info["perm_spatial"]),
                "flip_axes": list(transform_info["flip_axes"]),
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

        perm_spatial = trace["params"].get("perm_spatial")
        flip_axes = trace["params"].get("flip_axes")
        if perm_spatial is None or flip_axes is None:
            return bundle
        perm_spatial = tuple(int(axis) for axis in perm_spatial)
        flip_axes = tuple(int(axis) for axis in flip_axes)

        def apply_inverse_orientation(tensor: tf.Tensor, _: str) -> tf.Tensor:
            return self.inverse_orient_tensor(tensor, perm_spatial, flip_axes)

        present_keys = [key for key in trace["params"].get("keys", []) if key in bundle.data]
        self.apply_to_present_keys(bundle, apply_inverse_orientation, keys=present_keys)
        bundle.meta["affine"] = tf.cast(original_affine, tf.float32)
        return bundle

    def orient_tensor(
        self,
        tensor: tf.Tensor,
        perm_spatial: tuple[int, int, int],
        flip_axes: tuple[int, ...],
    ) -> tf.Tensor:
        """Reorient one tensor using a spatial permutation followed by flips."""
        reoriented = tf.transpose(tensor, perm=[*perm_spatial, 3])
        if not flip_axes:
            return reoriented
        return tf.reverse(reoriented, axis=list(flip_axes))

    def inverse_orient_tensor(
        self,
        tensor: tf.Tensor,
        perm_spatial: tuple[int, int, int],
        flip_axes: tuple[int, ...],
    ) -> tf.Tensor:
        """Invert a spatial permutation and flips applied by ``orient_tensor``."""
        restored = tf.reverse(tensor, axis=list(flip_axes)) if flip_axes else tensor
        inverse_perm_spatial = [0, 0, 0]
        for output_axis, input_axis in enumerate(perm_spatial):
            inverse_perm_spatial[input_axis] = output_axis
        return tf.transpose(restored, perm=[*inverse_perm_spatial, 3])

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
        input_spatial_shape: tf.Tensor,
        perm_spatial: tuple[int, int, int],
        flip_axes: tuple[int, ...],
    ) -> tf.Tensor:
        """Update affine metadata for a spatial permutation and flips."""
        affine = tf.cast(affine, tf.float32)
        input_spatial_shape = tf.cast(input_spatial_shape, tf.float32)
        transform = tf.eye(4, dtype=tf.float32)
        for output_axis, input_axis in enumerate(perm_spatial):
            sign = -1.0 if output_axis in flip_axes else 1.0
            column = tf.constant([[0.0], [0.0], [0.0], [0.0]], dtype=tf.float32)
            column = tf.tensor_scatter_nd_update(
                column,
                indices=[[input_axis, 0]],
                updates=[sign],
            )
            transform = tf.tensor_scatter_nd_update(
                transform,
                indices=[[row, output_axis] for row in range(4)],
                updates=tf.reshape(column, [-1]),
            )
            if sign < 0:
                transform = tf.tensor_scatter_nd_update(
                    transform,
                    indices=[[input_axis, 3]],
                    updates=[input_spatial_shape[input_axis] - 1.0],
                )
        return tf.linalg.matmul(affine, transform)

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

    def _compute_orientation_transform(
        self,
        affine: tf.Tensor,
        target_tensor_axcodes: str,
    ) -> dict[str, tuple[int, ...]]:
        matrix = tf.cast(affine[:3, :3], tf.float32)
        current_axes = tf.argmax(tf.abs(matrix), axis=0, output_type=tf.int32)
        gather_indices = tf.stack([current_axes, tf.range(3, dtype=tf.int32)], axis=1)
        current_signs = tf.sign(tf.gather_nd(matrix, gather_indices))
        current_signs = tf.where(current_signs == 0, tf.ones_like(current_signs), current_signs)

        target_axes = [self._AXIS_TO_WORLD[c] for c in target_tensor_axcodes]
        perm_spatial = tuple(
            int(tf.argmax(tf.cast(tf.equal(current_axes, target_axis), tf.int32), output_type=tf.int32))
            for target_axis in target_axes
        )
        current_signs_for_output = tf.gather(
            current_signs,
            tf.constant(perm_spatial, dtype=tf.int32),
        )
        target_signs = tf.constant(
            [self._AXIS_TO_SIGN[c] for c in target_tensor_axcodes], dtype=tf.float32
        )
        flip_axes = tuple(
            int(axis)
            for axis in tf.reshape(
                tf.where(tf.not_equal(current_signs_for_output, target_signs)),
                [-1],
            ).numpy().tolist()
        )
        return {"perm_spatial": perm_spatial, "flip_axes": flip_axes}

    def _target_tensor_axcodes(self, axcodes: str) -> str:
        """Map anatomical axis-code order to Medic-AI's depth-first tensor order."""
        return axcodes[::-1]

    def _get_last_orientation_trace(self, bundle: TensorBundle) -> dict | None:
        for entry in reversed(bundle.get_applied_transforms()):
            if entry.get("name") == type(self).__name__:
                return entry
        return None

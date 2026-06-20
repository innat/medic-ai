from typing import Sequence

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import get_spatial_rank


class Orientation(KeyedTransform, InvertibleTransform):
    """Reorient tensors to a target anatomical axis code.

    ``Orientation`` reorders and flips volumetric tensors so their voxel axes
    match a requested anatomical orientation such as ``"RAS"`` while preserving
    Medic-AI's internal tensor layout convention of depth-first,
    channel-last data.

    In Medic-AI, 3D sample tensors are expected to use the shape
    ``(D, H, W, C)``, where:

    - ``D`` is the depth axis
    - ``H`` is the height axis
    - ``W`` is the width axis
    - ``C`` is the channel axis

    The associated ``affine`` metadata must describe that same voxel order.
    This means any loader that converts file-native arrays into Medic-AI's
    ``(D, H, W, C)`` layout must also reorder the affine consistently before
    calling this transform.

    User-facing ``axcodes`` follow standard anatomical orientation order
    ``(R/L, A/P, S/I)``. Internally, because Medic-AI tensors are depth-first,
    the requested target is translated into the corresponding tensor-axis order
    ``(D, H, W) -> (S/I, A/P, R/L)``. For example, ``axcodes="RAS"`` means
    that after reorientation the tensor's depth axis corresponds to
    superior-inferior, height corresponds to anterior-posterior, and width
    corresponds to right-left.

    This transform is invertible. During ``apply()``, it records the original
    affine and axis mapping into the ``TensorBundle`` transform trace so
    ``inverse()`` can restore both the tensor layout and affine metadata.

    Args:
        keys: Keys of tensors in the bundle to reorient. Each selected tensor
            must be a 3D channel-last sample with shape ``(D, H, W, C)``.
        axcodes: Target anatomical orientation code written in standard medical
            imaging convention, such as ``"RAS"``, ``"LPS"``, or ``"LAS"``.
            The string must contain exactly three characters.
        allow_missing_keys: If ``True``, missing keys are skipped. If ``False``,
            missing requested keys raise an error.

    Example:
        Reorient an image-label pair to ``RAS`` and then restore the original
        orientation:

        .. code-block:: python

            import numpy as np
            import tensorflow as tf
            from medicai.transforms import Orientation, TensorBundle

            orient = Orientation(keys=["image", "label"], axcodes="RAS")
            image = tf.random.normal((32, 64, 64, 1))
            label = tf.random.uniform((32, 64, 64, 1), maxval=2, dtype=tf.int32)
            affine = tf.constant(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=tf.float32,
            )

            bundle = TensorBundle(
                {
                    "image": image,
                    "label": label
                },
                meta={
                    "affine": affine
                },
            )

            forward = orient(bundle)
            restored = orient.inverse(forward)
            print(np.all(restored["image"].numpy() == image.numpy())) # True

    Returns:
        ``TensorBundle``: The input bundle with selected tensors reoriented in
        place, updated ``affine`` metadata, and an invertible transform trace
        appended to ``bundle.meta["applied_transforms"]``.

    Raises:
        ValueError: If ``axcodes`` is not a 3-character string.
        ValueError: If ``affine`` metadata is missing.
        ValueError: If any selected tensor is not 3D spatially.
        KeyError: If none of the requested keys are present and ``allow_missing_keys=False``.
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

        self._validate_bundle_is_3d(bundle)
        current_orientation = self.get_orientation_from_affine(affine)
        sample_tensor = bundle.data[sample_key]
        target_tensor_axcodes = self._target_tensor_axcodes(self.axcodes)
        transform_info = self._compute_orientation_transform(affine, target_tensor_axcodes)

        def apply_orientation(tensor: tf.Tensor, _: str) -> tf.Tensor:
            return self.orient_tensor(
                tensor, transform_info["perm_spatial"], transform_info["flip_axes"]
            )

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

    def _validate_bundle_is_3d(self, bundle: TensorBundle) -> None:
        """Validate that present tensors use Medic-AI 3D sample layout."""
        for key in self.keys:
            if key not in bundle.data:
                continue
            tensor = bundle.data[key]
            spatial_rank = get_spatial_rank(tensor)
            if spatial_rank != 3:
                raise ValueError(
                    "Orientation supports only 3D channel-last tensors shaped "
                    f"(D, H, W, C). Key '{key}' has shape {tensor.shape} with spatial rank "
                    f"{spatial_rank}."
                )

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
            int(
                tf.argmax(
                    tf.cast(tf.equal(current_axes, target_axis), tf.int32), output_type=tf.int32
                )
            )
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
            )
            .numpy()
            .tolist()
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

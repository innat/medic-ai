from __future__ import annotations

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

        Reorient one volume and then map it back with ``inverse()``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Orientation, TensorBundle

            transform = Orientation(keys=["image"], axcodes="RAS")
            image = tf.random.normal((16, 32, 32, 1))
            affine = tf.eye(4)

            bundle = TensorBundle({"image": image}, {"affine": affine})
            forward = transform(bundle)
            restored = transform.inverse(forward)

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
                "original_axcodes": self.get_orientation_from_affine(affine),
                "target_axcodes": self.axcodes,
                "target_tensor_axcodes": target_tensor_axcodes,
                "perm_spatial": transform_info["perm_spatial"],
                "flip_axes": transform_info["flip_axes"],
            },
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = self._get_last_orientation_trace(bundle)
        if trace is None:
            return bundle

        original_affine = trace["params"].get("original_affine")
        if original_affine is None:
            return bundle

        affine = bundle.meta.get("affine")
        if affine is None:
            return bundle

        perm_spatial = trace["params"].get("perm_spatial")
        flip_axes = trace["params"].get("flip_axes")
        if perm_spatial is None or flip_axes is None:
            return bundle
        perm_spatial = tf.cast(tf.convert_to_tensor(perm_spatial), tf.int32)
        flip_axes = tf.cast(tf.convert_to_tensor(flip_axes), tf.int32)

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
                    f"{type(self).__name__} supports only 3D channel-last tensors shaped "
                    f"(D, H, W, C). Key '{key}' has shape {tensor.shape} with spatial rank "
                    f"{spatial_rank}."
                )

    def orient_tensor(
        self,
        tensor: tf.Tensor,
        perm_spatial: tuple[int, int, int] | tf.Tensor,
        flip_axes: tuple[int, ...] | tf.Tensor,
    ) -> tf.Tensor:
        """Reorient one tensor using a spatial permutation followed by flips."""
        perm_spatial = tf.cast(tf.convert_to_tensor(perm_spatial), tf.int32)
        perm = tf.concat([perm_spatial, tf.constant([3], dtype=tf.int32)], axis=0)
        reoriented = tf.transpose(tensor, perm=perm)

        flip_axes = tf.cast(tf.convert_to_tensor(flip_axes), tf.int32)
        return tf.cond(
            tf.shape(flip_axes)[0] > 0,
            lambda: tf.reverse(reoriented, axis=flip_axes),
            lambda: reoriented,
        )

    def inverse_orient_tensor(
        self,
        tensor: tf.Tensor,
        perm_spatial: tuple[int, int, int] | tf.Tensor,
        flip_axes: tuple[int, ...] | tf.Tensor,
    ) -> tf.Tensor:
        """Invert a spatial permutation and flips applied by ``orient_tensor``."""
        perm_spatial = tf.cast(tf.convert_to_tensor(perm_spatial), tf.int32)
        flip_axes = tf.cast(tf.convert_to_tensor(flip_axes), tf.int32)
        restored = tf.cond(
            tf.shape(flip_axes)[0] > 0,
            lambda: tf.reverse(tensor, axis=flip_axes),
            lambda: tensor,
        )
        inverse_perm_spatial = tf.scatter_nd(
            indices=tf.expand_dims(perm_spatial, axis=1),
            updates=tf.range(3, dtype=tf.int32),
            shape=(3,),
        )
        inverse_perm = tf.concat([inverse_perm_spatial, tf.constant([3], dtype=tf.int32)], axis=0)
        return tf.transpose(restored, perm=inverse_perm)

    def get_orientation_from_affine(self, affine: tf.Tensor) -> tf.Tensor:
        """Infer a three-letter orientation code from a 4x4 affine matrix."""
        matrix = tf.cast(affine[:3, :3], tf.float32)
        current_axes = tf.argmax(tf.abs(matrix), axis=0, output_type=tf.int32)
        gather_indices = tf.stack([current_axes, tf.range(3, dtype=tf.int32)], axis=1)
        signs = tf.gather_nd(matrix, gather_indices) >= 0

        def axis_code(axis_index: tf.Tensor, sign: tf.Tensor) -> tf.Tensor:
            return tf.case(
                [
                    (tf.equal(axis_index, 0), lambda: tf.where(sign, "R", "L")),
                    (tf.equal(axis_index, 1), lambda: tf.where(sign, "A", "P")),
                ],
                default=lambda: tf.where(sign, "S", "I"),
                exclusive=True,
            )

        codes = tf.map_fn(
            lambda pair: axis_code(pair[0], tf.cast(pair[1], tf.bool)),
            (current_axes, tf.cast(signs, tf.int32)),
            fn_output_signature=tf.string,
        )
        return tf.strings.reduce_join(codes)

    def reoriented_affine(
        self,
        affine: tf.Tensor,
        input_spatial_shape: tf.Tensor,
        perm_spatial: tuple[int, int, int] | tf.Tensor,
        flip_axes: tuple[int, ...] | tf.Tensor,
    ) -> tf.Tensor:
        """Update affine metadata for a spatial permutation and flips."""
        affine = tf.cast(affine, tf.float32)
        input_spatial_shape = tf.cast(input_spatial_shape, tf.float32)
        perm_spatial = tf.cast(tf.convert_to_tensor(perm_spatial), tf.int32)
        flip_axes = tf.cast(tf.convert_to_tensor(flip_axes), tf.int32)

        flipped_output_mask = tf.scatter_nd(
            indices=tf.expand_dims(flip_axes, axis=1),
            updates=tf.ones_like(flip_axes, dtype=tf.float32),
            shape=(3,),
        )
        signs = 1.0 - 2.0 * flipped_output_mask

        transform = tf.eye(4, dtype=tf.float32)
        spatial_block = tf.transpose(
            tf.one_hot(perm_spatial, depth=3, dtype=tf.float32) * signs[:, None]
        )
        transform = tf.tensor_scatter_nd_update(
            transform,
            indices=[
                [0, 0],
                [0, 1],
                [0, 2],
                [1, 0],
                [1, 1],
                [1, 2],
                [2, 0],
                [2, 1],
                [2, 2],
            ],
            updates=tf.reshape(spatial_block, [-1]),
        )

        flipped_input_mask = tf.scatter_nd(
            indices=tf.expand_dims(perm_spatial, axis=1),
            updates=flipped_output_mask,
            shape=(3,),
        )
        translations = flipped_input_mask * (input_spatial_shape - 1.0)
        transform = tf.tensor_scatter_nd_update(
            transform,
            indices=[[0, 3], [1, 3], [2, 3]],
            updates=translations,
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
    ) -> dict[str, tf.Tensor]:
        matrix = tf.cast(affine[:3, :3], tf.float32)
        current_axes = tf.argmax(tf.abs(matrix), axis=0, output_type=tf.int32)
        gather_indices = tf.stack([current_axes, tf.range(3, dtype=tf.int32)], axis=1)
        current_signs = tf.sign(tf.gather_nd(matrix, gather_indices))
        current_signs = tf.where(current_signs == 0, tf.ones_like(current_signs), current_signs)

        target_axes = [self._AXIS_TO_WORLD[c] for c in target_tensor_axcodes]
        perm_spatial = tf.stack(
            [
                tf.argmax(
                    tf.cast(tf.equal(current_axes, target_axis), tf.int32), output_type=tf.int32
                )
                for target_axis in target_axes
            ]
        )
        current_signs_for_output = tf.gather(
            current_signs,
            perm_spatial,
        )
        target_signs = tf.constant(
            [self._AXIS_TO_SIGN[c] for c in target_tensor_axcodes], dtype=tf.float32
        )
        flip_axes = tf.reshape(
            tf.where(tf.not_equal(current_signs_for_output, target_signs)),
            [-1],
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

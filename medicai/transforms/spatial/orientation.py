from __future__ import annotations

from typing import Any, Sequence

from keras import ops

from ..base import InvertibleTransform, KeyedTransform, _pop_last_transform_trace
from ..tensor_bundle import TensorBundle
from ..utils import (
    compute_orientation_transform,
    get_spatial_rank,
    get_tensor_rank,
    orientation_from_affine,
    reoriented_affine,
    resolve_input_layout,
    validate_affine_matrix,
    validate_tensor_matches_layout,
)


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
    This transform is sample-only and expects one 3D volume at a time.

    Args:
        keys: Keys of tensors in the bundle to reorient. Each selected tensor
            must be a 3D channel-last sample with shape ``(D, H, W, C)``.
        axcodes: Target anatomical orientation code written in standard medical
            imaging convention, such as ``"RAS"``, ``"LPS"``, or ``"LAS"``.
            The string must contain exactly three characters, use only
            ``R/L/A/P/S/I``, and specify one code from each anatomical axis
            family.
        input_layout: Tensor layout contract for selected tensors.
            ``Orientation`` currently supports only ``"DHWC"`` because it is
            affine-aware and sample-level.
        allow_missing_keys: If ``True``, missing keys are skipped. If ``False``,
            missing requested keys raise an error.

    Example:
        Keras selects its backend before the first Keras import. Each example
        below is an independent process.

        TensorFlow backend:

        .. code-block:: python

            import os

            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import Orientation

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

            forward = orient(
                {"image": image, "label": label},
                {"affine": affine},
            )
            restored = orient.inverse(forward)
            print(restored["image"].shape)

        JAX backend:

        .. code-block:: python

            import os

            os.environ["KERAS_BACKEND"] = "jax"

            import jax.numpy as jnp
            from medicai.transforms import Orientation

            transform = Orientation(keys=["image"], axcodes="RAS")
            image = jnp.ones((16, 32, 32, 1), dtype=jnp.float32)
            result = transform({"image": image}, {"affine": jnp.eye(4)})
            print(result["image"].shape)

        Torch backend:

        .. code-block:: python

            import os

            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import Orientation

            transform = Orientation(keys=["image"], axcodes="RAS")
            image = torch.ones((16, 32, 32, 1))
            result = transform({"image": image}, {"affine": torch.eye(4)})
            print(result["image"].shape)

    Returns:
        ``TensorBundle``: The input bundle with selected tensors reoriented in
        place, updated ``affine`` metadata, and an invertible transform trace
        appended to ``bundle.meta["applied_transforms"]``.

    Raises:
        ValueError: If ``axcodes`` is malformed or uses invalid orientation
            codes.
        ValueError: If ``affine`` metadata is missing or is not shaped
            ``(4, 4)``.
        ValueError: If any selected tensor is not 3D spatially.
        KeyError: If none of the requested keys are present and ``allow_missing_keys=False``.
    """

    _AXIS_TO_WORLD = {"R": 0, "L": 0, "A": 1, "P": 1, "S": 2, "I": 2}

    def __init__(
        self,
        keys: Sequence[str] = ("image", "label"),
        axcodes: str = "RAS",
        *,
        input_layout: str = "DHWC",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        axcodes = axcodes.upper()
        self._validate_axcodes(axcodes)
        self.axcodes = axcodes
        self.input_layout = resolve_input_layout(
            input_layout=input_layout,
            allowed_layouts=("DHWC",),
            transform_name=type(self).__name__,
        )

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        affine = bundle.meta.get("affine")
        if affine is None:
            raise ValueError("Affine matrix is required for orientation transformation.")
        affine = validate_affine_matrix(affine)

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
        transform_info = compute_orientation_transform(affine, target_tensor_axcodes)
        perm_spatial = self._as_static_axes(transform_info["perm_spatial"])
        flip_axes = self._as_static_axes(transform_info["flip_axes"])

        def apply_orientation(tensor: Any, _: str) -> Any:
            return self.orient_tensor(tensor, perm_spatial, flip_axes)

        present_keys = self.apply_to_present_keys(bundle, apply_orientation)
        bundle.meta["affine"] = reoriented_affine(
            affine,
            ops.shape(sample_tensor)[:3],
            perm_spatial,
            flip_axes,
        )
        try:
            original_axcodes = orientation_from_affine(affine)
        except (TypeError, ValueError):
            original_axcodes = None
        self.record_transform(
            bundle,
            {
                "keys": list(present_keys),
                "original_affine": ops.cast(affine, "float32"),
                "original_axcodes": original_axcodes,
                "target_axcodes": self.axcodes,
                "target_tensor_axcodes": target_tensor_axcodes,
                "perm_spatial": perm_spatial,
                "flip_axes": flip_axes,
                "input_layout": self.input_layout,
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

        if bundle.meta.get("affine") is None:
            return bundle

        perm_spatial = trace["params"].get("perm_spatial")
        flip_axes = trace["params"].get("flip_axes")
        if perm_spatial is None or flip_axes is None:
            return bundle
        perm_spatial = self._as_static_axes(perm_spatial)
        flip_axes = self._as_static_axes(flip_axes)

        def apply_inverse_orientation(tensor: Any, _: str) -> Any:
            return self.inverse_orient_tensor(tensor, perm_spatial, flip_axes)

        self.apply_to_present_keys(
            bundle,
            apply_inverse_orientation,
            keys=trace["params"].get("keys", []),
        )
        bundle.meta["affine"] = ops.cast(original_affine, "float32")
        return bundle

    def _validate_bundle_is_3d(self, bundle: TensorBundle) -> None:
        """Validate that present tensors use Medic-AI 3D sample layout."""
        for key in self.keys:
            if key not in bundle.data:
                continue
            tensor = bundle.data[key]
            rank = get_tensor_rank(tensor)
            if rank != 4:
                raise ValueError(
                    f"{type(self).__name__} supports only 3D channel-last tensors shaped "
                    f"(D, H, W, C). Key '{key}' has shape {tensor.shape}."
                )
            validate_tensor_matches_layout(
                tensor,
                self.input_layout,
                transform_name=type(self).__name__,
            )
            spatial_rank = get_spatial_rank(tensor)
            if spatial_rank != 3:
                raise ValueError(
                    f"{type(self).__name__} supports only 3D channel-last tensors shaped "
                    f"(D, H, W, C). Key '{key}' has shape {tensor.shape} with spatial rank "
                    f"{spatial_rank}."
                )

    def orient_tensor(
        self,
        tensor: Any,
        perm_spatial: tuple[int, int, int] | Any,
        flip_axes: tuple[int, ...] | Any,
    ) -> Any:
        """Reorient one tensor using a spatial permutation followed by flips."""
        perm_spatial = tuple(perm_spatial)
        perm = (*perm_spatial, 3)
        reoriented = ops.transpose(tensor, axes=perm)
        return self._flip_tensor_axes(reoriented, flip_axes)

    def inverse_orient_tensor(
        self,
        tensor: Any,
        perm_spatial: tuple[int, int, int] | Any,
        flip_axes: tuple[int, ...] | Any,
    ) -> Any:
        """Invert a spatial permutation and flips applied by ``orient_tensor``."""
        perm_spatial = tuple(perm_spatial)
        restored = self._flip_tensor_axes(tensor, flip_axes)
        inverse_perm_spatial = [0, 0, 0]
        for output_axis, input_axis in enumerate(perm_spatial):
            inverse_perm_spatial[input_axis] = output_axis
        inverse_perm = (*inverse_perm_spatial, 3)
        return ops.transpose(restored, axes=inverse_perm)

    @staticmethod
    def _flip_tensor_axes(tensor: Any, flip_axes: tuple[int, ...] | Any) -> Any:
        """Flip one axis at a time for compatibility with older Keras releases."""
        result = tensor
        for axis in tuple(flip_axes):
            # Keras 3.13 wraps the axis internally as ``[axis]``. Passing a
            # tuple here would therefore create a rank-2 TensorFlow ``dims``.
            result = ops.flip(result, axis=int(axis))
        return result

    @staticmethod
    def _as_static_axes(value: Any) -> tuple[int, ...]:
        """Convert a concrete axis tensor to a Python tuple for backend APIs."""
        return tuple(int(axis) for axis in ops.convert_to_numpy(value).reshape(-1))

    def _target_tensor_axcodes(self, axcodes: str) -> str:
        """Map anatomical axis-code order to Medic-AI's depth-first tensor order."""
        return axcodes[::-1]

    def _validate_axcodes(self, axcodes: str) -> None:
        """Validate orientation codes against anatomical axis families."""
        if len(axcodes) != 3:
            raise ValueError("axcodes must be a 3-character string.")

        invalid = [code for code in axcodes if code not in self._AXIS_TO_WORLD]
        if invalid:
            raise ValueError(
                "axcodes must use only anatomical orientation codes from "
                f"R/L/A/P/S/I. Received '{axcodes}'."
            )

        world_axes = {self._AXIS_TO_WORLD[code] for code in axcodes}
        if len(world_axes) != 3:
            raise ValueError(
                "axcodes must contain exactly one code from each anatomical axis family. "
                f"Received '{axcodes}'."
            )

    def _get_last_orientation_trace(self, bundle: TensorBundle) -> dict | None:
        return _pop_last_transform_trace(bundle, type(self).__name__)

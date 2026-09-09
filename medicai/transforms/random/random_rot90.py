from typing import Sequence

import keras
from keras import ops

from ..base import (
    RandomTransform,
    _apply_if_applied,
    _normalize_keys,
    _pop_last_transform_trace,
    _trace_applied_to_bool,
)
from ..spatial.rotate90 import Rotate90
from ..tensor_bundle import TensorBundle


class RandomRotate90(RandomTransform):
    """Randomly rotate selected tensors by quarter turns.

    ``RandomRotate90`` samples whether to apply a rotation and, when applied,
    samples an integer ``k`` in ``[1, max_k]`` before delegating to
    :class:`~medicai.transforms.Rotate90`.

    .. note::

        In batch mode, the apply decision and quarter-turn count ``k`` are
        sampled independently for each item.

        For graph execution, the two selected rotation axes must have equal
        lengths. A 90- or 270-degree rotation swaps those axes, and graph control
        flow requires the applied and skipped branches to return the same shape.
        For example, axial rotation of ``(D, H, W, C)`` requires ``H == W``;
        ``D == H == W`` is not required unless all three planes are used.

    Args:
        keys: Keys of the tensors to rotate.
        prob: Probability of applying the rotation.
        max_k: Maximum number of quarter turns sampled per call.
        spatial_axis: Two axes defining the rotation plane. If ``None``, the
            last two spatial dimensions are used. For 2D tensors, the only
            meaningful plane is ``(0, 1)``, which corresponds to the
            vertical-height and horizontal-width image axes. For 3D tensors,
            the valid spatial planes are ``(1, 2)`` for the axial plane,
            ``(0, 2)`` for the coronal plane, and ``(0, 1)`` for the
            sagittal plane when using sample-space axis numbering ``(D, H, W)``.
        input_layout: Channel-last tensor layout. Supported values are
            ``"HWC"``, ``"DHWC"``, ``"BHWC"``, and ``"BDHWC"``.
        seed: Optional random seed. Supports ``None``, an integer seed, or a
            ``keras.random.SeedGenerator``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:

        TensorFlow backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import RandomRotate90

            transform = RandomRotate90(
                keys=["image"],
                prob=0.5,
                max_k=3,
                input_layout="HWC",
            )
            image = tf.random.normal((64, 64, 1))
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        JAX backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "jax"

            import jax
            from medicai.transforms import RandomRotate90

            transform = RandomRotate90(
                keys=["image"],
                prob=0.5,
                max_k=3,
                input_layout="DHWC",
            )
            image = jax.random.normal(
                jax.random.PRNGKey(7), shape=(32, 64, 64, 1)
            )
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Torch backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "torch"

            import torch
            from medicai.transforms import RandomRotate90

            transform = RandomRotate90(
                keys=["image"], prob=0.5, max_k=3, input_layout="BHWC"
            )
            torch.manual_seed(7)
            batch = torch.randn((2, 64, 64, 1))
            result = transform({"image": batch})
            output = result["image"]
            print(output.shape)
    """

    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.1,
        max_k: int = 3,
        spatial_axis: Sequence[int] | None = None,
        *,
        input_layout: str,
        seed: int | keras.random.SeedGenerator | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob, seed=seed)
        if max_k < 1:
            raise ValueError(f"`max_k` must be >= 1. Received {max_k}.")
        self.keys = _normalize_keys(keys)
        self.max_k = max_k
        self.spatial_axis = spatial_axis
        self.allow_missing_keys = allow_missing_keys
        self.rotate = Rotate90(
            keys=self.keys,
            k=1,
            spatial_axis=self.spatial_axis,
            input_layout=input_layout,
            allow_missing_keys=self.allow_missing_keys,
        )
        self.input_layout = self.rotate.input_layout

    @property
    def invertible(self) -> bool:
        return True

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        params = self.get_random_params(bundle)
        return self.apply_with_params(bundle, params)

    def get_random_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Sample independent decisions and quarter-turn counts per batch item."""
        present_key = next((key for key in self.keys if key in bundle.data), None)
        if present_key is None:
            batch_size = 1
            parameter_shape = ()
        elif self.rotate.layout_info.batched:
            batch_size = ops.shape(bundle.data[present_key])[0]
            parameter_shape = (batch_size,)
        else:
            batch_size = 1
            parameter_shape = ()
        return {
            "should_apply": self.random_uniform(
                shape=parameter_shape,
                minval=0.0,
                maxval=1.0,
                dtype="float32",
            )
            < self.prob,
            "k": self.random_integers(
                shape=parameter_shape,
                minval=1,
                maxval=self.max_k + 1,
                dtype="int32",
            ),
            "spatial_axis": self.spatial_axis,
            "input_layout": self.input_layout,
        }

    def apply_with_params(
        self,
        bundle: TensorBundle,
        params: dict[str, object],
    ) -> TensorBundle:
        """Apply the sampled quarter-turn rotation to all selected keys."""
        present_keys = self.rotate.apply_to_present_keys(
            bundle,
            lambda tensor, key: self.transform_tensor(tensor, key, params),
        )
        self.record_random_transform(
            bundle,
            params=self.build_trace_params(params, present_keys),
            applied=ops.any(ops.cast(params["should_apply"], "bool")),
            kernel="Rotate90",
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = self._get_last_random_rotate90_trace(bundle)
        if trace is None:
            return bundle

        applied = trace["params"].get("should_apply", trace.get("applied", False))
        k = trace["params"].get("k")

        def apply_inverse_rotate(tensor, _: str):
            inverse_k = ops.mod(-ops.cast(k, "int32"), 4)
            return _apply_if_applied(
                applied,
                lambda tensor=tensor: self._rotate_with_dynamic_k(
                    tensor, inverse_k, trace["params"].get("spatial_axis")
                ),
                lambda tensor=tensor: tensor,
            )

        self.rotate.apply_to_present_keys(
            bundle, apply_inverse_rotate, keys=trace["params"].get("keys", [])
        )
        return bundle

    def transform_tensor(
        self,
        tensor,
        key: str,
        params: dict[str, object],
    ):
        """Apply the sampled rotation conditionally to one tensor."""
        del key
        try:
            applied = _trace_applied_to_bool(params["should_apply"])
        except ValueError:
            applied = None
        if applied is False:
            return tensor
        if applied is True or self.rotate.layout_info.batched:
            self._validate_square_rotation_plane(tensor, params["spatial_axis"])
        if self.rotate.layout_info.batched:
            return self._rotate_batch_with_broadcast(
                tensor,
                params["k"],
                params["spatial_axis"],
                params["should_apply"],
            )

        return _apply_if_applied(
            params["should_apply"],
            lambda tensor=tensor: self._rotate_with_dynamic_k(
                tensor,
                params["k"],
                params["spatial_axis"],
            ),
            lambda tensor=tensor: tensor,
        )

    def _rotate_batch_with_broadcast(
        self,
        tensor,
        k,
        spatial_axis: Sequence[int] | None,
        should_apply,
    ):
        """Select one of four fixed-shape rotations independently per item."""
        effective_k = ops.mod(ops.cast(k, "int32"), 4)
        effective_k = ops.where(
            ops.cast(should_apply, "bool"),
            effective_k,
            ops.zeros_like(effective_k),
        )
        branches = [
            tensor,
            self.rotate.rotate_tensor(tensor, k=1, spatial_axis=spatial_axis),
            self.rotate.rotate_tensor(tensor, k=2, spatial_axis=spatial_axis),
            self.rotate.rotate_tensor(tensor, k=3, spatial_axis=spatial_axis),
        ]
        mask_shape = [ops.shape(tensor)[0]] + [1] * (len(tensor.shape) - 1)
        result = branches[0]
        for branch_index in range(1, 4):
            branch_mask = ops.reshape(
                ops.equal(effective_k, branch_index),
                mask_shape,
            )
            result = ops.where(branch_mask, branches[branch_index], result)
        return result

    def _rotate_with_dynamic_k(
        self,
        tensor,
        k,
        spatial_axis: Sequence[int] | None,
    ):
        """Dispatch dynamic quarter turns through shape-compatible branches."""
        effective_k = ops.mod(ops.cast(k, "int32"), 4)

        def rotate(k_value: int):
            return self.rotate.rotate_tensor(
                tensor,
                k=k_value,
                spatial_axis=spatial_axis,
            )

        # Keep k as a Python integer inside each leaf. This avoids passing a
        # symbolic value to the rotation kernel while remaining graph-safe.
        return ops.cond(
            ops.greater(effective_k, 1),
            lambda: ops.cond(
                ops.equal(effective_k, 3),
                lambda: rotate(3),
                lambda: rotate(2),
            ),
            lambda: ops.cond(
                ops.equal(effective_k, 1),
                lambda: rotate(1),
                lambda: rotate(0),
            ),
        )

    def _validate_square_rotation_plane(
        self,
        tensor,
        spatial_axis: Sequence[int] | None,
    ) -> None:
        """Require equal static lengths for the selected graph rotation plane."""
        axes = self.rotate._resolve_axes(tensor, spatial_axis=spatial_axis)
        shape = tensor.shape
        first_size, second_size = shape[axes[0]], shape[axes[1]]
        if first_size is not None and second_size is not None and first_size != second_size:
            raise ValueError(
                "RandomRotate90 requires equal sizes for the selected rotation "
                f"axes in graph mode, but got {first_size} and {second_size} "
                f"for axes {axes}. Use square patches or deterministic Rotate90 "
                "for rectangular inputs."
            )

    def build_trace_params(
        self,
        params: dict[str, object],
        present_keys: Sequence[str],
    ) -> dict[str, object]:
        """Build random trace metadata for the current quarter-turn rotation."""
        return {
            "keys": list(present_keys),
            "k": params["k"],
            "should_apply": params["should_apply"],
            "spatial_axis": params["spatial_axis"],
            "input_layout": params["input_layout"],
        }

    def _get_last_random_rotate90_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)

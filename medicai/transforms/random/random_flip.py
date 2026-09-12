from typing import Sequence

import keras
from keras import ops

from ..base import RandomTransform, _apply_if_applied, _pop_last_transform_trace
from ..spatial.flip import Flip
from ..tensor_bundle import TensorBundle

_DEFAULT_PROB = 0.1
_DEFAULT_SPATIAL_AXIS = None


class RandomFlip(RandomTransform):
    """Randomly flip selected tensors along specified spatial axes.

    ``RandomFlip`` wraps the deterministic :class:`~medicai.transforms.Flip`
    kernel with probabilistic application. When the sampled Bernoulli draw is
    ``True``, each selected channel-last tensor is reversed along the
    configured axes.

    Args:
        keys: Keys of the tensors to flip.
        prob: Probability of applying the flip.
        spatial_axis: Spatial axis or axes to reverse. For 2D tensors, ``0``
            refers to the vertical-height axis and ``1`` refers to the
            horizontal-width axis. For 3D tensors using sample-space axis
            numbering ``(D, H, W)``, axis ``0`` is the depth direction, axis
            ``1`` is the height direction, and axis ``2`` is the width
            direction; these correspond to the sagittal, coronal, and axial
            viewing orientations depending on which axis is being mirrored.
        input_layout: Channel-last tensor layout. Supported values are
            ``"HWC"``, ``"DHWC"``, ``"BHWC"``, and ``"BDHWC"``.
        seed: Optional random seed. Supports ``None``, an integer seed, or a
            ``keras.random.SeedGenerator``. The seed controls the Bernoulli
            apply/skip draw for each call. A fresh transform instance created
            with the same integer seed replays the same random sequence, while
            repeated calls on one instance advance its internal seed stream.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:

        TensorFlow backend:

        .. code-block:: python

            import os
            os.environ["KERAS_BACKEND"] = "tensorflow"

            import tensorflow as tf
            from medicai.transforms import RandomFlip

            transform = RandomFlip(
                keys=["image"],
                prob=0.5,
                spatial_axis=0,
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
            from medicai.transforms import RandomFlip

            transform = RandomFlip(
                keys=["image"],
                prob=0.5,
                spatial_axis=0,
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
            from medicai.transforms import RandomFlip

            transform = RandomFlip(
                keys=["image"], prob=0.5, spatial_axis=1, input_layout="BHWC"
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
        prob: float = _DEFAULT_PROB,
        spatial_axis: int | Sequence[int] | None = _DEFAULT_SPATIAL_AXIS,
        *,
        input_layout: str,
        seed: int | keras.random.SeedGenerator | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob, seed=seed)

        self.flip = Flip(
            keys=keys,
            spatial_axis=spatial_axis,
            input_layout=input_layout,
            allow_missing_keys=allow_missing_keys,
        )

    @property
    def invertible(self) -> bool:
        return self.flip.spatial_axis is not None

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        params = self.get_random_params(bundle)
        if not params["enabled"]:
            self.record_skipped(bundle, params)
            return bundle

        return self.apply_with_params(bundle, params)

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        if self.flip.spatial_axis is None:
            return bundle

        trace = self._get_last_random_flip_trace(bundle)
        if trace is None:
            return bundle

        applied = trace["params"].get("should_apply", trace.get("applied", False))

        def apply_inverse_flip(tensor, _: str):
            flipped = self.flip.flip_tensor(tensor)
            if self.flip.layout_info.batched:
                mask_shape = [ops.shape(tensor)[0]] + [1] * (len(tensor.shape) - 1)
                mask = ops.reshape(ops.cast(applied, "bool"), mask_shape)
                return ops.where(mask, flipped, tensor)
            return _apply_if_applied(
                applied,
                lambda: flipped,
                lambda tensor=tensor: tensor,
            )

        self.flip.apply_to_present_keys(
            bundle,
            apply_inverse_flip,
            keys=trace["params"].get("keys", []),
        )
        return bundle

    def get_random_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Sample an independent Bernoulli decision for each batch item."""
        present_key = next(
            (key for key in self.flip.keys if key in bundle.data),
            None,
        )
        if present_key is not None and self.flip.layout_info.batched:
            batch_size = ops.shape(bundle.data[present_key])[0]
            shape = (batch_size,)
        else:
            shape = ()

        return {
            "enabled": self.flip.spatial_axis is not None,
            "should_apply": self.random_uniform(
                shape=shape,
                minval=0.0,
                maxval=1.0,
                dtype="float32",
            )
            < self.prob,
            "spatial_axis": self.flip.spatial_axis,
            "input_layout": self.flip.input_layout,
        }

    def apply_with_params(
        self,
        bundle: TensorBundle,
        params: dict[str, object],
    ) -> TensorBundle:
        """Apply the flip kernel using each item's sampled decision."""
        present_keys = self.flip.apply_to_present_keys(
            bundle,
            lambda tensor, key: self.transform_tensor(tensor, key, params),
        )
        self.record_random_transform(
            bundle,
            params=self.build_trace_params(
                params,
                present_keys,
            ),
            applied=ops.any(ops.cast(params["should_apply"], "bool")),
            kernel="Flip",
        )
        return bundle

    def record_skipped(
        self,
        bundle: TensorBundle,
        params: dict[str, object],
    ) -> TensorBundle:
        """Record a skipped random flip when no axes are configured."""
        self.record_random_transform(
            bundle,
            params=self.build_trace_params(params, []),
            applied=False,
            kernel="Flip",
        )
        return bundle

    def transform_tensor(
        self,
        tensor,
        key: str,
        params: dict[str, object],
    ):
        """Apply the sampled flip decision to one tensor."""
        del key
        flipped = self.flip.flip_tensor(tensor, spatial_axis=params["spatial_axis"])
        if self.flip.layout_info.batched:
            mask_shape = [ops.shape(tensor)[0]] + [1] * (len(tensor.shape) - 1)
            mask = ops.reshape(ops.cast(params["should_apply"], "bool"), mask_shape)
            return ops.where(mask, flipped, tensor)
        return _apply_if_applied(params["should_apply"], lambda: flipped, lambda: tensor)

    def build_trace_params(
        self,
        params: dict[str, object],
        present_keys: Sequence[str],
    ) -> dict[str, object]:
        """Build random trace metadata for the current flip."""
        return {
            "keys": list(present_keys),
            "should_apply": params["should_apply"],
            "spatial_axis": params["spatial_axis"],
            "input_layout": params["input_layout"],
        }

    def _get_last_random_flip_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)

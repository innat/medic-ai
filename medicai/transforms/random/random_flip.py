from typing import Sequence, Union

import keras
import tensorflow as tf

from ..base import RandomTransform, _pop_last_transform_trace, _trace_applied_to_bool
from ..spatial.flip import Flip
from ..tensor_bundle import TensorBundle


class RandomFlip(RandomTransform):
    """Randomly flip selected tensors along specified spatial axes.

    ``RandomFlip`` wraps the deterministic :class:`~medicai.transforms.Flip`
    kernel with probabilistic application. When the sampled Bernoulli draw is
    ``True``, each selected channel-last tensor is reversed along the
    configured axes.

    Depending on ``input_mode``, this transform supports:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

    Args:
        keys: Keys of the tensors to flip.
        prob: Probability of applying the flip.
        spatial_axis: Spatial axis or axes to reverse. If ``None``, the
            transform behaves as a no-op.
        input_mode: Either ``"sample"`` for ``(H, W, C)`` / ``(D, H, W, C)``
            tensors, or ``"batch"`` for ``(B, H, W, C)`` / ``(B, D, H, W, C)``
            tensors.
        seed: Optional random seed. Supports ``None``, an integer seed, or a
            ``keras.random.SeedGenerator``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Randomly flip a 2D image using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomFlip

            transform = RandomFlip(keys=["image"], prob=0.5, spatial_axis=0)
            image = tf.random.normal((64, 64, 1))
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Randomly flip a 3D image stored in a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomFlip, TensorBundle

            transform = RandomFlip(keys=["image"], prob=0.5, spatial_axis=0)
            image = tf.random.normal((32, 64, 64, 1))
            bundle = TensorBundle({"image": image})
            result = transform(bundle)
            output = result["image"]
            print(output.shape)
    """

    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.1,
        spatial_axis: Union[int, Sequence[int], None] = None,
        input_mode: str = "sample",
        seed: int | keras.random.SeedGenerator | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob, seed=seed)
        self.flip = Flip(
            keys=keys,
            spatial_axis=spatial_axis,
            input_mode=input_mode,
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

        applied = trace.get("applied", False)

        def apply_inverse_flip(tensor: tf.Tensor, _: str) -> tf.Tensor:
            if tf.is_tensor(applied):
                return tf.cond(
                    tf.cast(applied, tf.bool),
                    lambda tensor=tensor: self.flip.flip_tensor(tensor),
                    lambda tensor=tensor: tensor,
                )
            if _trace_applied_to_bool(applied):
                return self.flip.flip_tensor(tensor)
            return tensor

        self.flip.apply_to_present_keys(
            bundle, apply_inverse_flip, keys=trace["params"].get("keys", [])
        )
        return bundle

    def get_random_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Sample one Bernoulli decision shared across all selected keys."""
        del bundle
        return {
            "enabled": self.flip.spatial_axis is not None,
            "should_apply": self.sample_should_apply(),
            "spatial_axis": self.flip.spatial_axis,
            "input_mode": self.flip.input_mode,
        }

    def apply_with_params(
        self,
        bundle: TensorBundle,
        params: dict[str, object],
    ) -> TensorBundle:
        """Apply the shared flip kernel conditionally using sampled params."""
        present_keys = self.flip.apply_to_present_keys(
            bundle,
            lambda tensor, key: self.transform_tensor(tensor, key, params),
        )
        self.record_random_transform(
            bundle,
            params=self.build_trace_params(params, present_keys),
            applied=params["should_apply"],
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
        tensor: tf.Tensor,
        key: str,
        params: dict[str, object],
    ) -> tf.Tensor:
        """Apply the sampled flip decision to one tensor."""
        del key
        return tf.cond(
            tf.cast(params["should_apply"], tf.bool),
            lambda tensor=tensor: self.flip.flip_tensor(
                tensor,
                spatial_axis=params["spatial_axis"],
            ),
            lambda tensor=tensor: tensor,
        )

    def build_trace_params(
        self,
        params: dict[str, object],
        present_keys: Sequence[str],
    ) -> dict[str, object]:
        """Build random trace metadata for the current flip."""
        return {
            "keys": list(present_keys),
            "spatial_axis": params["spatial_axis"],
            "input_mode": params["input_mode"],
        }

    def _get_last_random_flip_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)

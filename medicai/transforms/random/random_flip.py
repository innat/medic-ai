from typing import Sequence, Union

import tensorflow as tf

from ..base import RandomTransform, _pop_last_transform_trace, _trace_applied_to_bool
from ..spatial.flip import Flip
from ..tensor_bundle import TensorBundle


class RandomFlip(RandomTransform):
    """Randomly flip selected tensors along specified spatial axes.

    ``RandomFlip`` wraps the deterministic :class:`~medicai.transforms.Flip`
    kernel with probabilistic application. When the sampled Bernoulli draw is
    ``True``, each selected channel-last sample tensor is reversed along the
    configured axes.

    This transform supports:

    - 2D tensors shaped ``(H, W, C)``
    - 3D tensors shaped ``(D, H, W, C)``

    Args:
        keys: Keys of the tensors to flip.
        prob: Probability of applying the flip.
        spatial_axis: Spatial axis or axes to reverse. If ``None``, the
            transform behaves as a no-op.
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
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob)
        self.flip = Flip(
            keys=keys,
            spatial_axis=spatial_axis,
            allow_missing_keys=allow_missing_keys,
        )

    @property
    def invertible(self) -> bool:
        return self.flip.spatial_axis is not None

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        should_flip = self.sample_should_apply()

        if self.flip.spatial_axis is None:
            self.record_random_transform(
                bundle,
                params={"keys": [], "spatial_axis": None},
                applied=False,
                kernel="Flip",
            )
            return bundle

        present_keys = self.flip.apply_to_present_keys(
            bundle,
            lambda tensor, _: tf.cond(
                should_flip,
                lambda tensor=tensor: self.flip.flip_tensor(tensor),
                lambda tensor=tensor: tensor,
            ),
        )
        self.record_random_transform(
            bundle,
            params={
                "keys": list(present_keys),
                "spatial_axis": self.flip.spatial_axis,
            },
            applied=should_flip,
            kernel="Flip",
        )
        return bundle

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

        self.flip.apply_to_present_keys(bundle, apply_inverse_flip, keys=trace["params"].get("keys", []))
        return bundle

    def _get_last_random_flip_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)

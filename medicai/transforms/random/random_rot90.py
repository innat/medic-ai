from typing import Sequence

import keras
import tensorflow as tf

from ..base import (
    RandomTransform,
    _normalize_keys,
    _pop_last_transform_trace,
    _trace_applied_to_bool,
)
from ..spatial.rotate90 import Rotate90
from ..tensor_bundle import TensorBundle
from ..utils import validate_input_mode


class RandomRotate90(RandomTransform):
    """Randomly rotate selected tensors by quarter turns.

    ``RandomRotate90`` samples whether to apply a rotation and, when applied,
    samples an integer ``k`` in ``[1, max_k]`` before delegating to
    :class:`~medicai.transforms.Rotate90`.

    Depending on ``input_mode``, this transform supports:

    - sample 2D tensors shaped ``(H, W, C)``
    - sample 3D tensors shaped ``(D, H, W, C)``
    - batch 2D tensors shaped ``(B, H, W, C)``
    - batch 3D tensors shaped ``(B, D, H, W, C)``

    In batch mode, one quarter-turn count ``k`` is sampled per transform call
    and that same rotation is applied across the whole batch.

    Args:
        keys: Keys of the tensors to rotate.
        prob: Probability of applying the rotation.
        max_k: Maximum number of quarter turns sampled per call.
        spatial_axis: Two axes defining the rotation plane. If ``None``, the
            last two spatial dimensions are used.
        input_mode: Either ``"sample"`` for ``(H, W, C)`` / ``(D, H, W, C)``
            tensors, or ``"batch"`` for ``(B, H, W, C)`` / ``(B, D, H, W, C)``
            tensors.
        seed: Optional random seed. Supports ``None``, an integer seed, or a
            ``keras.random.SeedGenerator``.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Randomly rotate a 2D image using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomRotate90

            transform = RandomRotate90(keys=["image"], prob=0.5, max_k=3)
            image = tf.random.normal((64, 64, 1))
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Randomly rotate a 3D image stored in a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import RandomRotate90, TensorBundle

            transform = RandomRotate90(keys=["image"], prob=0.5, max_k=3)
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
        max_k: int = 3,
        spatial_axis: Sequence[int] | None = None,
        input_mode: str = "sample",
        seed: int | keras.random.SeedGenerator | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob, seed=seed)
        if max_k < 1:
            raise ValueError(f"`max_k` must be >= 1. Received {max_k}.")
        self.keys = _normalize_keys(keys)
        self.max_k = max_k
        self.spatial_axis = spatial_axis
        self.input_mode = validate_input_mode(input_mode, transform_name=type(self).__name__)
        self.allow_missing_keys = allow_missing_keys

    @property
    def invertible(self) -> bool:
        return True

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        params = self.get_random_params(bundle)
        return self.apply_with_params(bundle, params)

    def get_random_params(self, bundle: TensorBundle) -> dict[str, object]:
        """Sample the shared Bernoulli decision and quarter-turn count."""
        del bundle
        return {
            "should_apply": self.sample_should_apply(),
            "k": self.random_integers(shape=(), minval=1, maxval=self.max_k + 1, dtype=tf.int32),
            "spatial_axis": self.spatial_axis,
            "input_mode": self.input_mode,
        }

    def apply_with_params(
        self,
        bundle: TensorBundle,
        params: dict[str, object],
    ) -> TensorBundle:
        """Apply the sampled quarter-turn rotation to all selected keys."""
        rotate = self._build_rotate_kernel()
        present_keys = rotate.apply_to_present_keys(
            bundle,
            lambda tensor, key: self.transform_tensor(tensor, key, params, rotate),
        )
        self.record_random_transform(
            bundle,
            params=self.build_trace_params(params, present_keys),
            applied=params["should_apply"],
            kernel="Rotate90",
        )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        trace = self._get_last_random_rotate90_trace(bundle)
        if trace is None:
            return bundle

        applied = trace.get("applied", False)
        k = trace["params"].get("k")
        rotate = self._build_rotate_kernel()

        def apply_inverse_rotate(tensor: tf.Tensor, _: str) -> tf.Tensor:
            inverse_k = tf.math.floormod(-tf.cast(k, tf.int32), 4)
            if tf.is_tensor(applied):
                return tf.cond(
                    tf.cast(applied, tf.bool),
                    lambda tensor=tensor: rotate.rotate_tensor(tensor, k=inverse_k),
                    lambda tensor=tensor: tensor,
                )
            if _trace_applied_to_bool(applied):
                return rotate.rotate_tensor(tensor, k=inverse_k)
            return tensor

        rotate.apply_to_present_keys(
            bundle, apply_inverse_rotate, keys=trace["params"].get("keys", [])
        )
        return bundle

    def transform_tensor(
        self,
        tensor: tf.Tensor,
        key: str,
        params: dict[str, object],
        rotate: Rotate90,
    ) -> tf.Tensor:
        """Apply the sampled rotation conditionally to one tensor."""
        del key
        return tf.cond(
            tf.cast(params["should_apply"], tf.bool),
            lambda tensor=tensor: rotate.rotate_tensor(
                tensor,
                k=params["k"],
                spatial_axis=params["spatial_axis"],
            ),
            lambda tensor=tensor: tensor,
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
            "spatial_axis": params["spatial_axis"],
            "input_mode": params["input_mode"],
        }

    def _build_rotate_kernel(self) -> Rotate90:
        """Construct the deterministic rotation kernel reused by this wrapper."""
        return Rotate90(
            keys=self.keys,
            k=1,
            spatial_axis=self.spatial_axis,
            input_mode=self.input_mode,
            allow_missing_keys=self.allow_missing_keys,
        )

    def _get_last_random_rotate90_trace(self, bundle: TensorBundle):
        return _pop_last_transform_trace(bundle, type(self).__name__)

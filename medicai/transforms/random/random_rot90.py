from typing import Sequence

import tensorflow as tf

from ..base import RandomTransform
from ..spatial.rotate90 import Rotate90
from ..tensor_bundle import TensorBundle


class RandomRotate90(RandomTransform):
    """Randomly rotate selected tensors by quarter turns.

    ``RandomRotate90`` samples whether to apply a rotation and, when applied,
    samples an integer ``k`` in ``[1, max_k]`` before delegating to
    :class:`~medicai.transforms.Rotate90`.

    This transform supports:

    - 2D tensors shaped ``(H, W, C)``
    - 3D tensors shaped ``(D, H, W, C)``

    Args:
        keys: Keys of the tensors to rotate.
        prob: Probability of applying the rotation.
        max_k: Maximum number of quarter turns sampled per call.
        spatial_axis: Two axes defining the rotation plane. If ``None``, the
            last two spatial dimensions are used.
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
        allow_missing_keys: bool = False,
    ):
        super().__init__(prob=prob)
        if max_k < 1:
            raise ValueError(f"`max_k` must be >= 1. Received {max_k}.")
        self.keys = tuple(keys)
        self.max_k = max_k
        self.spatial_axis = spatial_axis
        self.allow_missing_keys = allow_missing_keys

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        should_rotate = self.sample_should_apply()
        k = tf.random.uniform([], minval=1, maxval=self.max_k + 1, dtype=tf.int32)
        rotate = Rotate90(
            keys=self.keys,
            k=1,
            spatial_axis=self.spatial_axis,
            allow_missing_keys=self.allow_missing_keys,
        )

        present_keys = rotate.apply_to_present_keys(
            bundle,
            lambda tensor, _: tf.cond(
                should_rotate,
                lambda tensor=tensor: rotate.rotate_tensor(tensor, k=k),
                lambda tensor=tensor: tensor,
            ),
        )
        self.record_random_transform(
            bundle,
            params={
                "keys": list(present_keys),
                "k": k,
                "spatial_axis": self.spatial_axis,
            },
            applied=should_rotate,
            kernel="Rotate90",
        )
        return bundle

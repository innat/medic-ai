from typing import Sequence, Union

import tensorflow as tf

from ..base import InvertibleTransform, KeyedTransform
from ..tensor_bundle import TensorBundle
from ..utils import get_spatial_rank, normalize_spatial_axes


class Flip(KeyedTransform, InvertibleTransform):
    """Flip selected tensors along one or more spatial axes.

    ``Flip`` deterministically reverses channel-last sample tensors using
    TensorFlow's ``tf.reverse``. It can be applied to common Medic-AI
    dictionary-style samples such as image-label pairs and supports both:

    - 2D tensors shaped ``(H, W, C)``
    - 3D tensors shaped ``(D, H, W, C)``

    The transform is invertible because applying the same flip twice restores
    the original orientation. During :meth:`apply`, the normalized axes are
    recorded in the ``TensorBundle`` transform trace so downstream pipelines
    can inspect what was applied.

    Args:
        keys: Keys of the tensors to flip.
        spatial_axis: Spatial axis or axes to reverse. Axes follow the tensor's
            sample layout, so ``0`` refers to ``H`` for 2D tensors and ``D``
            for 3D tensors. If ``None``, the transform is a no-op.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Flip a 3D image-label pair along the depth axis:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Flip

            transform = Flip(keys=["image", "label"], spatial_axis=0)

            image = tf.random.normal((32, 64, 64, 1))
            label = tf.random.uniform(
                (32, 64, 64, 1), maxval=2, dtype=tf.int32
            )

            result = transform({"image": image, "label": label})

        Restore a flipped tensor with ``inverse()``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Flip, TensorBundle

            transform = Flip(keys=["image"], spatial_axis=1)
            image = tf.random.normal((64, 64, 1))

            forward = transform({"image": image})
            restored = transform.inverse(forward)

            print(forward["image"].shape)
            print(restored["image"].shape)

        Flip a 2D image stored in a ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import Flip, TensorBundle

            transform = Flip(keys=["image"], spatial_axis=0)
            image = tf.random.normal((64, 64, 1))

            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

    Returns:
        ``TensorBundle``: The input bundle with flipped tensors and an
        invertible transform trace entry when a flip is applied.

    Raises:
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
        ValueError: If ``spatial_axis`` contains invalid axes for a selected
            tensor.
    """

    def __init__(
        self,
        keys: Sequence[str],
        spatial_axis: Union[int, Sequence[int], None] = None,
        allow_missing_keys: bool = False,
    ):
        KeyedTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.spatial_axis = spatial_axis

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        if self.spatial_axis is None:
            return bundle

        normalized_axes = {"value": None}

        def apply_flip(tensor: tf.Tensor, _: str) -> tf.Tensor:
            normalized_axes["value"] = self._resolve_axes(tensor)
            return self.flip_tensor(tensor)

        present_keys = self.apply_to_present_keys(bundle, apply_flip)

        if normalized_axes["value"] is not None:
            self.record_transform(
                bundle,
                {
                    "keys": list(present_keys),
                    "spatial_axis": normalized_axes["value"],
                },
            )
        return bundle

    def inverse(self, bundle: TensorBundle) -> TensorBundle:
        if self.spatial_axis is None:
            return bundle

        self.apply_to_present_keys(bundle, lambda tensor, _: self.flip_tensor(tensor))
        return bundle

    def flip_tensor(self, tensor: tf.Tensor) -> tf.Tensor:
        """Flip one tensor using the configured spatial axes.

        Args:
            tensor: Channel-last 2D or 3D sample tensor to reverse.

        Returns:
            ``tf.Tensor``: The flipped tensor. If ``spatial_axis`` is
            ``None``, the input tensor is returned unchanged.
        """
        if self.spatial_axis is None:
            return tensor
        return tf.reverse(tensor, axis=self._resolve_axes(tensor))

    def _resolve_axes(self, tensor: tf.Tensor) -> tuple[int, ...]:
        axes = self.spatial_axis
        if isinstance(axes, int):
            axes = (axes,)
        if axes is None:
            return ()
        spatial_rank = get_spatial_rank(tensor)
        return normalize_spatial_axes(tuple(axes), spatial_rank, name="spatial_axis")

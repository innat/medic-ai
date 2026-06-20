from typing import Optional, Sequence

import tensorflow as tf

from ..base import KeyedTransform
from ..tensor_bundle import TensorBundle


class ScaleIntensityRange(KeyedTransform):
    """Linearly map selected tensor intensities from one numeric range to another.

    ``ScaleIntensityRange`` applies an affine intensity transform using the
    source interval ``[a_min, a_max]`` and, when provided, the target interval
    ``[b_min, b_max]``. This is useful for bringing image intensities into a
    stable range such as ``[0, 1]`` or ``[-1, 1]`` before training.

    The transform expects channel-last image tensors such as ``(H, W, C)`` or
    ``(D, H, W, C)``. It does not infer source ranges from the data; callers
    must provide medically meaningful source bounds.

    Args:
        keys: Keys of the tensors to scale.
        a_min: Lower bound of the source intensity range.
        a_max: Upper bound of the source intensity range.
        b_min: Lower bound of the target range. If ``None`` together with
            ``b_max=None``, the normalized ``[0, 1]`` result is kept.
        b_max: Upper bound of the target range.
        clip: If ``True`` and both ``b_min`` and ``b_max`` are provided, clip
            the output to the target interval after scaling.
        dtype: Output dtype used for computation and returned tensors.
        allow_missing_keys: If ``True``, missing keys are skipped.

    Example:
        Map a 2D image into ``[-1, 1]`` using a raw Python dictionary:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import ScaleIntensityRange

            transform = ScaleIntensityRange(
                keys=["image"],
                a_min=0.0,
                a_max=255.0,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
            )

            image = tf.random.uniform((64, 64, 1), minval=0.0, maxval=255.0)
            result = transform({"image": image})
            output = result["image"]
            print(output.shape)

        Map a 3D image volume from a clipped CT range into ``[0, 1]`` using a
        ``TensorBundle``:

        .. code-block:: python

            import tensorflow as tf
            from medicai.transforms import ScaleIntensityRange, TensorBundle

            transform = ScaleIntensityRange(
                keys=["image"],
                a_min=-175.0,
                a_max=250.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )

            image = tf.random.normal((32, 64, 64, 1))
            bundle = TensorBundle({"image": image})
            result = transform(bundle)
            output = result["image"]
            print(output.shape)

    Returns:
        ``TensorBundle``: The input bundle with selected tensors scaled in
        place and a non-invertible trace entry appended.

    Raises:
        KeyError: If a requested key is missing and
            ``allow_missing_keys=False``.
    """

    def __init__(
        self,
        keys: Sequence[str],
        a_min: float,
        a_max: float,
        b_min: Optional[float] = None,
        b_max: Optional[float] = None,
        clip: bool = False,
        dtype: tf.DType = tf.float32,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.dtype = dtype

    def apply(self, bundle: TensorBundle) -> TensorBundle:
        present_keys = self.apply_to_present_keys(bundle, lambda tensor, _: self.scale_tensor(tensor))
        bundle.push_transform(
            self.build_trace_entry(
                params={
                    "keys": list(present_keys),
                    "a_min": self.a_min,
                    "a_max": self.a_max,
                    "b_min": self.b_min,
                    "b_max": self.b_max,
                    "clip": self.clip,
                },
                applied=True,
                random=False,
                invertible=False,
            )
        )
        return bundle

    def scale_tensor(self, tensor: tf.Tensor) -> tf.Tensor:
        """Scale one tensor from source range to target range."""
        tensor = tf.convert_to_tensor(tensor, dtype=self.dtype)
        if self.a_max == self.a_min:
            result = tensor - self.a_min if self.b_min is None else tensor - self.a_min + self.b_min
            return tf.cast(result, dtype=self.dtype)

        tensor = (tensor - self.a_min) / (self.a_max - self.a_min)
        if self.b_min is not None and self.b_max is not None:
            tensor = tensor * (self.b_max - self.b_min) + self.b_min
        if self.clip and self.b_min is not None and self.b_max is not None:
            tensor = tf.clip_by_value(tensor, self.b_min, self.b_max)
        return tf.cast(tensor, dtype=self.dtype)

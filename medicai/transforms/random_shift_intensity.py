from typing import Dict, Sequence, Tuple, Union

import tensorflow as tf

from .tensor_bundle import TensorBundle


class RandShiftIntensity:
    """Randomly shifts the intensity values of specified tensors.

    This transform applies a random offset to the intensity of the tensors
    specified by ``keys``. The shift is applied with probability ``prob``. A
    single offset can be applied to the whole tensor, or separate offsets can
    be sampled for each channel when ``channel_wise=True`` and the tensor is
    4D. Keys that are not present in the input are skipped.

    Args:
        keys (Sequence[str]): Keys of the tensors to shift.
        offsets (Union[float, Tuple[float, float]]): Range from which the
            intensity offset is sampled.
            - If a single float is provided, the sampled range becomes ``(-abs(offsets), abs(offsets))``.
            - If a tuple is provided, the sampled range becomes ``(min(offsets), max(offsets))``.
        prob (float): Probability of applying the shift. Default is ``0.1``.
        channel_wise (bool): If ``True`` and the tensor has shape
            ``(D, H, W, C)``, a separate random offset is sampled for each
            channel. Otherwise, one scalar offset is sampled for the whole
            tensor.

    Example:
        Randomly shift the intensity of an image tensor::

            import tensorflow as tf
            from medicai.transforms import RandShiftIntensity

            shifter = RandShiftIntensity(
                keys=["image"],
                offsets=0.1,
                prob=0.5,
                channel_wise=False,
            )

            image = tf.random.normal((64, 64, 64, 1))
            result = shifter({"image": image})
            shifted_image = result["image"]

    Returns:
        ``TensorBundle``: The transformed output. We can retrieve the shifted
        tensors using the same keys as the input.
    """

    def __init__(
        self,
        keys: Sequence[str],
        offsets: Union[float, Tuple[float, float]],
        prob: float = 0.1,
        channel_wise: bool = False,
    ):
        self.keys = keys
        if isinstance(offsets, (int, float)):
            self.offsets = (-abs(offsets), abs(offsets))
        else:
            self.offsets = (min(offsets), max(offsets))

        self.prob = prob
        self.channel_wise = channel_wise

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:
        """Apply the random intensity shift to the specified tensors in the input TensorBundle.

        Args:
            inputs (TensorBundle): A sample dictionary or ``TensorBundle`` containing
                the tensors to shift.

        Returns:
            TensorBundle: The transformed output. We can retrieve the shifted
            tensors using the same keys as the input.
        """

        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        rand_val = tf.random.uniform(())

        def apply_shift():
            shifted_data = inputs.data.copy()
            for key in self.keys:
                if key in shifted_data:
                    img = shifted_data[key]
                    if self.channel_wise and len(img.shape) == 4:
                        offsets = tf.random.uniform(
                            (1, 1, 1, img.shape[-1]), self.offsets[0], self.offsets[1]
                        )
                    else:
                        offsets = tf.random.uniform((), self.offsets[0], self.offsets[1])
                    shifted_data[key] = img + offsets
            return shifted_data

        def no_shift():
            return inputs.data.copy()

        shifted_data = tf.cond(rand_val <= self.prob, apply_shift, no_shift)
        return TensorBundle(shifted_data, inputs.meta)

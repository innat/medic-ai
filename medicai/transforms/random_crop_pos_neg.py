from typing import Dict, Tuple, Union

from medicai.utils.general import hide_warnings

hide_warnings()
import tensorflow as tf

from .tensor_bundle import TensorBundle


class RandCropByPosNegLabel:
    """
    Randomly crops 3D image patches based on positive and negative label ratios.

    This transformation extracts patches from the input image and label tensor,
    ensuring a balance between positive and negative label samples. The cropping
    is performed based on the given spatial size and sampling ratios.

    Attributes:
        spatial_size (Tuple[int, int, int]): The size of the cropped patch (depth, height, width).
        pos (int): Number of positive samples.
        neg (int): Number of negative samples.
        num_samples (int): Number of patches to extract per input.
    """

    def __init__(
        self, keys, spatial_size: Tuple[int, int, int], pos: int, neg: int, num_samples: int = 1
    ):
        """
        Initializes the RandCropByPosNegLabel transform.

        Args:
            keys (Sequence[str]): Keys of the image and label tensors in the input dictionary.
            spatial_size (Tuple[int, int, int]): The desired spatial size (depth, height, width)
                of the cropped patches.
            pos (int): The number of positive samples to aim for in the batch of cropped patches.
                A positive sample is defined as a patch centered around a positive label.
            neg (int): The number of negative samples to aim for in the batch of cropped patches.
                A negative sample is defined as a patch centered around a negative label.
            num_samples (int): The total number of random patches to extract per input.
                The ratio of positive to negative samples will be approximately pos / (pos + neg).
                Default is 1.

        Raises:
            ValueError: If `pos` or `neg` is negative.
            ValueError: If both `pos` and `neg` are zero.
        """
        if pos < 0 or neg < 0:
            raise ValueError("pos and neg must be non-negative.")
        if pos == 0 and neg == 0:
            raise ValueError("pos and neg cannot both be zero.")

        self.keys = keys
        self.spatial_size = spatial_size
        self.pos = pos
        self.neg = neg
        self.num_samples = num_samples
        self.pos_ratio = pos / (pos + neg)

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:
        """
        Applies the random cropping transformation to the input TensorBundle.

        Args:
            inputs (TensorBundle): A dictionary containing tensors, expected to have
                'image' and 'label' keys with 4D tensors (depth, height, width, channels).

        Returns:
            TensorBundle: A dictionary containing the cropped image and label patches.
                If `num_samples` is 1, the values are single 4D tensors.
                If `num_samples` > 1, the values are 5D tensors (num_samples, depth, height, width, channels).
        """

        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        image = inputs.data["image"]
        label = inputs.data["label"]

        image_patches, label_patches = tf.map_fn(
            lambda _: self._process_sample(image, label),
            tf.range(self.num_samples, dtype=tf.int32),
            dtype=(tf.float32, tf.float32),
        )

        if self.num_samples == 1:
            image_patches = tf.squeeze(image_patches, axis=0)
            label_patches = tf.squeeze(label_patches, axis=0)

        inputs.data["image"] = image_patches
        inputs.data["label"] = label_patches
        return inputs

    def _process_sample(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Randomly decides whether to sample a positive or negative patch and calls the sampler.

        Args:
            image (tf.Tensor): The input image tensor (depth, height, width, channels).
            label (tf.Tensor): The corresponding label tensor (depth, height, width, channels).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the cropped image and label patch.
        """
        rand_val = tf.random.uniform(shape=[], minval=0, maxval=1)
        return self._sample_patch(image, label, positive=rand_val < self.pos_ratio)

    def _sample_patch(
        self, image: tf.Tensor, label: tf.Tensor, positive: bool
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Extracts a patch from the image and label tensor based on sampling criteria.

        Args:
            image (tf.Tensor): The input image tensor.
            label (tf.Tensor): The corresponding label tensor.
            positive (bool): Whether to sample a positive or negative patch.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The cropped image and label patch.
        """
        shape = tf.shape(image, out_type=tf.int32)

        coords = tf.where(label > 0) if positive else tf.where(label == 0)
        if tf.equal(tf.shape(coords)[0], 0):
            coords = tf.where(tf.ones_like(label) > 0)
        idx = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(coords)[0], dtype=tf.int32)
        center = tf.cast(coords[idx], tf.int32)

        start = [tf.maximum(center[i] - self.spatial_size[i] // 2, 0) for i in range(3)]
        end = [tf.minimum(start[i] + self.spatial_size[i], shape[i]) for i in range(3)]
        start = [end[i] - self.spatial_size[i] for i in range(3)]

        patch_image = image[start[0] : end[0], start[1] : end[1], start[2] : end[2], :]
        patch_label = label[start[0] : end[0], start[1] : end[1], start[2] : end[2], :]

        return patch_image, patch_label

from typing import Dict, Tuple, Union

from medicai.utils.general import hide_warnings

hide_warnings()
from typing import Dict, Tuple, Union

import tensorflow as tf

from .tensor_bundle import TensorBundle


class RandCropByPosNegLabel:
    """
    Randomly crops 3D image patches based on positive and negative label ratios.

    This transformation extracts patches from the input image and label tensor,
    ensuring a balance between positive and negative label samples. The cropping
    is performed based on the given spatial size and sampling ratios.
    Optionally uses a reference image and threshold to refine negative sample locations.

    Attributes:
        spatial_size (Tuple[int, int, int]): The size of the cropped patch (depth, height, width).
        pos (int): Number of positive samples to aim for.
        neg (int): Number of negative samples to aim for.
        num_samples (int): Number of patches to extract per input (currently supports only 1).
        image_reference_key (str, optional): Key for an optional reference image tensor.
            If provided, negative samples are centered where label is 0 AND the
            corresponding `image_reference` intensity (maximum across channels) is
            greater than `image_threshold`. Defaults to None.
        image_threshold (float): If `image_reference_key` is provided, this threshold
            is used to filter background regions for negative sampling based on the
            `image_reference` intensity. Defaults to 0.0.
    """

    def __init__(
        self,
        keys,
        spatial_size: Tuple[int, int, int],
        pos: int,
        neg: int,
        num_samples: int = 1,
        image_reference_key: str = None,
        image_threshold: float = 0.0,
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
                Currently, only `num_samples=1` is supported. Default is 1.
            image_reference_key (str, optional): Key in the input dictionary for an
                optional reference image tensor used for refined negative sampling.
                Defaults to None.
            image_threshold (float): Threshold value used with the `image_reference_key`
                to select background regions for negative sampling. Defaults to 0.0.

        Raises:
            ValueError: If `pos` or `neg` is negative.
            ValueError: If both `pos` and `neg` are zero.
            ValueError: If `keys` does not contain exactly two elements.
            ValueError: If `num_samples` is not 1.
        """
        if pos < 0 or neg < 0:
            raise ValueError("pos and neg must be non-negative.")
        if pos == 0 and neg == 0:
            raise ValueError("pos and neg cannot both be zero.")

        if len(keys) != 2:
            class_name = type(self).__name__
            raise ValueError(
                f"{class_name} transformation requires a pair of image and label as keys. "
            )

        if num_samples != 1:
            class_name = self.__class__.__name__
            raise ValueError(f"{class_name} transformation currently supports only num_samples=1.")

        self.keys = keys
        self.spatial_size = spatial_size
        self.pos = pos
        self.neg = neg
        self.num_samples = num_samples
        self.pos_ratio = pos / (pos + neg)
        self.image_reference_key = image_reference_key
        self.image_threshold = image_threshold

    def __call__(self, inputs: Union[TensorBundle, Dict[str, tf.Tensor]]) -> TensorBundle:
        """
        Applies the random cropping transformation to the input TensorBundle.

        Args:
            inputs (TensorBundle): A dictionary containing tensors, expected to have
                the keys specified in `self.keys` ('image' and 'label') and optionally
                `self.image_reference_key`. The image and label tensors should be
                4D (depth, height, width, channels).

        Returns:
            TensorBundle: A dictionary containing the cropped image and label patches.
                The values are 4D tensors (depth, height, width, channels).
        """

        if isinstance(inputs, dict):
            inputs = TensorBundle(inputs)

        # unpack the keys
        image_key, label_key = self.keys

        image = inputs.data[image_key]
        label = inputs.data[label_key]
        image_reference_key = inputs.data.get(self.image_reference_key)

        image_patches, label_patches = tf.map_fn(
            lambda _: self._process_sample(image, label, image_reference_key),
            tf.range(self.num_samples, dtype=tf.int32),
            dtype=(image.dtype, label.dtype),
        )

        if self.num_samples == 1:
            image_patches = tf.squeeze(image_patches, axis=0)
            label_patches = tf.squeeze(label_patches, axis=0)

        inputs.data[image_key] = image_patches
        inputs.data[label_key] = label_patches
        return inputs

    def _process_sample(
        self, image: tf.Tensor, label: tf.Tensor, image_reference_key: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Randomly decides whether to sample a positive or negative patch and calls the sampler.

        Args:
            image (tf.Tensor): The input image tensor (depth, height, width, channels).
            label (tf.Tensor): The corresponding label tensor (depth, height, width, channels).

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing the cropped image and label patch.
        """
        rand_val = tf.random.uniform(shape=[], minval=0, maxval=1)
        return self._sample_patch(
            image, label, image_reference_key, positive=rand_val < self.pos_ratio
        )

    def _sample_patch(
        self, image: tf.Tensor, label: tf.Tensor, image_reference_key: tf.Tensor, positive: bool
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Extracts a patch from the image and label tensor based on sampling criteria,
        considering the optional image reference and threshold for negative sampling.

        Args:
            image (tf.Tensor): The input image tensor.
            label (tf.Tensor): The corresponding label tensor.
            image_reference (tf.Tensor): The optional reference image tensor.
            positive (bool): Whether to sample a positive or negative patch.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The cropped image and label patch.
        """
        shape = tf.shape(image, out_type=tf.int32)

        if positive:
            coords = tf.where(label > 0)
        else:
            if image_reference_key is not None and self.image_threshold is not None:
                # Reduce image_ref across channels to a single channel (e.g., max intensity)
                max_intensity_ref = tf.reduce_max(image_reference_key, axis=-1, keepdims=True)
                # Ensure label also has a comparable channel dimension (if multi-channel, check if ANY is 0)
                label_is_zero = tf.reduce_any(label == 0, axis=-1, keepdims=True)
                valid_mask = label_is_zero & (max_intensity_ref > self.image_threshold)
                coords = tf.where(valid_mask)
            else:
                coords = tf.where(tf.reduce_any(label == 0, axis=-1))

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

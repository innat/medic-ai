import tensorflow as tf
from typing import Tuple
from medicai.transforms import MetaTensor

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
    def __init__(self, keys, spatial_size: Tuple[int, int, int], pos: int, neg: int, num_samples: int = 1):
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

    def __call__(self, inputs: MetaTensor) -> MetaTensor:
        """
        Applies the random cropping transformation.
        
        Args:
            inputs (Dict[str, tf.Tensor]): A dictionary with keys 'image' and 'label' 
            both being 4D tensors.
                Shape: (depth, height, width, channels).
        
        Returns:
            Dict[str, tf.Tensor]: A dictionary containing cropped image and label patches.
        """

        image = inputs.data['image']
        label = inputs.data['label']

        image_patches, label_patches = tf.map_fn(
            lambda _: self._process_sample(image, label),
            tf.range(self.num_samples, dtype=tf.int32),
            dtype=(tf.float32, tf.float32)
        )

        if self.num_samples == 1:
            image_patches = tf.squeeze(image_patches, axis=0)
            label_patches = tf.squeeze(label_patches, axis=0)

        inputs.data['image'] = image_patches
        inputs.data['label'] = label_patches
        return inputs

    def _process_sample(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Samples a patch based on a random decision for positive or negative sampling.
        
        Args:
            image (tf.Tensor): The input image tensor.
            label (tf.Tensor): The corresponding label tensor.
        
        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A cropped image-label pair.
        """
        rand_val = tf.random.uniform(shape=[], minval=0, maxval=1)
        return self._sample_patch(image, label, positive=rand_val < self.pos_ratio)

    def _sample_patch(self, image: tf.Tensor, label: tf.Tensor, positive: bool) -> Tuple[tf.Tensor, tf.Tensor]:
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

        patch_image = image[start[0]:end[0], start[1]:end[1], start[2]:end[2], :]
        patch_label = label[start[0]:end[0], start[1]:end[1], start[2]:end[2], :]

        return patch_image, patch_label

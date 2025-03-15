
import tensorflow as tf

class RandomIntensity:
    def __init__(self, offsets, prob=0.1, channel_wise=False):
        """
        Args:
            offsets: Tuple (min_offset, max_offset) or a single float.
            prob: Probability of applying the intensity shift.
            channel_wise: If True, applies different shifts per channel.
        """
        if isinstance(offsets, (int, float)):
            self.offsets = (-abs(offsets), abs(offsets))
        else:
            self.offsets = (min(offsets), max(offsets))
        
        self.prob = prob
        self.channel_wise = channel_wise
    
    def __call__(self, inputs):
        """
        Args:
            img: 3D Tensor of shape (depth, height, width) or (channels, depth, height, width).
        Returns:
            Tensor with intensity shifted.
        """

        if tf.random.uniform(()) > self.prob:
            return inputs

        img = inputs['image']
        label = inputs['label']
        
        if self.channel_wise and len(img.shape) == 4:  # (D, H, W, C)
            offsets = tf.random.uniform((1, 1, 1, img.shape[-1]), self.offsets[0], self.offsets[1])
        else:
            offsets = tf.random.uniform((), self.offsets[0], self.offsets[1])

        return {'image':img + offsets, 'label':tf.convert_to_tensor(label, dtype=label.dtype)}
import tensorflow as tf

class CropForeground:
    """
    Crops an image and its corresponding label to the smallest bounding box
    that contains non-zero values in the source data.
    
    Args:
        source_key (str): The key in the input dictionary to use for determining the bounding box.
                          Defaults to 'image'.
    """
    
    def __init__(self, source_key="image"):
        self.source_key = source_key

    def __call__(self, inputs):
        """
        Crops the 'image' and 'label' tensors based on the bounding box of non-zero values
        in the specified source tensor.
        
        Args:
            inputs (dict): Dictionary containing 'image' and 'label' tensors.
                           The source tensor used for cropping is determined by `source_key`.
        
        Returns:
            dict: Dictionary with cropped 'image' and 'label' tensors.
        """
        source_data = inputs[self.source_key]
        image = inputs['image']
        label = inputs['label']

        if len(image.shape) != 4 or len(label.shape) != 4:
            raise ValueError("Input tensors must have shape (height, width, depth, channels).")

        min_coords, max_coords = self.find_bounding_box(source_data)

        # Crop the image and label using the bounding box
        cropped_image = image[min_coords[0]:max_coords[0] + 1,
                              min_coords[1]:max_coords[1] + 1,
                              min_coords[2]:max_coords[2] + 1, :]
        cropped_label = label[min_coords[0]:max_coords[0] + 1,
                              min_coords[1]:max_coords[1] + 1,
                              min_coords[2]:max_coords[2] + 1, :]

        return {'image': cropped_image, 'label': cropped_label}

    def find_bounding_box(self, image):
        """
        Finds the minimum and maximum coordinates of non-zero values in the given tensor.
        
        Args:
            image (tf.Tensor): The input tensor to determine the bounding box.
        
        Returns:
            tuple: (min_coords, max_coords) representing the bounding box corners.
        """
        mask = tf.reduce_any(tf.not_equal(image, 0), axis=-1)
        coords = tf.where(mask)
        min_coords = tf.reduce_min(coords, axis=0)
        max_coords = tf.reduce_max(coords, axis=0)
        return min_coords, max_coords
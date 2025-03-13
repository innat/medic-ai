
import tensorflow as tf

def CropForeground(source_key="image"):
    def wrapper(inputs):
        source_data = inputs[source_key]
        image = inputs['image']
        label = inputs['label']

        if len(image.shape) != 4 or len(label.shape) != 4:
            raise ValueError("Input tensors must have shape (height, width, depth, channels).")

        min_coords, max_coords = find_bounding_box(source_data)

        # crop the image and label using the bounding box
        cropped_image = image[min_coords[0]:max_coords[0] + 1,
                              min_coords[1]:max_coords[1] + 1,
                              min_coords[2]:max_coords[2] + 1, :]
        cropped_label = label[min_coords[0]:max_coords[0] + 1,
                              min_coords[1]:max_coords[1] + 1,
                              min_coords[2]:max_coords[2] + 1, :]

        return {'image': cropped_image, 'label': cropped_label}

    def find_bounding_box(image):
        mask = tf.reduce_any(tf.not_equal(image, 0), axis=-1)
        coords = tf.where(mask)
        min_coords = tf.reduce_min(coords, axis=0)
        max_coords = tf.reduce_max(coords, axis=0)
        return min_coords, max_coords

    return wrapper
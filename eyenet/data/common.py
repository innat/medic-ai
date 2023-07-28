
import tensorflow as tf

def augment(image, target_shape):
    image = tf.image.resize(image, *target_shape)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.5, 2.0)
    image = tf.image.random_brightness(image, 0.5)
    return image
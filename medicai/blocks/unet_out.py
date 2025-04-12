from keras import layers


def UnetOutBlock(num_classes, dropout_rate=None, activation=None):
    """The output block of a 3D UNet, consisting of a 1x1x1 convolutional layer
    to map the features to the desired number of classes, with optional dropout
    and a final activation function.

    Args:
        num_classes (int): The number of output classes for the segmentation task.
            This determines the number of output channels of the convolutional layer.
        dropout_rate (Optional[float]): The dropout rate (between 0 and 1). If None,
            no dropout is applied (default: None).
        activation (Optional[Union[str, layers.Activation]]): The activation function
            to apply to the output of the convolutional layer. This can be a string
            (e.g., 'softmax', 'sigmoid') or a Keras activation layer instance.
            If None, no activation is applied (default: None).

    Returns:
        Callable: A function that takes an input tensor and returns the output
            tensor after applying the 1x1x1 convolution and optional dropout
            and activation.
    """

    def wrapper(inputs):
        x = layers.Conv3D(
            num_classes, kernel_size=1, strides=1, use_bias=True, activation=activation
        )(inputs)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)
        return x

    return wrapper

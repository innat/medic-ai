from eyenet.layers import ChannelWiseAttention, ElementWiseAttention
from tensorflow import keras
from tensorflow.keras import layers


def FunctionalModel(config):

    input_shape = (config.dataset.image_size,) * 2
    num_classes = config.dataset.num_classes
    input = keras.Input(shape=(*input_shape, 3))

    def apply(entity):
        model_x = entity(weights=config.model.weight, include_top=False, input_tensor=input)
        feat_x = layers.Dense(num_classes, activation="relu")(model_x.output)

        channel_x = ChannelWiseAttention(config)(model_x.get_layer(config.model.layers[0]).output)
        element_x = ElementWiseAttention(config)(channel_x)

        feat_x = layers.GlobalAveragePooling2D()(feat_x)
        element_x = layers.GlobalAveragePooling2D()(element_x)
        feat_element_x = layers.concatenate([feat_x, element_x])

        feat_element_x = layers.Dense(
            num_classes, activation="softmax", name="primary", dtype="float32"
        )(feat_element_x)
        element_x = layers.Dense(
            num_classes, activation="softmax", name="auxilary", dtype="float32"
        )(element_x)
        return input, feat_element_x, element_x

    return apply

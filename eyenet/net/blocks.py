
from tensorflow import keras
from tensorflow.keras import layers
from layers import ChannelWiseAttention
from layers import ElementWiseAttention

num_of_class = 5

def FunctionalModel(input_shape):

    input = keras.Input(shape=(*input_shape, 3))

    def apply(entity):
        model_x = entity[0](
            weight=entity[1],
            include_top=False,
            input_tensor=input
        )
        feat_x = layers.Dense(num_of_class, activation='relu')(model_x.output)

        channel_x = ChannelWiseAttention()(
            model_x.get_layer("block5a_expand_conv").output
        )
        element_x = ElementWiseAttention()(
            channel_x
        )

        feat_x = layers.GlobalAveragePooling2D()(feat_x)
        element_x = layers.GlobalAveragePooling2D()(element_x)
        feat_element_x = layers.concatenate([feat_x, element_x])

        feat_element_x = layers.Dense(
            num_of_class, 
            activation='softmax', 
            name='primary', 
            dtype='float32'
        )(feat_element_x) 

        element_x = layers.Dense(
            num_of_class, 
            activation='softmax', 
            name='auxilary', 
            dtype='float32'
        )(element_x)

        return input, feat_element_x, element_x

    return apply
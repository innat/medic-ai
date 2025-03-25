import keras
from keras import layers

from medicai.layers import ChannelWiseAttention, ElementWiseAttention

model_instance = {
    "efficientnetb0": keras.applications.EfficientNetB0,
    "efficientnetb1": keras.applications.EfficientNetB1,
    "efficientnetb2": keras.applications.EfficientNetB2,
    "efficientnetb3": keras.applications.EfficientNetB3,
}


def AttentionBlocks(num_classes):
    def apply(incoming):
        feat_x = layers.Dense(num_classes, activation="relu")(incoming.output)
        channel_x = ChannelWiseAttention(config)(incoming.get_layer("block5a_expand_conv").output)
        element_x = ElementWiseAttention(config)(channel_x)

        feat_x = layers.GlobalAveragePooling2D()(feat_x)
        element_x = layers.GlobalAveragePooling2D()(element_x)
        feat_element_x = layers.concatenate([feat_x, element_x])

        feat_element_x = layers.Dense(
            num_classes, activation="softmax", name="primary", dtype="float32"
        )(feat_element_x)

        return feat_element_x

    return apply


def DuelAttentionNet(config):
    attnblock = AttentionBlocks(config)
    backbone = model_instance[config.model.name]

    input_shape = (config.dataset.image_size,) * 2
    input_tensor = keras.Input(shape=(*input_shape, 3))
    backbone = backbone(weights=config.model.weight, include_top=False, input_tensor=input_tensor)
    base_maps = attnblock(backbone)

    model = keras.Model(inputs=[input_tensor], outputs=[base_maps])

    return model

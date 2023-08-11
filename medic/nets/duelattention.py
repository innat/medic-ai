from medic.layers.attention import ChannelWiseAttention2D
from medic.layers.attention import ElementWiseAttention2D
from medic.losses import WeightedKappaLoss
from medic.metrics import CohenKappa


from tensorflow import keras
from tensorflow.keras import layers as nn


model_instance = {
    "efficientnetb0": keras.applications.EfficientNetB0,
    "efficientnetb1": keras.applications.EfficientNetB1,
    "efficientnetb2": keras.applications.EfficientNetB2,
    "efficientnetb3": keras.applications.EfficientNetB3,
}


def AttentionBlocks2D(config):
    num_classes = config.dataset.num_classes

    def apply(incoming):
        feat_x = nn.Dense(num_classes, activation="relu")(incoming.output)
        channel_x = ChannelWiseAttention2D(config)(incoming.get_layer("block5a_expand_conv").output)
        element_x = ElementWiseAttention2D(config)(channel_x)

        feat_x = nn.GlobalAveragePooling2D()(feat_x)
        element_x = nn.GlobalAveragePooling2D()(element_x)
        feat_element_x = nn.concatenate([feat_x, element_x])

        feat_element_x = nn.Dense(
            num_classes, activation="softmax", name="primary", dtype="float32"
        )(feat_element_x)

        return feat_element_x

    return apply


def DuelAttentionNet2D(config):
    attnblock = AttentionBlocks2D(config)
    backbone = model_instance[config.model.name]

    input_shape = (config.dataset.image_size,) * 2
    input_tensor = keras.Input(shape=(*input_shape, 3))
    backbone = backbone(weights=config.model.weight, include_top=False, input_tensor=input_tensor)
    base_maps = attnblock(backbone)

    model = keras.Model(inputs=[input_tensor], outputs=[base_maps])

    return model


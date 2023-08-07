from eyenet.utils import GradientAccumulator
from eyenet.layers.attention import ChannelWiseAttention
from eyenet.layers.attention import ElementWiseAttention
from eyenet.losses import WeightedKappaLoss
from eyenet.metrics import CohenKappa


from tensorflow import keras
from tensorflow.keras import layers as nn


model_instance = {"efficientnetb0": keras.applications.EfficientNetB0}


def AttentionBlocks(config):
    num_classes = config.dataset.num_classes

    def apply(incoming):
        feat_x = nn.Dense(num_classes, activation="relu")(incoming.output)
        channel_x = ChannelWiseAttention(config)(incoming.get_layer("block5a_expand_conv").output)
        element_x = ElementWiseAttention(config)(channel_x)

        feat_x = nn.GlobalAveragePooling2D()(feat_x)
        element_x = nn.GlobalAveragePooling2D()(element_x)
        feat_element_x = nn.concatenate([feat_x, element_x])

        feat_element_x = nn.Dense(
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
    model = get_compiled(model, config)

    return model


def get_compiled(model, config):
    if config.losses == "cohen_kappa":
        loss_fn = WeightedKappaLoss(
            num_classes=config.dataset.num_classes,
            weightage="quadratic",
        )

    if config.metrics == "cohen_kappa":
        metrics_fn = CohenKappa(
            num_classes=config.dataset.num_classes,
            weightage="quadratic",
        )
   
    if config.trainer.optimizer == "adam":
        optim = keras.optimizers.Adam(learning_rate=config.trainer.learning_rate)

    model.compile(
        loss=loss_fn,
        metrics=metrics_fn,
        optimizer=optim,
    )

    return model

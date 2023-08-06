from eyenet.utils import GradientAccumulator
from eyenet.layers.attention import ChannelWiseAttention
from eyenet.layers.attention import ElementWiseAttention
from eyenet.losses import WeightedKappaLoss
from eyenet.metrics import CohenKappa


from tensorflow import keras
from tensorflow.keras import layers as nn


model_instance = {"efficientnet": keras.applications.EfficientNetB0}


def AttentionBlocks(config):
    num_classes = config.dataset.num_classes

    def apply(incoming):
        feat_x = nn.Dense(num_classes, activation="relu")(incoming.output)
        channel_x = ChannelWiseAttention(config)(incoming.get_layer(config.model.layers[0]).output)
        element_x = ElementWiseAttention(config)(channel_x)

        feat_x = nn.GlobalAveragePooling2D()(feat_x)
        element_x = nn.GlobalAveragePooling2D()(element_x)
        feat_element_x = nn.concatenate([feat_x, element_x])

        feat_element_x = nn.Dense(
            num_classes, activation="softmax", name="primary", dtype="float32"
        )(feat_element_x)
        element_x = nn.Dense(num_classes, activation="softmax", name="auxilary", dtype="float32")(
            element_x
        )

        return feat_element_x, element_x

    return apply


def DuelAttentionNet(config):
    grad_accumulation = config.trainer.gradient_accumulation
    attnblock = AttentionBlocks(config)
    backbone = model_instance[config.model.name]

    input_shape = (config.dataset.image_size,) * 2
    input_tensor = keras.Input(shape=(*input_shape, 3))
    backbone = backbone(weights=config.model.weight, include_top=False, input_tensor=input_tensor)
    base_maps, attn_maps = attnblock(backbone)

    if grad_accumulation:
        model = GradientAccumulator(
            n_gradients=grad_accumulation,
            inputs=[input_tensor],
            outputs=[base_maps, attn_maps],
        )
    else:
        model = keras.Model(inputs=[input_tensor], outputs=[base_maps, attn_maps])

    model = get_compiled(model, config)
    return model


def get_compiled(model, config):
    if config.losses.primary == "cohen_kappa_loss":
        primary_loss = WeightedKappaLoss(
            num_classes=config.dataset.num_classes,
            weightage="quadratic",
            name="primary_loss",
        )

    if config.losses.auxilary == "categorical_crossentropy":
        auxilary_loss = keras.losses.CategoricalCrossentropy(
            label_smoothing=config.losses.label_smoothing, name="aux_loss"
        )

    if config.metrics.primary == "cohen_kappa":
        primary_metrics = CohenKappa(
            num_classes=config.dataset.num_classes,
            weightage="quadratic",
            name="primary_metrics",
        )
    if config.metrics.auxilary == "accuracy":
        auxilary_metrics = keras.metrics.CategoricalAccuracy(name="auxilary_metrics")

    if config.trainer.optimizer == "adam":
        optim = keras.optimizers.Adam(learning_rate=config.trainer.learning_rate)

    model.compile(
        loss={
            "primary": primary_loss,
            "auxilary": auxilary_loss,
        },
        metrics={
            "primary": [
                "accuracy",
                primary_metrics,
            ],
            "auxilary": [auxilary_metrics],
        },
        loss_weights={"primary": 1.0, "auxilary": 0.3},
        optimizer=optim,
    )

    return model

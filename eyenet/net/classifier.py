from .blocks import FunctionalModel
from eyenet.layers import GradientAccumulator
from tensorflow import keras
from tensorflow.keras import applications
import tensorflow_addons as tfa

model_instance = {"efficientnet": applications.EfficientNetB0}


def get_model(config):
    image_size = config.dataset.image_size  # noqa: F841
    grad_accumulation = config.trainer.gradient_accumulation

    # TODO: Update and make general
    backbone = model_instance[config.model.name]
    func_model = FunctionalModel(config)
    input, base_maps, attn_maps = func_model(backbone)

    if grad_accumulation:
        model = GradientAccumulator(
            n_gradients=grad_accumulation,
            inputs=[input],
            outputs=[base_maps, attn_maps],
        )
    else:
        model = keras.Model(inputs=[input], outputs=[base_maps, attn_maps])

    model = get_compiled(model, config)

    return model


def get_compiled(model, config):

    if config.losses.primary == "cohen_kappa_loss":
        primary_loss = tfa.losses.WeightedKappaLoss(
            num_classes=config.dataset.num_classes,
            weightage="quadratic",
            name="primary_loss",
        )

    if config.losses.auxilary == "categorical_crossentropy":
        auxilary_loss = keras.losses.CategoricalCrossentropy(
            label_smoothing=config.losses.label_smoothing, name="aux_loss"
        )

    if config.metrics.primary == "cohen_kappa":
        primary_metrics = tfa.metrics.CohenKappa(
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

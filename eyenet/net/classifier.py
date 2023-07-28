from .blocks import FunctionalModel
from layers import GradientAccumulator
from tensorflow import keras
from tensorflow.keras import applications


def get_model(config):
    image_size = config.dataset.image_size
    num_classes = config.dataset.num_classes
    grad_accumulation = config.trainer.gradient_accumulation

    # TODO: Update and make general
    backbone = applications.EfficientNetB0(
        include_top=False,
        weights=config.model.weight,
        input_shape=(image_size, image_size, 3)
    )

    func_model = FunctionalModel(config)
    input, base_maps, attn_maps = func_model(backbone)

    if grad_accumulation:
        model = GradientAccumulator(
            n_gradients=grad_accumulation,
            inputs=[input],
            outputs=[base_maps, attn_maps],
        )
    else:
        model = keras.Model(
            inputs=[input], outputs=[base_maps, attn_maps]
        )
    
    return model


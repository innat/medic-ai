from blocks import FunctionalModel
from layers import GradientAccumulator
from tensorflow import keras
from tensorflow.keras import applications

input_shape = (224, 224)
num_of_class = 5
accumulate_gradient = True

image_net_model = applications.EfficientNetB0()
func_model = FunctionalModel(input_shape)
input, base_maps, attn_maps = func_model(image_net_model)

if accumulate_gradient:
    model = GradientAccumulator(
        n_gradients=10,
        inputs=[input],
        outputs=[base_maps, attn_maps],
    )
else:
    model = keras.Model(inputs=[input], outputs=[base_maps, attn_maps])

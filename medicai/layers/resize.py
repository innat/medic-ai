from keras import layers, ops

from medicai.utils import resize_volumes


class ResizingND(layers.Layer):
    def __init__(self, target_shape, interpolation="nearest", **kwargs):
        super().__init__(**kwargs)

        if interpolation not in ("nearest", "bilinear", "trilinear"):
            raise ValueError(
                f"Interpolation must be one of ('nearest', 'bilinear', 'trilinear') but got '{interpolation}'"
            )

        self.target_shape = target_shape
        self.interpolation = interpolation

    def build(self, input_shape):
        self.spatial_dims = len(input_shape) - 2
        self.channels = input_shape[-1]
        if self.spatial_dims not in (2, 3):
            raise ValueError(
                f"{self.__class__.__name__} only supports 2D or 3D inputs. "
                f"Got spatial_dims={self.spatial_dims}"
            )
        super().build(input_shape)

    def call(self, inputs):
        if self.spatial_dims == 3:
            d, h, w = self.target_shape
            return resize_volumes(inputs, d, h, w, method=self.interpolation)
        else:
            return ops.image.resize(inputs, self.target_shape, interpolation=self.interpolation)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.target_shape, self.channels)

    def get_config(self):
        config = super().get_config()
        config.update({"target_shape": self.target_shape, "interpolation": self.interpolation})
        return config

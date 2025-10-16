from keras import layers, ops

from medicai.utils import resize_volumes


class ResizingND(layers.Layer):
    def __init__(self, target_shape=None, scale_factor=None, interpolation="nearest", **kwargs):
        super().__init__(**kwargs)

        if target_shape is None and scale_factor is None:
            raise ValueError("Either `target_shape` or `scale_factor` must be provided")
        if target_shape is not None and scale_factor is not None:
            raise ValueError("Only one of `target_shape` or `scale_factor` can be provided")

        if interpolation not in ("nearest", "bilinear", "trilinear"):
            raise ValueError(
                f"Interpolation must be one of ('nearest', 'bilinear', 'trilinear') but got '{interpolation}'"
            )

        self.target_shape = target_shape
        self.scale_factor = scale_factor
        self.interpolation = interpolation
        # Store the original values for serialization
        self._original_scale_factor = scale_factor

    def build(self, input_shape):
        self.spatial_dims = len(input_shape) - 2
        self.channels = input_shape[-1]

        if self.spatial_dims not in (2, 3):
            raise ValueError(
                f"{self.__class__.__name__} only supports 2D or 3D inputs. "
                f"Got spatial_dims={self.spatial_dims}"
            )
        if self.spatial_dims == 2 and self.interpolation not in ("nearest", "bilinear"):
            raise ValueError(
                f"For 2D inputs, interpolation must be one of ('nearest', 'bilinear'), but got '{self.interpolation}'."
            )
        if self.spatial_dims == 3 and self.interpolation not in ("nearest", "trilinear"):
            raise ValueError(
                f"For 3D inputs, interpolation must be one of ('nearest', 'trilinear'), but got '{self.interpolation}'."
            )

        # Compute target_shape from scale_factor if provided
        if self.scale_factor is not None:
            if isinstance(self.scale_factor, (int, float)):
                # Uniform scaling for all spatial dimensions
                self.target_shape = [
                    int(input_shape[i + 1] * self.scale_factor) for i in range(self.spatial_dims)
                ]
            elif isinstance(self.scale_factor, (list, tuple)):
                if len(self.scale_factor) != self.spatial_dims:
                    raise ValueError(
                        f"scale_factor must have length {self.spatial_dims} for {self.spatial_dims}D inputs, "
                        f"got {len(self.scale_factor)}"
                    )
                # Different scaling for each spatial dimension
                self.target_shape = [
                    int(input_shape[i + 1] * self.scale_factor[i]) for i in range(self.spatial_dims)
                ]
            else:
                raise ValueError(
                    f"{self.__class__.__name__} `scale_factor` must be int, float, list or tuple "
                    f"Got {type(self.scale_factor)}"
                )
        else:
            if len(self.target_shape) != self.spatial_dims:
                raise ValueError(
                    f"target_shape must have {self.spatial_dims} elements for {self.spatial_dims}D inputs, "
                    f"got {len(self.target_shape)}"
                )

        # Validate the computed shape
        if any(dim <= 0 for dim in self.target_shape):
            raise ValueError(
                f"{self.__class__.__name__} Invalid target shape: {self.target_shape}. "
                "All dimensions must be positive."
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
        config.update(
            {
                "target_shape": self.target_shape if self.scale_factor is None else None,
                "scale_factor": self._original_scale_factor,
                "interpolation": self.interpolation,
            }
        )
        return config

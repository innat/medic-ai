import keras
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

        if scale_factor is not None:
            if isinstance(scale_factor, (int, float)):
                if scale_factor <= 0:
                    raise ValueError(f"scale_factor must be positive, got {scale_factor}")
            elif isinstance(scale_factor, (list, tuple)):
                if len(scale_factor) == 0:
                    raise ValueError("scale_factor list/tuple cannot be empty")

                for i, factor in enumerate(scale_factor):
                    if not isinstance(factor, (int, float)):
                        raise ValueError(
                            f"All scale_factor elements must be int or float, "
                            f"but got {type(factor)} at index {i}"
                        )
                    if factor <= 0:
                        raise ValueError(
                            f"All scale_factor elements must be positive, "
                            f"got {factor} at index {i}"
                        )
            else:
                raise ValueError(
                    f"scale_factor must be int, float, list or tuple, "
                    f"but got {type(scale_factor)}"
                )

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

        # Normalize scale_factor to always be a list
        if self.scale_factor is not None:
            if isinstance(self.scale_factor, (int, float)):
                self.scale_factor = [self.scale_factor] * self.spatial_dims

            if len(self.scale_factor) != self.spatial_dims:
                raise ValueError(
                    f"scale_factor must have length {self.spatial_dims} for {self.spatial_dims}D inputs, "
                    f"but got {len(self.scale_factor)}"
                )

        # validation for `target_shape` length
        if self.target_shape is not None and len(self.target_shape) != self.spatial_dims:
            raise ValueError(
                f"target_shape must have {self.spatial_dims} elements for {self.spatial_dims}D inputs, "
                f"got {len(self.target_shape)}"
            )

        # Validate `target_shape` values if provided
        if self.target_shape is not None:
            for i, dim in enumerate(self.target_shape):
                if dim is None:
                    # Allow dynamic shapes for flexible input resizing
                    continue
                if not isinstance(dim, int) or dim <= 0:
                    raise ValueError(
                        f"All target_shape dimensions must be positive integers or None, "
                        f"got {dim} (type: {type(dim)}) at index {i}"
                    )

        # Jax doesn't support dynamic shape.
        if keras.config.backend() == "jax":
            # Precompute target shape statically to avoid tracing
            if self.scale_factor is not None:
                self.computed_target_shape = [
                    int(round(input_shape[i + 1] * self.scale_factor[i]))
                    for i in range(self.spatial_dims)
                ]
            else:
                self.computed_target_shape = list(self.target_shape)
        else:
            self.computed_target_shape = None

        super().build(input_shape)

    def call(self, inputs):

        if keras.config.backend() == "jax":
            target_shape = self.computed_target_shape
        else:
            input_shape = ops.shape(inputs)

            if self.scale_factor is not None:
                target_shape = [
                    ops.cast(input_shape[i + 1] * self.scale_factor[i], "int32")
                    for i in range(self.spatial_dims)
                ]
            else:
                target_shape = self.target_shape

        if self.spatial_dims == 3:
            return resize_volumes(inputs, *target_shape, method=self.interpolation)
        else:
            return ops.image.resize(inputs, target_shape, interpolation=self.interpolation)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]

        if keras.config.backend() == "jax":
            new_spatial = self.computed_target_shape
        elif self.scale_factor is not None:
            new_spatial = [
                (
                    int(input_shape[i + 1] * self.scale_factor[i])
                    if input_shape[i + 1] is not None
                    else None
                )
                for i in range(self.spatial_dims)
            ]
        else:
            new_spatial = list(self.target_shape)

        return (batch_size, *new_spatial, self.channels)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "target_shape": self.target_shape,
                "scale_factor": self.scale_factor,
                "interpolation": self.interpolation,
            }
        )
        return config

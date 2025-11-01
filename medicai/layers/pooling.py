from keras import layers, ops

from medicai.utils.swi_utils import ensure_tuple_rep


class AdaptivePooling2D(layers.Layer):
    """Parent class for 2D pooling layers with adaptive kernel size.

    This layer performs pooling over the input height (H) and width (W) dimensions
    such that the output dimensions match the specified `output_size`.
    It supports arbitrary input sizes, even when the input H/W is not divisible
    by the output H/W.

    It assumes the 'channels_last' data format: (batch, H, W, C).

    Args:
        reduce_function: The reduction method to apply, e.g. `keras.ops.mean` or
          `keras.ops.max`.
        output_size: An integer or tuple/list of 2 integers specifying
          (pooled_rows, pooled_cols). The new size of the H and W dimensions.
    """

    def __init__(
        self,
        reduce_function,
        output_size,
        **kwargs,
    ):
        self.reduce_function = reduce_function
        self.output_size = ensure_tuple_rep(output_size, 2)
        super().__init__(**kwargs)

    def call(self, inputs):
        # Assume channels_last: (batch, H, W, C)
        h_bins = self.output_size[0]
        w_bins = self.output_size[1]

        # Get input dimensions H and W (at axes 1 and 2)
        input_shape = ops.shape(inputs)
        h = input_shape[1]
        w = input_shape[2]

        # Calculate the start and end indices for each bin using linspace
        h_idx = ops.linspace(0, h, h_bins + 1)
        w_idx = ops.linspace(0, w, w_bins + 1)

        outputs = []
        for i in range(h_bins):
            row_outputs = []

            # Calculate height indices (axes 1)
            h_start = ops.cast(ops.floor(h_idx[i]), "int32")
            h_end = ops.cast(ops.ceil(h_idx[i + 1]), "int32")
            h_end = ops.where(h_end > h, h, h_end)

            for j in range(w_bins):
                # Calculate width indices (axes 2)
                w_start = ops.cast(ops.floor(w_idx[j]), "int32")
                w_end = ops.cast(ops.ceil(w_idx[j + 1]), "int32")
                w_end = ops.where(w_end > w, w, w_end)

                # Slicing: inputs[:, H_slice, W_slice, :]
                region = inputs[:, h_start:h_end, w_start:w_end, :]

                # Reduction axes are H (1) and W (2)
                pooled = self.reduce_function(region, axis=[1, 2], keepdims=True)
                row_outputs.append(pooled)

            # Concatenate pooled regions along the width axis (axis 2)
            outputs.append(ops.concatenate(row_outputs, axis=2))

        # Concatenate pooled rows along the height axis (axis 1)
        outputs = ops.concatenate(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        shape = (
            input_shape[0],
            self.output_size[0],
            self.output_size[1],
            input_shape[3],
        )
        return shape

    def get_config(self):
        config = {
            "output_size": self.output_size,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class AdaptiveAveragePooling2D(AdaptivePooling2D):
    """Adaptive Average Pooling 2D layer (channels_last).

    This layer resizes the 2D input (H, W) to a fixed size using average pooling.

    Args:
        output_size: An integer or tuple/list of 2 integers specifying
          (pooled_rows, pooled_cols).
    """

    def __init__(self, output_size, **kwargs):
        super().__init__(ops.mean, output_size, **kwargs)


class AdaptiveMaxPooling2D(AdaptivePooling2D):
    """Adaptive Max Pooling 2D layer (channels_last).

    This layer resizes the 2D input (H, W) to a fixed size using max pooling.

    Args:
        output_size: An integer or tuple/list of 2 integers specifying
          (pooled_rows, pooled_cols).
    """

    def __init__(self, output_size, **kwargs):
        super().__init__(ops.max, output_size, **kwargs)


class AdaptivePooling3D(layers.Layer):
    """Parent class for 3D pooling layers with adaptive kernel size.

    This layer performs pooling over the input depth (D), height (H), and width (W)
    dimensions such that the output dimensions match the specified `output_size`.
    It supports arbitrary input sizes.

    It assumes the 'channels_last' data format: (batch, D, H, W, C).

    Args:
        reduce_function: The reduction method to apply, e.g. `keras.ops.mean` or
          `keras.ops.max`.
        output_size: An integer or tuple/list of 3 integers specifying
           (pooled_depth, pooled_rows, pooled_cols). The new size of
           the D, H, and W dimensions.
    """

    def __init__(
        self,
        reduce_function,
        output_size,
        **kwargs,
    ):
        self.reduce_function = reduce_function
        self.output_size = ensure_tuple_rep(output_size, 3)
        super().__init__(**kwargs)

    def call(self, inputs):
        # Assume channels_last: (batch, D, H, W, C)
        d_bins = self.output_size[0]
        h_bins = self.output_size[1]
        w_bins = self.output_size[2]

        # Get input dimensions D, H, W (at axes 1, 2, 3)
        input_shape = ops.shape(inputs)
        d = input_shape[1]  # Depth
        h = input_shape[2]  # Height
        w = input_shape[3]  # Width

        # Calculate the start and end indices for each bin using linspace
        d_idx = ops.linspace(0, d, d_bins + 1)
        h_idx = ops.linspace(0, h, h_bins + 1)
        w_idx = ops.linspace(0, w, w_bins + 1)

        depth_outputs = []
        for i in range(d_bins):
            # Calculate Depth indices (axis 1)
            d_start = ops.cast(ops.floor(d_idx[i]), "int32")
            d_end = ops.cast(ops.ceil(d_idx[i + 1]), "int32")
            d_end = ops.where(d_end > d, d, d_end)

            row_outputs = []
            for j in range(h_bins):
                # Calculate Height indices (axis 2)
                h_start = ops.cast(ops.floor(h_idx[j]), "int32")
                h_end = ops.cast(ops.ceil(h_idx[j + 1]), "int32")
                h_end = ops.where(h_end > h, h, h_end)

                col_outputs = []
                for k in range(w_bins):
                    # Calculate Width indices (axis 3)
                    w_start = ops.cast(ops.floor(w_idx[k]), "int32")
                    w_end = ops.cast(ops.ceil(w_idx[k + 1]), "int32")
                    w_end = ops.where(w_end > w, w, w_end)

                    # Slicing: inputs[:, D_slice, H_slice, W_slice, :]
                    region = inputs[:, d_start:d_end, h_start:h_end, w_start:w_end, :]

                    # Reduction axes are D (1), H (2), W (3)
                    pooled = self.reduce_function(region, axis=[1, 2, 3], keepdims=True)
                    col_outputs.append(pooled)

                # Concatenate pooled regions along the width axis (axis 3)
                row_outputs.append(ops.concatenate(col_outputs, axis=3))

            # Concatenate pooled rows along the height axis (axis 2)
            depth_outputs.append(ops.concatenate(row_outputs, axis=2))

        # Concatenate pooled depth slices along the depth axis (axis 1)
        outputs = ops.concatenate(depth_outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        shape = (
            input_shape[0],
            self.output_size[0],
            self.output_size[1],
            self.output_size[2],
            input_shape[4],
        )
        return shape

    def get_config(self):
        config = {
            "output_size": self.output_size,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class AdaptiveAveragePooling3D(AdaptivePooling3D):
    """Adaptive Average Pooling 3D layer (channels_last).

    This layer resizes the 3D input (D, H, W) to a fixed size using average pooling.

    Args:
        output_size: An integer or tuple/list of 3 integers specifying
          (pooled_depth, pooled_rows, pooled_cols).
    """

    def __init__(self, output_size, **kwargs):
        super().__init__(ops.mean, output_size, **kwargs)


class AdaptiveMaxPooling3D(AdaptivePooling3D):
    """Adaptive Max Pooling 3D layer (channels_last).

    This layer resizes the 3D input (D, H, W) to a fixed size using max pooling.

    Args:
        output_size: An integer or tuple/list of 3 integers specifying
          (pooled_depth, pooled_rows, pooled_cols).
    """

    def __init__(self, output_size, **kwargs):
        super().__init__(ops.max, output_size, **kwargs)

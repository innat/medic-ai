from keras import layers, ops

from medicai.utils.swi_utils import ensure_tuple_rep


class AdaptivePooling2D(layers.Layer):
    """Parent class for 2D pooling layers with adaptive kernel size.

    This class only exists for code reuse. It will never be an exposed API.

    Args:
      reduce_function: The reduction method to apply, e.g. `tf.reduce_max`.
      output_size: An integer or tuple/list of 2 integers specifying (pooled_rows, pooled_cols).
        The new size of output channels.
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
        h_bins = self.output_size[0]
        w_bins = self.output_size[1]

        split_cols = ops.split(inputs, h_bins, axis=1)
        split_cols = ops.stack(split_cols, axis=1)
        split_rows = ops.split(split_cols, w_bins, axis=3)
        split_rows = ops.stack(split_rows, axis=3)
        out_vect = self.reduce_function(split_rows, axis=[2, 4])
        return out_vect

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
    """Average Pooling with adaptive kernel size.

    Args:
      output_size: Tuple of integers specifying (pooled_rows, pooled_cols).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`.

    Input shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, height, width, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, height, width)`.

    Output shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
    """

    def __init__(self, output_size, **kwargs):
        super().__init__(ops.mean, output_size, **kwargs)


class AdaptiveMaxPooling2D(AdaptivePooling2D):
    """Max Pooling with adaptive kernel size.

    Args:
      output_size: Tuple of integers specifying (pooled_rows, pooled_cols).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`.

    Input shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, height, width, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, height, width)`.

    Output shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
    """

    def __init__(self, output_size, **kwargs):
        super().__init__(ops.max, output_size, **kwargs)


class AdaptivePooling3D(layers.Layer):
    """Parent class for 3D pooling layers with adaptive kernel size.

    This class only exists for code reuse. It will never be an exposed API.

    Args:
      reduce_function: The reduction method to apply, e.g. `tf.reduce_max`.
      output_size: An integer or tuple/list of 3 integers specifying (pooled_dim1, pooled_dim2, pooled_dim3).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
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
        h_bins = self.output_size[0]
        w_bins = self.output_size[1]
        d_bins = self.output_size[2]
        split_cols = ops.split(inputs, h_bins, axis=1)
        split_cols = ops.stack(split_cols, axis=1)
        split_rows = ops.split(split_cols, w_bins, axis=3)
        split_rows = ops.stack(split_rows, axis=3)
        split_depth = ops.split(split_rows, d_bins, axis=5)
        split_depth = ops.stack(split_depth, axis=5)
        out_vect = self.reduce_function(split_depth, axis=[2, 4, 6])
        return out_vect

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
    """Average Pooling with adaptive kernel size.

    Args:
      output_size: An integer or tuple/list of 3 integers specifying (pooled_depth, pooled_height, pooled_width).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`.

    Input shape:
      - If `data_format='channels_last'`:
        5D tensor with shape `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`.
      - If `data_format='channels_first'`:
        5D tensor with shape `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.

    Output shape:
      - If `data_format='channels_last'`:
        5D tensor with shape `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`.
      - If `data_format='channels_first'`:
        5D tensor with shape `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`.
    """

    def __init__(self, output_size, **kwargs):
        super().__init__(ops.mean, output_size, **kwargs)


class AdaptiveMaxPooling3D(AdaptivePooling3D):
    """Max Pooling with adaptive kernel size.

    Args:
      output_size: An integer or tuple/list of 3 integers specifying (pooled_depth, pooled_height, pooled_width).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`.

    Input shape:
      - If `data_format='channels_last'`:
        5D tensor with shape `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`.
      - If `data_format='channels_first'`:
        5D tensor with shape `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.

    Output shape:
      - If `data_format='channels_last'`:
        5D tensor with shape `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`.
      - If `data_format='channels_first'`:
        5D tensor with shape `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`.
    """

    def __init__(self, output_size, **kwargs):
        super().__init__(ops.max, output_size, **kwargs)

import keras
from keras import layers, ops

from medicai.utils.swi_utils import ensure_tuple_rep


class AdaptivePooling2D(layers.Layer):
    """
    Parent class for 2D pooling layers with adaptive kernel size using the
    efficient, vectorized Two-Pool Gather method. This implementation assumes
    'channels_last' (B, H, W, C) data format.

    Args:
        reduce_function: The Keras pooling op (ops.average_pool or ops.max_pool).
        output_size: An integer or tuple/list of 2 integers (H_out, W_out).
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

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                f"{self.__class__.__name__} expects input with 4 dims (batch, H, W, C), "
                f"but got {input_shape}"
            )
        super().build(input_shape)

    def pool_single_axis(self, inputs, output_size, axis):
        """
        Performs the adaptive pooling logic for a single axis (H or W).
        """
        # 1. Get dimensions and kernel sizes
        input_dim = ops.shape(inputs)[axis]
        small_window, big_window = get_adaptive_window_sizes(input_dim, output_size)

        # 2. Define 2D kernel size
        if axis == 1:
            ksize_small, ksize_big = (small_window, 1), (big_window, 1)
        else:
            ksize_small, ksize_big = (1, small_window), (1, big_window)

        # 3. Perform the two fixed-size pools (Stride=1, Padding='valid')
        small_pool = self.reduce_function(
            inputs,
            pool_size=ksize_small,
            strides=(1, 1),
            padding="valid",
            data_format="channels_last",
        )
        big_pool = self.reduce_function(
            inputs,
            pool_size=ksize_big,
            strides=(1, 1),
            padding="valid",
            data_format="channels_last",
        )

        # 4. Compute indices and gather the results
        small_pool_len = ops.shape(small_pool)[axis]
        gather_indices = compute_adaptive_gather_indices(
            input_dim=input_dim,
            output_size=output_size,
            big_window=big_window,
            small_pool_len=small_pool_len,
        )

        # Concatenate and Gather (vectorized selection of the correct window)
        combined_pool = ops.concatenate([small_pool, big_pool], axis=axis)
        return ops.take(combined_pool, gather_indices, axis=axis)

    def call(self, inputs):
        # 1. Handle (1, 1) case (Global Avg/Max Pooling)
        if self.output_size == (1, 1):
            if self.reduce_function == ops.average_pool:
                return ops.mean(inputs, axis=[1, 2], keepdims=True)
            else:
                return ops.max(inputs, axis=[1, 2], keepdims=True)

        # 2. Sequentially pool Height (axis=1) then Width (axis=2)
        x = self.pool_single_axis(inputs, output_size=self.output_size[0], axis=1)
        x = self.pool_single_axis(x, output_size=self.output_size[1], axis=2)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size[0], self.output_size[1], input_shape[3])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_size": self.output_size,
            }
        )
        return config


class AdaptivePooling3D(keras.layers.Layer):
    """
    Parent class for 3D pooling layers with adaptive kernel size using the
    vectorized Two-Pool Gather method. This implementation assumes
    'channels_last' (B, D, H, W, C) data format.

    Args:
        reduce_function: The Keras pooling op (ops.average_pool or ops.max_pool).
        output_size: An integer or tuple/list of 3 integers (D_out, H_out, W_out).
    """

    def __init__(
        self,
        reduce_function,
        output_size,
        **kwargs,
    ):
        self.reduce_function = reduce_function
        self.output_size = ensure_tuple_rep(output_size, 3)
        self.data_format = "channels_last"
        super().__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 5:
            raise ValueError(
                f"{self.__class__.__name__} expects input with 5 dims (batch, D, H, W, C), "
                f"but got {input_shape}"
            )
        super().build(input_shape)

    def pool_single_axis(self, inputs, output_size, axis):
        # 1. Get dimensions and kernel sizes
        input_dim = ops.shape(inputs)[axis]
        small_window, big_window = get_adaptive_window_sizes(input_dim, output_size)

        # 2. Define Ksize for the pooling operation.
        ksize_ones = [1, 1, 1]

        # Map axis (1, 2, 3) to kernel list index (0, 1, 2)
        spatial_axis = axis - 1

        # Set the pooling dimension in the 3D kernel tuple
        ksize_small_list = ksize_ones[:]
        ksize_small_list[spatial_axis] = small_window
        ksize_small = tuple(ksize_small_list)

        ksize_big_list = ksize_ones[:]
        ksize_big_list[spatial_axis] = big_window
        ksize_big = tuple(ksize_big_list)

        # 3. Perform the two fixed-size pools (Stride=1, Padding='valid')
        small_pool = self.reduce_function(
            inputs,
            pool_size=ksize_small,
            strides=(1, 1, 1),
            padding="valid",
            data_format=self.data_format,
        )
        big_pool = self.reduce_function(
            inputs,
            pool_size=ksize_big,
            strides=(1, 1, 1),
            padding="valid",
            data_format=self.data_format,
        )

        # 4. Compute indices and gather the results
        small_pool_len = ops.shape(small_pool)[axis]
        gather_indices = compute_adaptive_gather_indices(
            input_dim=input_dim,
            output_size=output_size,
            big_window=big_window,
            small_pool_len=small_pool_len,
        )

        # Concatenate and Gather (vectorized selection)
        combined_pool = ops.concatenate([small_pool, big_pool], axis=axis)
        return ops.take(combined_pool, gather_indices, axis=axis)

    def call(self, inputs):
        # 1. Handle (1, 1, 1) case (Global Avg/Max Pooling)
        if self.output_size == (1, 1, 1):
            if self.reduce_function == ops.average_pool:
                return ops.mean(inputs, axis=[1, 2, 3], keepdims=True)
            else:
                return ops.max(inputs, axis=[1, 2, 3], keepdims=True)

        # 2. Sequentially pool Depth (axis 1), Height (axis 2), then Width (axis 3)
        x = self.pool_single_axis(inputs, output_size=self.output_size[0], axis=1)
        x = self.pool_single_axis(x, output_size=self.output_size[1], axis=2)
        x = self.pool_single_axis(x, output_size=self.output_size[2], axis=3)
        return x

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.output_size[0],
            self.output_size[1],
            self.output_size[2],
            input_shape[4],
        )

    def get_config(self):
        config = super().get_config()
        config.update({"output_size": self.output_size})
        return config


class AdaptiveAveragePooling2D(AdaptivePooling2D):
    """Adaptive Average Pooling 2D layer using the vectorized Two-Pool Gather method."""

    def __init__(self, output_size, **kwargs):
        super().__init__(ops.average_pool, output_size, **kwargs)


class AdaptiveMaxPooling2D(AdaptivePooling2D):
    """Adaptive Max Pooling 2D layer using the vectorized Two-Pool Gather method."""

    def __init__(self, output_size, **kwargs):
        super().__init__(ops.max_pool, output_size, **kwargs)


class AdaptiveAveragePooling3D(AdaptivePooling3D):
    """Adaptive Average Pooling 3D layer using the vectorized Two-Pool Gather method."""

    def __init__(self, output_size, **kwargs):
        super().__init__(ops.average_pool, output_size, **kwargs)


class AdaptiveMaxPooling3D(AdaptivePooling3D):
    """Adaptive Max Pooling 3D layer using the vectorized Two-Pool Gather method."""

    def __init__(self, output_size, **kwargs):
        super().__init__(ops.max_pool, output_size, **kwargs)


def get_adaptive_window_sizes(input_dim, output_dim):
    """Calculates the two possible kernel sizes (small/big window) needed
    for adaptive pooling.

    Args:
        input_dim: The input dimension size (D, H, or W).
        output_dim: The target output dimension size.

    Returns:
        A tuple (small_window, big_window) as int32 Tensors.
    """
    input_dim = ops.cast(input_dim, "float32")
    output_dim = ops.cast(output_dim, "float32")
    small_window = ops.cast(ops.ceil(input_dim / output_dim), "int32")
    big_window = small_window + 1
    return small_window, big_window


def compute_adaptive_gather_indices(input_dim, output_size, big_window, small_pool_len):
    """Computes the indices to 'gather' the correct slice from the combined
    (small_pool + big_pool) tensor.

    Args:
        input_dim: The input dimension size.
        output_size: The target output dimension size.
        big_window: The size of the largest possible pooling window.
        small_pool_len: The size of the small_pool result tensor along the axis.

    Returns:
        A 1D int32 Tensor of indices for ops.take.
    """
    input_dim = ops.cast(input_dim, "float32")
    output_size = ops.cast(output_size, "float32")

    # 1. Calculate window start/end boundaries
    window_starts = ops.floor((ops.arange(output_size, dtype="float32") * input_dim) / output_size)
    window_ends = ops.ceil(
        (ops.arange(1, output_size + 1, dtype="float32") * input_dim) / output_size
    )

    # 2. Determine which output bin requires the 'big' window size
    window_sizes = ops.cast(window_ends, "int32") - ops.cast(window_starts, "int32")
    is_big_window = window_sizes == big_window

    # 3. Calculate indices for the combined [small_pool, big_pool] tensor
    small_indices = ops.cast(window_starts, "int32")
    big_indices = small_indices + small_pool_len

    # Select the index from the correct section of the combined pool
    gather_indices = ops.where(is_big_window, big_indices, small_indices)
    return gather_indices

import math

import numpy as np
from keras import layers, ops

from medicai.utils.swi_utils import ensure_tuple_rep


class AdaptivePooling2D(layers.Layer):
    """
    Implements Adaptive Pooling 2D using the 'Two-Pool Gather' method.

    This layer dynamically adjusts the pooling kernel sizes to resize the spatial
    dimensions (H, W) of the input to a fixed `output_size`.

    Attributes:
        reduce_function: The Keras backend pooling function (e.g., ops.average_pool, ops.max_pool).
        output_size: The target (height, width) of the output tensor.
        precomputed_data: Dictionary holding pre-calculated window sizes and gather indices.
    """

    def __init__(self, reduce_function, output_size, **kwargs):
        super().__init__(**kwargs)
        self.reduce_function = reduce_function
        self.output_size = ensure_tuple_rep(output_size, 2)
        self.precomputed_data = {}

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input (B, H, W, C), got shape {input_shape}")

        # Pre-compute pooling parameters for height (axis=1) and width (axis=2)
        for axis_index, axis_name in [(1, "height"), (2, "width")]:
            input_dim = input_shape[axis_index]
            output_dim = self.output_size[axis_index - 1]

            # Only precompute if the input dimension is static (known at build time)
            if input_dim is not None:
                small_window, big_window = get_static_window_sizes(input_dim, output_dim)
                gather_indices = compute_static_gather_indices(input_dim, output_dim, big_window)

                self.precomputed_data[axis_name] = {
                    "small_window": small_window,
                    "big_window": big_window,
                    "gather_indices": gather_indices,
                }

        super().build(input_shape)

    def call(self, inputs):
        # Handle (1, 1) case with global pooling
        if self.output_size == (1, 1):
            if self.reduce_function == ops.average_pool:
                return ops.mean(inputs, axis=[1, 2], keepdims=True)
            else:
                return ops.max(inputs, axis=[1, 2], keepdims=True)

        # Sequential pooling over height (axis=1) then width (axis=2)
        x = self.pool_single_axis(inputs, axis=1)
        x = self.pool_single_axis(x, axis=2)
        return x

    def get_axis_data(self, axis):
        """Utility to retrieve precomputed data for a given axis (1 for H, 2 for W)."""
        axis_name = "height" if axis == 1 else "width"
        if axis_name not in self.precomputed_data:
            # Handle cases where input shape dimension was None during build
            raise NotImplementedError(
                f"Dynamic input shape for {axis_name} is not yet implemented. "
                "The input dimension must be known during build."
            )
        return self.precomputed_data[axis_name]

    def pool_single_axis(self, inputs, axis: int):
        """Applies 1D adaptive pooling along the specified axis (H or W)."""

        data = self.get_axis_data(axis)
        small_window = data["small_window"]
        big_window = data["big_window"]
        gather_indices = data["gather_indices"]

        # Define 2D kernel size for 1D pooling (H, W)
        if axis == 1:  # Height pooling
            ksize_small, ksize_big = (small_window, 1), (big_window, 1)
        else:  # Width pooling
            ksize_small, ksize_big = (1, small_window), (1, big_window)

        # 1. Pool with the smaller kernel (kernel_size, 1) or (1, kernel_size)
        small_pool = self.reduce_function(
            inputs,
            pool_size=ksize_small,
            strides=(1, 1),
            padding="valid",
            data_format="channels_last",
        )

        # 2. Pool with the bigger kernel
        big_pool = self.reduce_function(
            inputs,
            pool_size=ksize_big,
            strides=(1, 1),
            padding="valid",
            data_format="channels_last",
        )

        # 3. Concatenate the two pooled results along the pooling axis
        combined_pool = ops.concatenate([small_pool, big_pool], axis=axis)

        # 4. Gather the correct results using precomputed indices
        result = ops.take(combined_pool, gather_indices, axis=axis)
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size[0], self.output_size[1], input_shape[3])

    def get_config(self):
        config = super().get_config()
        config.update({"output_size": self.output_size})
        return config


class AdaptivePooling3D(layers.Layer):
    """
    Implements Adaptive Pooling 3D using the 'Two-Pool Gather' method.

    This layer dynamically adjusts the pooling kernel sizes to resize the spatial
    dimensions (D, H, W) of the input to a fixed `output_size`.

    Attributes:
        reduce_function: The Keras backend pooling function (e.g., ops.average_pool, ops.max_pool).
        output_size: The target (depth, height, width) of the output tensor.
        precomputed_data: Dictionary holding pre-calculated window sizes and gather indices.
    """

    def __init__(self, reduce_function, output_size, **kwargs):
        super().__init__(**kwargs)
        self.reduce_function = reduce_function
        self.output_size = ensure_tuple_rep(output_size, 3)
        self.data_format = "channels_last"
        self.precomputed_data = {}

    def build(self, input_shape):
        if len(input_shape) != 5:
            raise ValueError(f"Expected 5D input (B, D, H, W, C), got shape {input_shape}")

        # Pre-compute pooling parameters for depth (axis=1), height (axis=2), and width (axis=3)
        for axis_index, axis_name in [(1, "depth"), (2, "height"), (3, "width")]:
            input_dim = input_shape[axis_index]
            output_dim = self.output_size[axis_index - 1]

            # Precompute only if the input dimension is static (not None)
            if input_dim is not None:
                small_window, big_window = get_static_window_sizes(input_dim, output_dim)
                gather_indices = compute_static_gather_indices(input_dim, output_dim, big_window)

                self.precomputed_data[axis_name] = {
                    "small_window": small_window,
                    "big_window": big_window,
                    "gather_indices": gather_indices,
                }

        super().build(input_shape)

    def call(self, inputs):
        # Handle (1, 1, 1) case with global pooling
        if self.output_size == (1, 1, 1):
            if self.reduce_function == ops.average_pool:
                return ops.mean(inputs, axis=[1, 2, 3], keepdims=True)
            else:
                return ops.max(inputs, axis=[1, 2, 3], keepdims=True)

        # Sequential pooling over depth (axis=1), height (axis=2), then width (axis=3)
        x = self.pool_single_axis(inputs, axis=1)
        x = self.pool_single_axis(x, axis=2)
        x = self.pool_single_axis(x, axis=3)
        return x

    def get_axis_data(self, axis):
        """Utility to retrieve precomputed data for a given axis (1 for D, 2 for H, 3 for W)."""
        axis_names = {1: "depth", 2: "height", 3: "width"}
        axis_name = axis_names.get(axis)
        if not axis_name or axis_name not in self.precomputed_data:
            # Handle cases where input shape dimension was None during build
            raise NotImplementedError(
                f"Dynamic input shape for axis {axis} ({axis_name}) is not yet implemented. "
                "The input dimension must be known during build."
            )
        return self.precomputed_data[axis_name]

    def pool_single_axis(self, inputs, axis: int):
        """Applies 1D adaptive pooling along the specified axis (D, H, or W)."""

        data = self.get_axis_data(axis)
        small_window = data["small_window"]
        big_window = data["big_window"]
        gather_indices = data["gather_indices"]

        # Define 3D kernel size for 1D pooling (D, H, W)
        ksize_ones = [1, 1, 1]

        # Map axis (1, 2, 3) to kernel list index (0, 1, 2)
        spatial_axis = axis - 1

        # Build the kernel tuple for the small window
        ksize_small_list = ksize_ones[:]
        ksize_small_list[spatial_axis] = small_window
        ksize_small = tuple(ksize_small_list)

        # Build the kernel tuple for the big window
        ksize_big_list = ksize_ones[:]
        ksize_big_list[spatial_axis] = big_window
        ksize_big = tuple(ksize_big_list)

        # 1. Pool with the smaller kernel
        small_pool = self.reduce_function(
            inputs,
            pool_size=ksize_small,
            strides=(1, 1, 1),
            padding="valid",
            data_format=self.data_format,
        )

        # 2. Pool with the bigger kernel
        big_pool = self.reduce_function(
            inputs,
            pool_size=ksize_big,
            strides=(1, 1, 1),
            padding="valid",
            data_format=self.data_format,
        )

        # 3. Concatenate the two pooled results along the pooling axis
        combined_pool = ops.concatenate([small_pool, big_pool], axis=axis)

        # 4. Gather the correct results using precomputed indices
        result = ops.take(combined_pool, gather_indices, axis=axis)
        return result

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


def get_static_window_sizes(input_dim, output_dim):
    """
    Calculates the two possible kernel sizes ('small' and 'big') needed for
    adaptive pooling across a single dimension.

    Args:
        input_dim: The current size of the input dimension.
        output_dim: The target size of the output dimension.

    Returns:
        A tuple (small_window, big_window) containing the two kernel sizes.
    """
    # The smallest possible kernel size that can cover the input
    small_window = math.ceil(input_dim / output_dim)

    # The larger kernel size is always one greater than the small size
    big_window = small_window + 1
    return small_window, big_window


def compute_static_gather_indices(input_dim, output_size, big_window):
    """
    Computes the indices required to select the correct window result (either from
    the 'small_pool' or the 'big_pool') for each position in the target output.

    This function is based on the Two-Pool Gather adaptive pooling technique.

    Args:
        input_dim: The current size of the input dimension.
        output_size: The target size of the output dimension.
        big_window: The size of the larger pooling kernel.

    Returns:
        A NumPy array of indices (dtype=np.int32) used for ops.take.
    """

    # Calculate ideal window boundaries (inclusive start, exclusive end)
    # These determine where each output window *should* start in the input
    window_starts = np.floor((np.arange(output_size) * input_dim) / output_size).astype(np.int32)
    window_ends = np.ceil((np.arange(1, output_size + 1) * input_dim) / output_size).astype(
        np.int32
    )

    # Determine which windows are 'big' (size = big_window)
    window_sizes = window_ends - window_starts
    is_big_window = window_sizes == big_window

    # Calculate the length of the output for the 'small' kernel pool
    small_window = big_window - 1
    small_pool_len = input_dim - small_window + 1

    # Index for the small pool results: This is simply the window's starting index
    small_indices = window_starts

    # Index for the big pool results: This is the window's starting index,
    # plus an offset equal to the length of the small_pool tensor, as the
    # big_pool tensor is concatenated *after* small_pool.
    big_indices = window_starts + small_pool_len

    # Select the correct index based on whether the window size is 'big' or 'small'
    gather_indices = np.where(is_big_window, big_indices, small_indices)
    return gather_indices.astype(np.int32)

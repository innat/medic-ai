import keras
from keras import ops


def generate_data(
    method="normal",
    shape=(1, 3, 64, 64, 64),
    minval=0.0,
    maxval=1.0,
    dtype="float32",
    mean=0.0,
    stddev=1.0,
    low=0,
    high=3,
):
    """Generates Keras tensors with specified methods and parameters.

    Args:
        method (str): The method to use for data generation.
            Supported methods: "normal", "uniform", "randint", "ones", "zeros".
        shape (tuple): The shape of the tensor to generate.
        minval (float): Minimum value for uniform and randint.
        maxval (float): Maximum value for uniform.
        dtype (str): The data type of the tensor.
        mean (float): Mean for normal distribution.
        stddev (float): Standard deviation for normal distribution.
        low (int): Lower bound (inclusive) for randint.
        high (int): Upper bound (exclusive) for randint.

    Returns:
        keras.KerasTensor: The generated tensor.
    """
    method = method.lower()
    if method == "normal":
        return keras.random.normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype)
    elif method == "uniform":
        return keras.random.uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype)
    elif method == "randint":
        return keras.random.randint(shape=shape, minval=low, maxval=high, dtype="int32")
    elif method == "ones":
        return ops.ones(shape=shape, dtype=dtype)
    elif method == "zeros":
        return ops.zeros(shape=shape, dtype=dtype)
    else:
        raise ValueError(f"Unsupported data generation method: {method}")

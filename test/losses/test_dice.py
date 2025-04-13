import keras
from keras import ops

from medicai.losses.dice import BinaryDiceLoss  # noqa: F401
from medicai.losses.dice import CategoricalDiceLoss  # noqa: F401
from medicai.losses.dice import SparseDiceLoss  # noqa: F401

def generate_data(method="normal", shape=(1, 3, 64, 64, 64), minval=0.0, maxval=1.0, dtype="float32", mean=0.0, stddev=1.0, low=0, high=3):
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

def test_categorical_dice_loss():

    batch_size, num_classes, depth, height, width = 1, 3, 64, 64, 64

    # Generate random uniform logits
    pred = generate_data(
        method="uniform",
        shape=(batch_size, depth, height, width, num_classes),
        minval=0.0,
        maxval=1.0,
        dtype="float32"
    )
    # Generate random integer class indices
    target = generate_data(
        method="randint",
        shape=(batch_size, depth, height, width, 1),
        low=0,
        high=num_classes,
        dtype="int32"
    )

    target_one_hot = ops.one_hot(
        ops.squeeze(target, axis=-1), ops.shape(pred)[-1]
    )

    dice_loss = CategoricalDiceLoss(
        from_logits=True, 
        num_classes=pred.shape[-1], 
    )
    loss = dice_loss(target_one_hot, pred)
    assert loss.shape == (), "Categorical Dice Loss should be a scalar."
    

def test_sparse_categorical_dice_loss():

    batch_size, num_classes, depth, height, width = 1, 3, 64, 64, 64

    # Generate random uniform logits
    pred = generate_data(
        method="uniform",
        shape=(batch_size, depth, height, width, num_classes),
        minval=0.0,
        maxval=1.0,
        dtype="float32"
    )
    # Generate random integer class indices
    target = generate_data(
        method="randint",
        shape=(batch_size, depth, height, width, 1),
        low=0,
        high=num_classes,
        dtype="int32"
    )
    dice_loss = SparseDiceLoss(
        from_logits=True, 
        num_classes=pred.shape[-1], 
    )
    loss = dice_loss(target, pred)
    assert loss.shape == (), "Sparse Dice Loss should be a scalar."

def test_binary_dice_loss():

    batch_size, depth, height, width, channel = 2, 4, 8, 8, 1

    # Generate random integer binary targets
    bin_target = generate_data(
        method="randint",
        shape=(batch_size, depth, height, width, channel),
        low=0,
        high=2,
        dtype="int32"
    )
    # Generate random normal logits
    bin_logit = generate_data(
        method="normal",
        shape=(batch_size, depth, height, width, 1),
        dtype="float32"
    )

    dice_loss = BinaryDiceLoss(
        from_logits=True, 
        num_classes=bin_logit.shape[-1], 
    )
    loss = dice_loss(bin_target, bin_logit)
    assert loss.shape == (), "Binary Dice Loss should be a scalar."


def test_multilabel_binary_dice_loss():

    batch_size, depth, height, width, num_labels = 2, 4, 8, 8, 3

    # Generate random integer multi-label binary targets
    multi_label_target = generate_data(
        method="randint",
        shape=(batch_size, depth, height, width, num_labels),
        low=0,
        high=2,
        dtype="int32"
    )
    # Logit predictions (random normal)
    multi_label_logit = generate_data(
        method="normal",
        shape=(batch_size, depth, height, width, num_labels),
        dtype="float32"
    )

    dice_loss = BinaryDiceLoss(
        from_logits=True, 
        num_classes=multi_label_target.shape[-1],
    )
    loss = dice_loss(multi_label_target, multi_label_logit)
    assert loss.shape == (), "Binary Dice Loss should be a scalar."
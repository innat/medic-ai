from keras import ops

from medicai.losses.generalized_dice import BinaryGeneralizedDiceLoss  # noqa: F401
from medicai.losses.generalized_dice import CategoricalGeneralizedDiceLoss  # noqa: F401
from medicai.losses.generalized_dice import SparseGeneralizedDiceLoss  # noqa: F401

from .data import generate_data


def test_categorical_dice_loss():

    batch_size, num_classes, depth, height, width = 1, 3, 64, 64, 64

    # Generate random uniform logits
    pred = generate_data(
        method="uniform",
        shape=(batch_size, depth, height, width, num_classes),
        minval=0.0,
        maxval=1.0,
        dtype="float32",
    )
    # Generate random integer class indices
    target = generate_data(
        method="randint",
        shape=(batch_size, depth, height, width, 1),
        low=0,
        high=num_classes,
        dtype="int32",
    )

    target_one_hot = ops.one_hot(ops.squeeze(target, axis=-1), ops.shape(pred)[-1])

    dice_loss = CategoricalGeneralizedDiceLoss(
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
        dtype="float32",
    )
    # Generate random integer class indices
    target = generate_data(
        method="randint",
        shape=(batch_size, depth, height, width, 1),
        low=0,
        high=num_classes,
        dtype="int32",
    )
    dice_loss = SparseGeneralizedDiceLoss(
        from_logits=True,
        num_classes=pred.shape[-1],
    )
    loss = dice_loss(target, pred)
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
        dtype="int32",
    )
    # Generate random normal logits
    bin_logit = generate_data(
        method="normal", shape=(batch_size, depth, height, width, 1), dtype="float32"
    )

    dice_loss = BinaryGeneralizedDiceLoss(
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
        dtype="int32",
    )
    # Logit predictions (random normal)
    multi_label_logit = generate_data(
        method="normal", shape=(batch_size, depth, height, width, num_labels), dtype="float32"
    )

    dice_loss = BinaryGeneralizedDiceLoss(
        from_logits=True,
        num_classes=multi_label_logit.shape[-1],
    )
    loss = dice_loss(multi_label_target, multi_label_logit)
    assert loss.shape == (), "Binary Dice Loss should be a scalar."

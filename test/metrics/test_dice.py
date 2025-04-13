import numpy as np
import keras
from keras import ops

from medicai.metrics.dice import BinaryDiceMetric  # noqa: F401
from medicai.metrics.dice import CategoricalDiceMetric  # noqa: F401
from medicai.metrics.dice import SparseDiceMetric  # noqa: F401


def create_sensitive_test_data(seed=42):
    np.random.seed(seed)

    # Tiny test volume (1x2x2x2)
    y_true = np.zeros((1, 2, 2, 2, 1), dtype=np.int32)

    # Single voxel of class 1 (others remain background)
    y_true[0, 0, 0, 0, 0] = 1

    # Predictions with extreme randomness
    y_pred = np.zeros((1, 2, 2, 2, 3), dtype=np.float32)

    # Correct prediction for class 1 voxel
    y_pred[0, 0, 0, 0, 1] = 1.0

    # Make background predictions vary WILDLY
    background_mask = (y_true == 0).squeeze(-1)
    y_pred[background_mask] = np.random.dirichlet([0.01, 0.01, 0.01], size=background_mask.sum())

    return y_true.astype("float32"), y_pred.astype("float32")


def test_categorical_dice_metric():
    y_true, y_pred = create_sensitive_test_data(seed=50)

    dice_metric_mean = CategoricalDiceMetric(
        from_logits=1,
        ignore_empty=0,
        num_classes=y_pred.shape[-1],
        name="dice_score",
    )
    y_true_one_hot = ops.one_hot(ops.squeeze(y_true, axis=-1), num_classes=y_pred.shape[-1])
    dice_metric_mean.update_state(y_true_one_hot, y_pred)
    score = dice_metric_mean.result().numpy()
    dice_metric_mean.reset_states()
    np.testing.assert_allclose(
        score, np.array([0.40909126]), rtol=1e-5, err_msg="Categorical Dice Metric test failed."
    )

    # excluding channel index 0
    dice_metric_mean = CategoricalDiceMetric(
        from_logits=1,
        ignore_empty=0,
        class_ids=[1, 2],
        num_classes=y_pred.shape[-1],
        name="dice_score",
    )
    dice_metric_mean.update_state(y_true_one_hot, y_pred)
    score = dice_metric_mean.result().numpy()
    dice_metric_mean.reset_states()
    np.testing.assert_allclose(
        score, np.array([0.25000057]), rtol=1e-5, err_msg="Categorical Dice Metric test failed."
    )

    # excluding channel index 0
    dice_metric_mean = CategoricalDiceMetric(
        from_logits=1,
        ignore_empty=0,
        class_ids=[1, 2],
        num_classes=y_pred.shape[-1],
        name="dice_score",
    )
    dice_metric_mean.update_state(y_true_one_hot, y_pred)
    score = dice_metric_mean.result().numpy()
    dice_metric_mean.reset_states()
    np.testing.assert_allclose(
        score, np.array([0.25000057]), rtol=1e-5, err_msg="Categorical Dice Metric test failed."
    )

    # ignore empty gt
    dice_metric_mean = CategoricalDiceMetric(
        from_logits=1,
        ignore_empty=1,
        num_classes=y_pred.shape[-1],
        name="dice_score",
    )
    dice_metric_mean.update_state(y_true_one_hot, y_pred)
    score = dice_metric_mean.result().numpy()
    dice_metric_mean.reset_states()
    np.testing.assert_allclose(
        score, np.array([0.61363643]), rtol=1e-5, err_msg="Categorical Dice Metric test failed."
    )


def test_sparse_categorical_dice_metric():
    y_true, y_pred = create_sensitive_test_data(seed=50)

    # ignore empty gt + exclude channel index 0
    dice_metric_mean = SparseDiceMetric(
        from_logits=1,
        ignore_empty=1,
        class_ids=[1, 2],
        num_classes=y_pred.shape[-1],
        name="dice_score",
    )
    dice_metric_mean.update_state(y_true, y_pred)
    score = dice_metric_mean.result().numpy()
    dice_metric_mean.reset_states()
    np.testing.assert_allclose(
        score, np.array([0.5000001]), rtol=1e-5, err_msg="Categorical Dice Metric test failed."
    )


def test_binary_dice_metric():

    batch_size, depth, height, width, num_labels = 2, 5, 64, 64, 3

    y_true = keras.random.randint(
        shape=(batch_size, depth, height, width, num_labels), minval=0, maxval=2, dtype="int32"
    )
    y_true = ops.cast(y_true, 'float32')
    y_pred_logit = keras.random.normal(
        shape=(batch_size, depth, height, width, num_labels), dtype="float32"
    )

    dice_metric_from_logits = BinaryDiceMetric(
        from_logits=1, num_classes=num_labels, ignore_empty=False, class_ids=[0]
    )
    dice_metric_from_logits.update_state(y_true, y_pred_logit)
    score = dice_metric_from_logits.result().numpy()
    dice_metric_from_logits.reset_states()
    assert score.shape == (), "Score should be a scalar."

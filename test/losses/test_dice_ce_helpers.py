import numpy as np
import pytest
from keras import ops

from medicai.losses.dice_ce import apply_reduction, binary_cross_entropy, cross_entropy


@pytest.mark.unit
@pytest.mark.parametrize(
    "reduction,expected",
    [
        ("none", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
        ("sum", 10.0),
        ("mean", 2.5),
        ("sum_over_batch_size", 5.0),
    ],
)
def test_apply_reduction_supported_modes(reduction, expected):
    loss = ops.convert_to_tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    reduced = apply_reduction(loss, reduction)
    np.testing.assert_allclose(ops.convert_to_numpy(reduced), expected, rtol=1e-6)


@pytest.mark.unit
def test_apply_reduction_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unsupported reduction type"):
        apply_reduction(ops.convert_to_tensor(np.array([[1.0]], dtype=np.float32)), "median")


@pytest.mark.unit
def test_cross_entropy_with_mask_only_counts_valid_positions():
    y_true = ops.convert_to_tensor(np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32))
    y_pred = ops.convert_to_tensor(np.array([[[[0.9, 0.1], [0.2, 0.8]]]], dtype=np.float32))
    # Mask out second position in both channels
    mask = ops.convert_to_tensor(np.array([[[[1.0, 1.0], [0.0, 0.0]]]], dtype=np.float32))

    masked = cross_entropy(y_true, y_pred, mask=mask)
    unmasked = cross_entropy(y_true, y_pred, mask=None)

    # Masked loss should differ because one full spatial site is excluded.
    assert masked.shape == (1, 2)
    assert not np.allclose(ops.convert_to_numpy(masked), ops.convert_to_numpy(unmasked))


@pytest.mark.unit
def test_binary_cross_entropy_is_finite_and_mask_aware():
    y_true = ops.convert_to_tensor(np.array([[[[1.0], [0.0]]]], dtype=np.float32))
    y_pred = ops.convert_to_tensor(np.array([[[[0.8], [0.2]]]], dtype=np.float32))
    mask = ops.convert_to_tensor(np.array([[[[1.0], [0.0]]]], dtype=np.float32))

    masked = binary_cross_entropy(y_true, y_pred, mask=mask)
    unmasked = binary_cross_entropy(y_true, y_pred, mask=None)

    assert masked.shape == (1, 1)
    assert np.isfinite(ops.convert_to_numpy(masked)).all()
    assert not np.allclose(ops.convert_to_numpy(masked), ops.convert_to_numpy(unmasked))

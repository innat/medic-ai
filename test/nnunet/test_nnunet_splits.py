import pytest

from medicai.trainer.nnunet.data.cross_validation import generate_splits


def test_training_cases_generate_reproducible_cross_validation_folds():
    case_ids = [f"case_{index}.npz" for index in range(10)]

    first = generate_splits(case_ids, n_folds=5, seed=17)
    second = generate_splits(case_ids, n_folds=5, seed=17)

    assert first == second
    assert len(first) == 5
    all_ids = {f"case_{index}" for index in range(10)}
    for fold in first:
        assert set(fold["train"]).isdisjoint(fold["val"])
        assert set(fold["train"]) | set(fold["val"]) == all_ids


def test_cross_validation_requires_at_least_two_cases_and_folds():
    with pytest.raises(ValueError, match="at least 2"):
        generate_splits(["only_case"], n_folds=2)

    with pytest.raises(ValueError, match="exceeds number of available cases"):
        generate_splits(["case_1", "case_2"], n_folds=3)

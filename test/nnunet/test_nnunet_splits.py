import numpy as np
import pytest

from medicai.trainer.nnunet.data.splits.cross_validation import (
    generate_custom_splits,
    generate_splits,
    validate_splits,
)


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


def test_default_split_matches_seeded_sklearn_kfold_partition_algorithm():
    case_ids = [f"case_{index}" for index in range(8)]
    indices = np.arange(len(case_ids))
    np.random.RandomState(17).shuffle(indices)
    fold_sizes = np.full(3, len(case_ids) // 3, dtype=int)
    fold_sizes[: len(case_ids) % 3] += 1
    expected = []
    current = 0
    for fold_size in fold_sizes:
        val_indices = indices[current : current + fold_size]
        train_indices = np.setdiff1d(np.arange(len(case_ids)), val_indices)
        expected.append(
            {
                "train": [case_ids[index] for index in train_indices],
                "val": [case_ids[index] for index in val_indices],
            }
        )
        current += fold_size

    assert generate_splits(case_ids, n_folds=3, seed=17) == expected


def test_cross_validation_requires_at_least_two_cases_and_folds():
    with pytest.raises(ValueError, match="at least 2"):
        generate_splits(["only_case"], n_folds=2)

    with pytest.raises(ValueError, match="exceeds number of available cases"):
        generate_splits(["case_1", "case_2"], n_folds=3)


def test_custom_group_splitter_keeps_groups_out_of_both_partitions():
    class PairGroupSplitter:
        def split(self, case_ids, y=None, groups=None):
            assert len(case_ids) == len(groups) == 6
            for group_index in range(3):
                validation = [i for i, group in enumerate(groups) if group == group_index]
                training = [i for i in range(len(case_ids)) if i not in validation]
                yield training, validation

    case_ids = [f"case_{index}" for index in range(6)]
    groups = {case_id: index // 2 for index, case_id in enumerate(case_ids)}
    splits = generate_custom_splits(case_ids, PairGroupSplitter(), groups=groups)

    assert len(splits) == 3
    for split in splits:
        train_groups = {groups[case_id] for case_id in split["train"]}
        val_groups = {groups[case_id] for case_id in split["val"]}
        assert train_groups.isdisjoint(val_groups)


def test_explicit_splits_validate_group_leakage():
    splits = [
        {"train": ["case_b", "case_c"], "val": ["case_a"]},
        {"train": ["case_a", "case_c"], "val": ["case_b"]},
        {"train": ["case_a", "case_b"], "val": ["case_c"]},
    ]
    groups = {"case_a": "patient_1", "case_b": "patient_1", "case_c": "patient_2"}

    with pytest.raises(ValueError, match="leaks group"):
        validate_splits(splits, list(groups), groups=groups)


def test_explicit_splits_reject_unknown_cases():
    splits = [
        {"train": ["case_a", "case_c"], "val": ["case_b"]},
        {"train": ["case_a", "case_b"], "val": ["case_c"]},
    ]

    with pytest.raises(ValueError, match="unknown case IDs"):
        validate_splits(splits, ["case_a", "case_b"])

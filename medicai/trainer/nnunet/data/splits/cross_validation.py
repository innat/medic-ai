import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class CrossValidationConfig:
    """Describe how nnU-Net cases are partitioned into training folds.

    Provide either ``n_folds`` for generated K-fold splits, explicit ``splits``,
    or a scikit-learn-style ``splitter``. ``groups`` is passed to compatible
    splitters and can also validate group separation in explicit splits.

    Args:
        n_folds: Number of generated folds. Defaults to five when no explicit
            split strategy is supplied; inferred from ``splits`` or
            ``splitter`` otherwise.
        splits: Explicit fold dictionaries with ``train`` and ``val`` case IDs.
        splitter: Object exposing ``split(case_ids, groups=...)``.
        groups: Mapping from each case ID to its group identifier.
        seed: Seed for deterministic generated K-fold assignments.

    Raises:
        ValueError: If incompatible split strategies or invalid fold counts are
            supplied.
    """

    n_folds: int | None = None
    splits: Sequence[Mapping[str, Sequence[str]]] | None = None
    splitter: Any | None = None
    groups: Mapping[str, Any] | None = None
    seed: int = 12345

    def __post_init__(self) -> None:
        if self.n_folds is not None and (
            isinstance(self.n_folds, bool)
            or not isinstance(self.n_folds, int)
            or self.n_folds < 2
        ):
            raise ValueError("n_folds must be an integer of at least 2.")
        if self.splits is not None and self.splitter is not None:
            raise ValueError("Pass either explicit splits or a splitter, not both.")
        if self.groups is not None and self.splits is None and self.splitter is None:
            raise ValueError("groups require explicit splits or a splitter.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")
        if self.splits is not None and self.n_folds is not None:
            if len(self.splits) != self.n_folds:
                raise ValueError("n_folds must match the number of explicit splits.")


def normalize_case_id(case_id):
    """Normalize identifiers by taking the stem of the file path.
    The manifest handles all modality mappings, so we no longer need to
    manually strip modality suffixes.
    """
    normalized = Path(case_id).name
    for suffix in (".nii.gz", ".nii", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"):
        if normalized.lower().endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return re.sub(r"_\d{4}$", "", normalized)


# Split generation


def generate_splits(
    case_ids,
    n_folds=5,
    seed=12345,
):
    """
    Generate deterministic K-fold splits matching nnU-Net's default splitter.

    Parameters
    ----------
    case_ids : list of case identifier strings
    n_folds  : number of folds (default 5, matching original nnU-Net)
    seed     : random seed for reproducibility

    Returns
    -------
    List of dicts, each with keys 'train' and 'val'.
    Length = n_folds.

    Example
    -------
    ::

        splits = generate_splits(["case_001", "case_002", ...])
        train_ids = splits[0]["train"]
        val_ids   = splits[0]["val"]
    """
    case_ids = sorted({normalize_case_id(case_id) for case_id in case_ids})
    if not case_ids:
        raise ValueError("Cannot generate splits for an empty case list.")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if n_folds > len(case_ids):
        raise ValueError(f"n_folds={n_folds} exceeds number of available cases ({len(case_ids)}).")

    indices = np.arange(len(case_ids))
    np.random.RandomState(seed).shuffle(indices)
    fold_sizes = np.full(n_folds, len(case_ids) // n_folds, dtype=int)
    fold_sizes[: len(case_ids) % n_folds] += 1
    splits = []
    current = 0
    for fold_size in fold_sizes:
        val_indices = indices[current : current + fold_size]
        train_indices = np.setdiff1d(np.arange(len(case_ids)), val_indices)
        splits.append(
            {
                "train": [case_ids[index] for index in train_indices],
                "val": [case_ids[index] for index in val_indices],
            }
        )
        current += fold_size

    return splits


def validate_splits(splits, case_ids, groups=None):
    """Validate and normalize complete K-fold assignments by case ID."""
    case_ids = [normalize_case_id(case_id) for case_id in case_ids]
    expected = set(case_ids)
    if not expected or len(expected) != len(case_ids):
        raise ValueError("Case identifiers must be non-empty and unique after normalization.")
    if not isinstance(splits, (list, tuple)) or len(splits) < 2:
        raise ValueError("Custom cross-validation splits must contain at least two folds.")

    normalized = []
    validation_counts = {case_id: 0 for case_id in expected}
    for fold_index, fold in enumerate(splits):
        if not isinstance(fold, dict) or not {"train", "val"} <= set(fold):
            raise ValueError(f"Fold {fold_index} must contain 'train' and 'val' case IDs.")
        train_ids = [normalize_case_id(case_id) for case_id in fold["train"]]
        val_ids = [normalize_case_id(case_id) for case_id in fold["val"]]
        train, val = set(train_ids), set(val_ids)
        unknown = (train | val) - expected
        if unknown:
            raise ValueError(f"Fold {fold_index} contains unknown case IDs: {sorted(unknown)}.")
        if len(train) != len(train_ids) or len(val) != len(val_ids):
            raise ValueError(f"Fold {fold_index} contains duplicate case IDs.")
        if train & val:
            raise ValueError(f"Fold {fold_index} has cases in both train and validation.")
        if train | val != expected or not val:
            raise ValueError(
                f"Fold {fold_index} must partition all cases into non-empty validation data."
            )
        for case_id in val:
            validation_counts[case_id] += 1
        normalized.append({"train": sorted(train), "val": sorted(val)})

    invalid_coverage = sorted(case_id for case_id, count in validation_counts.items() if count != 1)
    if invalid_coverage:
        raise ValueError(
            "K-fold validation partitions must validate each case exactly once; "
            f"invalid coverage for {invalid_coverage}."
        )
    if groups is not None:
        normalized_groups = {normalize_case_id(case_id): value for case_id, value in groups.items()}
        if len(normalized_groups) != len(groups) or set(normalized_groups) != expected:
            raise ValueError("groups must map every case ID exactly once.")
        try:
            for fold_index, fold in enumerate(normalized):
                train_groups = {normalized_groups[case_id] for case_id in fold["train"]}
                val_groups = {normalized_groups[case_id] for case_id in fold["val"]}
                leaked_groups = train_groups & val_groups
                if leaked_groups:
                    raise ValueError(
                        f"Fold {fold_index} leaks group(s) across train and validation: "
                        f"{sorted(leaked_groups, key=str)}."
                    )
        except TypeError as exc:
            raise ValueError("Group identifiers must be hashable scalar values.") from exc
    return normalized


def generate_custom_splits(case_ids, splitter, groups=None):
    """Generate case-ID folds from a scikit-learn-style splitter object."""
    case_ids = sorted({normalize_case_id(case_id) for case_id in case_ids})
    if len(case_ids) < 2:
        raise ValueError("Custom cross-validation requires at least two cases.")
    if not callable(getattr(splitter, "split", None)):
        raise TypeError("splitter must provide a callable split(X, y=None, groups=None) method.")

    group_values = None
    if groups is not None:
        normalized_groups = {normalize_case_id(case_id): value for case_id, value in groups.items()}
        if len(normalized_groups) != len(groups):
            raise ValueError("groups contains duplicate case IDs after normalization.")
        missing = sorted(set(case_ids) - set(normalized_groups))
        extra = sorted(set(normalized_groups) - set(case_ids))
        if missing or extra:
            raise ValueError(
                f"groups must map every case ID exactly once; missing={missing}, extra={extra}."
            )
        group_values = [normalized_groups[case_id] for case_id in case_ids]

    try:
        index_splits = splitter.split(case_ids, groups=group_values)
        splits = []
        for train_indices, val_indices in index_splits:
            train_indices = [int(index) for index in train_indices]
            val_indices = [int(index) for index in val_indices]
            if any(index < 0 or index >= len(case_ids) for index in train_indices + val_indices):
                raise ValueError("splitter returned an out-of-range case index.")
            splits.append(
                {
                    "train": [case_ids[index] for index in train_indices],
                    "val": [case_ids[index] for index in val_indices],
                }
            )
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Unable to generate cross-validation splits: {exc}") from exc
    return validate_splits(splits, case_ids, groups=groups)


def save_splits(splits, path):
    """Save splits to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path):
    """Load splits from a JSON file."""
    with open(path) as f:
        return json.load(f)


# Result aggregation


def aggregate_fold_results(
    fold_results,
):
    """
    Aggregate per-fold metrics into mean ± std.

    Parameters
    ----------
    fold_results : list of dicts, one per fold.
                   Each dict maps metric name to scalar value.

    Returns
    -------
    Dict mapping metric name to summary values:
      {
        "mean_dice":     0.85,
        "std_dice":      0.03,
        "mean_dice_fold_0": 0.83,
        ...
      }

    Example
    -------
    ::

        results = aggregate_fold_results([
            {"mean_dice": 0.83, "val_loss": 0.25},
            {"mean_dice": 0.85, "val_loss": 0.22},
        ])
        # → {"mean_dice": 0.84, "std_dice": 0.01, ...}
    """
    if not fold_results:
        return {}

    all_keys = set()
    for d in fold_results:
        all_keys.update(d.keys())

    summary = {}

    for key in sorted(all_keys):
        values = [d[key] for d in fold_results if key in d]
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        summary[f"mean_{key}"] = float(arr.mean())
        summary[f"std_{key}"] = float(arr.std())
        for fold_idx, v in enumerate(values):
            summary[f"{key}_fold_{fold_idx}"] = float(v)

    return summary


def print_fold_summary(fold_results):
    """Print a human-readable summary of cross-validation results."""
    summary = aggregate_fold_results(fold_results)
    print("\n" + "=" * 60)
    print("Cross-Validation Results")
    print("=" * 60)

    # Per-fold
    for i, result in enumerate(fold_results):
        dice = result.get("mean_dice", float("nan"))
        print(f"  Fold {i}: mean Dice = {dice:.4f}")

    print("-" * 60)
    mean_dice = summary.get("mean_mean_dice", float("nan"))
    std_dice = summary.get("std_mean_dice", float("nan"))
    print(f"  Overall: {mean_dice:.4f} ± {std_dice:.4f}")
    print("=" * 60 + "\n")

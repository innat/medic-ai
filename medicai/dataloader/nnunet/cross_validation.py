import json
import re
from pathlib import Path

import numpy as np


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
    Generate stratified k-fold splits.

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

    rng = np.random.default_rng(seed)
    n = len(case_ids)

    # Shuffle once with fixed seed
    indices = rng.permutation(n).tolist()
    shuffled = [case_ids[i] for i in indices]

    # Partition into n_folds equal-ish subsets
    folds_indices = [shuffled[i::n_folds] for i in range(n_folds)]

    splits = []
    for val_fold_idx in range(n_folds):
        val_ids = folds_indices[val_fold_idx]
        train_ids = []
        for i, fold in enumerate(folds_indices):
            if i != val_fold_idx:
                train_ids.extend(fold)
        splits.append({"train": sorted(train_ids), "val": sorted(val_ids)})

    return splits


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

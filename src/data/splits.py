"""Stratified K-Fold cross-validation split utilities."""

import logging
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def get_cv_splits(
    labels: List[int],
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate stratified K-fold cross-validation splits.

    Args:
        labels: Integer class labels for each sample.
        n_splits: Number of folds.
        seed: Random state for reproducibility.

    Returns:
        List of (train_indices, val_indices) tuples, one per fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(labels))
    labels_arr = np.array(labels)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels_arr)):
        splits.append((train_idx, val_idx))
        _log_fold_distribution(fold, labels_arr, train_idx, val_idx)

    return splits


def _log_fold_distribution(
    fold: int,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> None:
    """Log class distribution per fold.

    Args:
        fold: Fold index (0-based).
        labels: Full label array.
        train_idx: Indices assigned to the training split.
        val_idx: Indices assigned to the validation split.
    """
    unique_classes = np.unique(labels)
    train_counts = {int(c): int(np.sum(labels[train_idx] == c)) for c in unique_classes}
    val_counts = {int(c): int(np.sum(labels[val_idx] == c)) for c in unique_classes}

    logger.info(
        f"Fold {fold} — "
        f"train: {len(train_idx)} samples {train_counts} | "
        f"val: {len(val_idx)} samples {val_counts}"
    )

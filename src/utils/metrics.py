"""Centralised metric computation for the Ulcer Classification Pipeline."""

import logging
from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all evaluation metrics from ground-truth labels and predicted probabilities.

    Args:
        y_true: Ground-truth integer labels of shape (N,).
        y_prob: Predicted probabilities for class 1 (NON-VASCULAR) of shape (N,).
            VASCULAR (label=0) is the positive class; y_prob = probs[:, 1].
        threshold: Decision threshold for converting probabilities to labels.

    Returns:
        Dictionary mapping metric name to scalar value.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # AUC-ROC
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError as exc:
        logger.warning(f"Could not compute AUC-ROC: {exc}")
        auc = float("nan")

    # F1 (macro), accuracy
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    # Sensitivity & specificity via confusion matrix.
    # VASCULAR (label=0) is the positive class.
    # cm.ravel() with labels=[0,1] yields [cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
    # stored in tn, fp, fn, tp (sklearn naming assumes class-1 positive — reversed here).
    # Mapping:  tn=TP_vasc, fp=FN_vasc, fn=FP_vasc, tp=TN_vasc
    # sensitivity = VASCULAR recall = TP_vasc / (TP_vasc + FN_vasc) = tn / (tn + fp)
    # specificity = NON-VASCULAR recall = TN_vasc / (TN_vasc + FP_vasc) = tp / (tp + fn)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        specificity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    else:
        sensitivity = float("nan")
        specificity = float("nan")

    return {
        "auc_roc": auc,
        "f1": f1,
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute a bootstrap confidence interval for a given metric.

    Args:
        y_true: Ground-truth integer labels of shape (N,).
        y_prob: Predicted probabilities of shape (N,).
        metric_fn: Function accepting (y_true, y_prob) and returning a scalar.
        n_iterations: Number of bootstrap resampling iterations.
        confidence_level: Desired confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for the bootstrap RNG.

    Returns:
        Tuple (lower_bound, upper_bound) of the confidence interval.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores: list[float] = []

    for _ in range(n_iterations):
        indices = rng.choice(n, n, replace=True)
        try:
            score = metric_fn(y_true[indices], y_prob[indices])
            scores.append(float(score))
        except (ValueError, ZeroDivisionError):
            pass

    if not scores:
        return float("nan"), float("nan")

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return lower, upper

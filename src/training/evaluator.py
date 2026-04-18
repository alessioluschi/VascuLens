"""Metrics computation, confidence intervals, and plot generation."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.utils.metrics import bootstrap_confidence_interval, compute_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """Compute metrics, bootstrap CIs, and save diagnostic plots.

    Args:
        cfg: Full OmegaConf configuration.
        output_dir: Directory where plots and result files are saved.
    """

    def __init__(self, cfg, output_dir: str) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Per-fold evaluation                                                  #
    # ------------------------------------------------------------------ #

    def evaluate_fold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        fold: int,
        tag: str = "",
    ) -> Dict[str, float]:
        """Compute metrics for a single fold.

        Args:
            y_true: Ground-truth labels (N,).
            y_prob: Predicted probabilities for class 1 (N,).
            fold: Fold index (used for plot naming).
            tag: Optional string appended to log messages.

        Returns:
            Dictionary of metric values with bootstrap CIs.
        """
        metrics = compute_metrics(y_true, y_prob)

        ecfg = self.cfg.evaluation

        # Bootstrap CI
        if ecfg.bootstrap_ci.enabled:
            metrics.update(
                self._compute_bootstrap_cis(
                    y_true,
                    y_prob,
                    n_iter=ecfg.bootstrap_ci.n_iterations,
                    ci_level=ecfg.bootstrap_ci.confidence_level,
                )
            )

        label_str = f"[fold {fold}{' | ' + tag if tag else ''}]"
        logger.info(
            f"{label_str} "
            + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if not k.endswith("_ci"))
        )

        # Confusion matrix plot
        if ecfg.save_confusion_matrix:
            self._save_confusion_matrix(y_true, y_prob, fold=fold, tag=tag)

        return metrics

    # ------------------------------------------------------------------ #
    # Aggregate over folds                                                 #
    # ------------------------------------------------------------------ #

    def aggregate_and_save(
        self,
        per_fold_metrics: List[Dict[str, float]],
        backbone_name: str,
    ) -> Dict[str, float]:
        """Aggregate fold metrics and save results.

        Args:
            per_fold_metrics: List of metric dicts, one per fold.
            backbone_name: Used in output filenames.

        Returns:
            Dictionary of mean ± std for each metric.
        """
        keys = [k for k in per_fold_metrics[0] if not k.endswith("_ci")]
        aggregated: Dict[str, float] = {}

        for k in keys:
            vals = [m[k] for m in per_fold_metrics if not np.isnan(m[k])]
            aggregated[f"{k}_mean"] = float(np.mean(vals))
            aggregated[f"{k}_std"] = float(np.std(vals))

        # Save JSON
        out_file = self.output_dir / f"{backbone_name}_aggregated.json"
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(aggregated, fh, indent=2)
        logger.info(f"Aggregated metrics saved → {out_file}")

        return aggregated

    def save_roc_curve(
        self,
        fold_results: List[Dict],
        tag: str = "ensemble",
    ) -> None:
        """Save mean ± std ROC curve across folds.

        Args:
            fold_results: List of dicts each containing ``"y_true"`` and ``"y_prob"``.
            tag: Label used in filename and title.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        tprs: List[np.ndarray] = []
        base_fpr = np.linspace(0, 1, 101)
        aucs: List[float] = []

        for fold_data in fold_results:
            y_true = fold_data["y_true"]
            y_prob = fold_data["y_prob"]
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                tpr_interp = np.interp(base_fpr, fpr, tpr)
                tpr_interp[0] = 0.0
                tprs.append(tpr_interp)
                aucs.append(roc_auc_score(y_true, y_prob))
                ax.plot(base_fpr, tpr_interp, alpha=0.3, color="steelblue", linewidth=1)
            except Exception as exc:
                logger.warning(f"Skipping fold ROC: {exc}")

        if tprs:
            mean_tpr = np.mean(tprs, axis=0)
            std_tpr = np.std(tprs, axis=0)
            mean_auc = float(np.mean(aucs))
            ax.plot(base_fpr, mean_tpr, "b-", linewidth=2, label=f"Mean ROC (AUC={mean_auc:.3f})")
            ax.fill_between(
                base_fpr,
                mean_tpr - std_tpr,
                mean_tpr + std_tpr,
                alpha=0.2,
                color="steelblue",
                label="±1 std",
            )

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {tag}")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

        roc_file = self.output_dir / f"roc_curve_{tag}.png"
        fig.savefig(roc_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"ROC curve saved → {roc_file}")

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _compute_bootstrap_cis(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_iter: int,
        ci_level: float,
    ) -> Dict[str, tuple]:
        """Return bootstrap CIs for AUC-ROC, F1, accuracy, sensitivity, specificity."""
        from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

        ci_results: Dict[str, tuple] = {}

        metric_fns = {
            "auc_roc": lambda yt, yp: roc_auc_score(yt, yp),
            "f1": lambda yt, yp: f1_score(
                yt, (yp >= 0.5).astype(int), average="macro", zero_division=0
            ),
            "accuracy": lambda yt, yp: accuracy_score(yt, (yp >= 0.5).astype(int)),
        }

        for name, fn in metric_fns.items():
            lo, hi = bootstrap_confidence_interval(
                y_true, y_prob, fn, n_iterations=n_iter, confidence_level=ci_level
            )
            ci_results[f"{name}_ci"] = (lo, hi)
            logger.debug(f"  {name} CI [{ci_level*100:.0f}%]: [{lo:.4f}, {hi:.4f}]")

        return ci_results

    def _save_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        fold: Optional[int] = None,
        tag: str = "",
    ) -> None:
        """Save a seaborn confusion matrix heatmap.

        Args:
            y_true: Ground-truth labels.
            y_prob: Predicted probabilities for class 1.
            fold: Fold index for filename.
            tag: Optional tag for the title.
        """
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        class_names = list(self.cfg.data.class_names)

        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        title = f"Confusion Matrix"
        if fold is not None:
            title += f" — fold {fold}"
        if tag:
            title += f" ({tag})"
        ax.set_title(title)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

        suffix = f"fold_{fold}" if fold is not None else "aggregated"
        if tag:
            suffix += f"_{tag}"
        fname = self.output_dir / f"confusion_matrix_{suffix}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Confusion matrix saved → {fname}")

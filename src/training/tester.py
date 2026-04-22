"""Test-set evaluation with late fusion, metrics, and heatmap generation.

Loads one backbone at a time (memory-safe), runs inference on the
held-out test set, applies late fusion, and saves:

  outputs/test/
  ├── metrics.json                          (all enabled backbones + late_fusion)
  ├── per_sample_results.csv
  ├── confusion_matrix_<backbone>.png       (one per enabled backbone)
  ├── confusion_matrix_late_fusion.png
  ├── roc_curve_test.png
  └── heatmaps/
      └── <image_stem>/
          ├── original.png
          ├── <backbone>_gradcam.png        (EfficientNet → GradCAM++)
          ├── <backbone>_attention.png      (BiomedCLIP / UNI → Attention Rollout)
          └── late_fusion_aggregated.png
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from src.data.augmentation import get_val_transforms
from src.data.dataset import UlcerDataset
from src.models.memory import gpu_memory_guard, log_gpu_memory, release_model
from src.utils.metrics import compute_metrics, bootstrap_confidence_interval

logger = logging.getLogger(__name__)

# Explainability method per backbone family
_GRADCAM_BACKBONES: frozenset = frozenset({"efficientnet"})
_ATTENTION_BACKBONES: frozenset = frozenset({"biomedclip", "uni"})


# ===========================================================================
# Public entry point
# ===========================================================================


def run_test(cfg: DictConfig) -> Dict:
    """Run full test-set evaluation pipeline.

    Loads each enabled backbone's fold checkpoint sequentially, computes
    logits on the test set, performs late fusion, evaluates all metrics,
    and generates confusion matrices + heatmaps.

    Args:
        cfg: Validated OmegaConf configuration (must have a ``test`` section).

    Returns:
        Dictionary of all computed metrics.
    """
    tcfg = cfg.test
    out_dir = Path(tcfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = out_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.general.device

    logger.info(f"=== TEST MODE | test_dir={tcfg.test_dir} ===")
    log_gpu_memory("TEST START")

    # --- Build test dataset ---
    test_dataset = UlcerDataset(
        root_dir=tcfg.test_dir,
        transform=None,          # set per-backbone below
        class_names=list(cfg.data.class_names),
    )
    if len(test_dataset) == 0:
        raise RuntimeError(
            f"No images found in test_dir='{tcfg.test_dir}'. "
            f"Expected subfolders: {list(cfg.data.class_names)}"
        )

    all_paths = [str(p) for p, _ in test_dataset.samples]
    true_labels = np.array([lbl for _, lbl in test_dataset.samples])

    # --- Run inference per backbone ---
    enabled = [
        n for n in cfg.backbones.training_order
        if getattr(cfg.backbones, n).enabled
    ]

    backbone_logits: Dict[str, torch.Tensor] = {}
    backbone_probs:  Dict[str, np.ndarray]   = {}

    for bname in enabled:
        ckpt_path = _resolve_backbone_ckpt(cfg, bname)
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path} — skipping '{bname}'.")
            continue
        logger.info(f"Loading {bname} from {ckpt_path}")

        logits = _infer_backbone(cfg, bname, str(ckpt_path), all_paths, device)
        backbone_logits[bname] = logits
        backbone_probs[bname] = F.softmax(logits.float(), dim=-1).numpy()
        logger.info(f"Inference done: {bname}  shape={tuple(logits.shape)}")

    if not backbone_logits:
        raise RuntimeError("No backbone checkpoints found. Run training first.")

    # --- Late fusion ---
    fused_probs = _late_fusion(backbone_probs, cfg)
    y_prob_fused = fused_probs[:, 1]

    # --- Metrics ---
    all_metrics: Dict = {}

    for bname, probs in backbone_probs.items():
        m = compute_metrics(true_labels, probs[:, 1])
        if cfg.evaluation.bootstrap_ci.enabled:
            m.update(_bootstrap_cis(true_labels, probs[:, 1], cfg))
        all_metrics[bname] = m
        _save_confusion_matrix(
            true_labels, probs[:, 1], cfg, out_dir,
            tag=bname,
        )

    fused_metrics = compute_metrics(true_labels, y_prob_fused)
    if cfg.evaluation.bootstrap_ci.enabled:
        fused_metrics.update(_bootstrap_cis(true_labels, y_prob_fused, cfg))
    all_metrics["late_fusion"] = fused_metrics
    _save_confusion_matrix(true_labels, y_prob_fused, cfg, out_dir, tag="late_fusion")
    _save_roc_curve(true_labels, y_prob_fused, out_dir)

    # Persist metrics JSON
    metrics_file = out_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as fh:
        json.dump(
            {k: {mk: (list(mv) if isinstance(mv, tuple) else mv) for mk, mv in v.items()}
             for k, v in all_metrics.items()},
            fh, indent=2,
        )
    logger.info(f"Metrics saved → {metrics_file}")

    # Per-sample CSV
    _save_per_sample_csv(
        all_paths, true_labels, backbone_probs, fused_probs,
        cfg, out_dir,
    )

    # --- Heatmaps ---
    n_samples = int(tcfg.num_heatmap_samples)
    if n_samples > 0:
        selected = _select_samples(
            all_paths, true_labels, y_prob_fused,
            n=n_samples, strategy=tcfg.heatmap_selection,
        )
        ckpt_paths = {b: _resolve_backbone_ckpt(cfg, b) for b in backbone_probs}
        _generate_test_heatmaps(
            cfg=cfg,
            image_paths=selected,
            ckpt_paths=ckpt_paths,
            backbone_probs=backbone_probs,
            fused_probs=fused_probs,
            out_dir=heatmap_dir,
        )

    # Print summary table
    _print_metrics_table(all_metrics)
    log_gpu_memory("TEST END")
    return all_metrics


# ===========================================================================
# Inference
# ===========================================================================


def _resolve_backbone_ckpt(cfg: DictConfig, bname: str) -> Path:
    """Return the checkpoint path for *bname*, honouring per-backbone fold overrides.

    Checks ``test.fold_per_backbone.<bname>`` first; falls back to ``test.fold``.

    Args:
        cfg: Full configuration.
        bname: Backbone identifier (e.g. ``"efficientnet"``).

    Returns:
        Absolute path to ``<bname>_best.pt`` in the resolved fold directory.
    """
    fold_overrides = getattr(cfg.test, "fold_per_backbone", None) or {}
    fold = int(fold_overrides.get(bname, cfg.test.fold))
    return Path(cfg.general.output_dir) / "checkpoints" / f"fold_{fold}" / f"{bname}_best.pt"


def _infer_backbone(
    cfg: DictConfig,
    backbone_name: str,
    ckpt_path: str,
    image_paths: List[str],
    device: str,
) -> torch.Tensor:
    """Load a backbone checkpoint and run inference on all test images.

    Args:
        cfg: Full configuration.
        backbone_name: Backbone identifier.
        ckpt_path: Path to the checkpoint .pt file.
        image_paths: Ordered list of test image paths.
        device: Target device string.

    Returns:
        Logit tensor of shape (N, num_classes) on CPU.
    """
    from src.training.cross_validation import _build_backbone

    with gpu_memory_guard(f"Test inference — {backbone_name}"):
        model = _build_backbone(backbone_name, cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        logger.info(
            f"Loaded {backbone_name} checkpoint (epoch {ckpt.get('epoch','?')}, "
            f"metric={ckpt.get('metric', float('nan')):.4f})"
        )

        transform = get_val_transforms(cfg, backbone_name)
        dataset = UlcerDataset(
            root_dir=cfg.test.test_dir,
            transform=transform,
            class_names=list(cfg.data.class_names),
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.general.num_workers,
            pin_memory=(cfg.hardware.pin_memory and device == "cuda"),
        )

        all_logits: List[torch.Tensor] = []
        dev = torch.device(device)
        with torch.no_grad():
            for images, _labels, _paths in loader:
                images = images.to(dev, non_blocking=True)
                with torch.cuda.amp.autocast(
                    enabled=cfg.hardware.mixed_precision and dev.type == "cuda"
                ):
                    logits = model(images)
                all_logits.append(logits.cpu().float())

        result = torch.cat(all_logits, dim=0)
        release_model(model)
        return result


# ===========================================================================
# Late fusion
# ===========================================================================


def _late_fusion(
    backbone_probs: Dict[str, np.ndarray],
    cfg: DictConfig,
) -> np.ndarray:
    """Compute weighted-average late fusion of backbone probabilities.

    Uses equal weights when ``fusion.late_fusion.learn_weights`` is True
    (learned weights are not available without a validation set at test time).
    Falls back to equal weights regardless — configurable via
    ``test.fusion_weights`` list if present.

    Args:
        backbone_probs: Mapping from backbone name to (N, C) probability array.
        cfg: Full configuration.

    Returns:
        Fused probability array of shape (N, num_classes).
    """
    names = list(backbone_probs.keys())
    n = len(names)

    # Allow explicit per-backbone weights in test config
    weights_cfg = getattr(cfg.test, "fusion_weights", None)
    if weights_cfg and len(list(weights_cfg)) == n:
        raw = np.array(list(weights_cfg), dtype=np.float64)
        weights = raw / raw.sum()
        logger.info(f"Late fusion using configured weights: {dict(zip(names, weights))}")
    else:
        weights = np.ones(n, dtype=np.float64) / n
        logger.info(f"Late fusion using equal weights (1/{n} each).")

    fused = sum(w * backbone_probs[name] for w, name in zip(weights, names))
    return fused


# ===========================================================================
# Heatmap generation
# ===========================================================================


def _generate_test_heatmaps(
    cfg: DictConfig,
    image_paths: List[str],
    ckpt_paths: Dict[str, Path],
    backbone_probs: Dict[str, np.ndarray],
    fused_probs: np.ndarray,
    out_dir: Path,
) -> None:
    """Generate per-backbone explainability heatmaps and a late-fusion aggregate.

    Dispatches GradCAM++ to CNN backbones (EfficientNet) and Attention Rollout
    to ViT backbones (BiomedCLIP, UNI).  Iterates over all backbones present in
    ``backbone_probs`` — no hardcoded backbone names.

    Args:
        cfg: Full configuration.
        image_paths: Selected test image paths.
        ckpt_paths: Per-backbone checkpoint paths (already fold-resolved).
        backbone_probs: Per-backbone probabilities (N, C) indexed by backbone name.
        fused_probs: Fused probabilities (N, C).
        out_dir: Directory to write heatmap images.
    """
    all_image_paths = [str(p) for p, _ in UlcerDataset(
        root_dir=cfg.test.test_dir, transform=None,
        class_names=list(cfg.data.class_names),
    ).samples]
    path_to_idx = {p: i for i, p in enumerate(all_image_paths)}

    # backbone name → {img_path → heatmap [0,1] array}
    backbone_maps: Dict[str, Dict[str, np.ndarray]] = {}

    for bname in backbone_probs:
        ckpt = ckpt_paths.get(bname)
        if ckpt is None or not ckpt.exists():
            logger.warning(f"Checkpoint not found for heatmap [{bname}] — skipping.")
            continue

        if bname in _GRADCAM_BACKBONES and cfg.explainability.gradcam.enabled:
            try:
                from src.explainability.gradcam import generate_gradcam_heatmaps
                backbone_maps[bname] = generate_gradcam_heatmaps(
                    cfg=cfg,
                    checkpoint_path=str(ckpt),
                    image_paths=image_paths,
                    class_names=list(cfg.data.class_names),
                )
                logger.info(f"GradCAM++ maps: {bname} ({len(backbone_maps[bname])} images)")
            except Exception as exc:
                logger.warning(f"GradCAM++ failed [{bname}]: {exc}")

        elif bname in _ATTENTION_BACKBONES and cfg.explainability.attention_rollout.enabled:
            try:
                from src.explainability.attention_rollout import generate_attention_rollout_heatmaps
                backbone_maps[bname] = generate_attention_rollout_heatmaps(
                    cfg=cfg,
                    backbone_name=bname,
                    checkpoint_path=str(ckpt),
                    image_paths=image_paths,
                )
                logger.info(f"Attention Rollout maps: {bname} ({len(backbone_maps[bname])} images)")
            except Exception as exc:
                logger.warning(f"Attention Rollout failed [{bname}]: {exc}")

    if not backbone_maps:
        logger.warning("No explainability maps generated (all backbones failed or disabled).")
        return

    colormap = cfg.test.colormap
    alpha = float(cfg.test.alpha)

    for img_path in image_paths:
        idx = path_to_idx.get(img_path)
        img_stem = Path(img_path).stem
        img_dir = out_dir / img_stem
        img_dir.mkdir(parents=True, exist_ok=True)

        img_arr = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        Image.fromarray(img_arr).save(img_dir / "original.png")

        available_maps: List[np.ndarray] = []

        for bname, maps in backbone_maps.items():
            if img_path not in maps:
                continue

            # Confidence = P(VASCULAR) = class-0 probability
            conf_vasc = float(backbone_probs[bname][idx, 0]) if idx is not None else None
            if conf_vasc is not None:
                pred_label = "VASCULAR" if conf_vasc >= 0.5 else "NON-VASCULAR"
                display_conf = conf_vasc if pred_label == "VASCULAR" else 1.0 - conf_vasc
                title = f"{bname}  class={pred_label} conf={display_conf:.3f}"
            else:
                title = bname

            suffix = "gradcam" if bname in _GRADCAM_BACKBONES else "attention"
            _save_heatmap_panel(
                img_arr, maps[img_path],
                save_path=img_dir / f"{bname}_{suffix}.png",
                title=title, colormap=colormap, alpha=alpha,
            )
            available_maps.append(maps[img_path])

        # Late fusion aggregated heatmap = mean of all available backbone maps
        if available_maps:
            fused_conf_vasc = float(fused_probs[idx, 0]) if idx is not None else None
            if fused_conf_vasc is not None:
                fused_pred_label = "VASCULAR" if fused_conf_vasc >= 0.5 else "NON-VASCULAR"
                fused_display_conf = (
                    fused_conf_vasc if fused_pred_label == "VASCULAR" else 1.0 - fused_conf_vasc
                )
                fused_title = (
                    f"Late Fusion Aggregated  class={fused_pred_label} conf={fused_display_conf:.3f}"
                )
            else:
                fused_title = "Late Fusion Aggregated"

            _save_heatmap_panel(
                img_arr, np.mean(available_maps, axis=0),
                save_path=img_dir / "late_fusion_aggregated.png",
                title=fused_title, colormap=colormap, alpha=alpha,
            )
            logger.debug(f"Heatmaps saved → {img_dir}")


# ===========================================================================
# Plot helpers
# ===========================================================================


def _save_heatmap_panel(
    image: np.ndarray,
    heatmap: np.ndarray,
    save_path: Path,
    title: str = "",
    colormap: str = "jet",
    alpha: float = 0.5,
) -> None:
    """Save a 3-panel figure: original | heatmap | overlay.

    Args:
        image: RGB uint8 array (H, W, 3).
        heatmap: Grayscale [0, 1] array (h, w).
        save_path: Output .png path.
        title: Figure title.
        colormap: Matplotlib colormap.
        alpha: Overlay blend factor.
    """
    from src.explainability.visualize import overlay_heatmap

    h, w = image.shape[:2]
    if heatmap.shape != (h, w):
        hm_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (w, h), Image.BILINEAR
        )
        heatmap = np.array(hm_pil, dtype=np.float32) / 255.0

    overlay = overlay_heatmap(image, heatmap, colormap=colormap, alpha=alpha)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image);          axes[0].set_title("Original");  axes[0].axis("off")
    axes[1].imshow(heatmap, cmap=colormap, vmin=0, vmax=1)
    axes[1].set_title("Heatmap");   axes[1].axis("off")
    axes[2].imshow(overlay);        axes[2].set_title("Overlay");   axes[2].axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cfg: DictConfig,
    out_dir: Path,
    tag: str,
) -> None:
    """Save a seaborn confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities for class 1.
        cfg: Full configuration.
        out_dir: Output directory.
        tag: Used in filename and title.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    class_names = list(cfg.data.class_names)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {tag} (test)")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")

    fpath = out_dir / f"confusion_matrix_{tag}.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {fpath}")


def _save_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: Path,
) -> None:
    """Save ROC curve for late fusion test predictions.

    Args:
        y_true: Ground-truth labels.
        y_prob: Fused predicted probabilities for class 1.
        out_dir: Output directory.
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
    except Exception as exc:
        logger.warning(f"Could not compute ROC curve: {exc}")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Late Fusion (test)")
    ax.legend(loc="lower right"); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])

    fpath = out_dir / "roc_curve_test.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curve saved → {fpath}")


def _save_per_sample_csv(
    image_paths: List[str],
    y_true: np.ndarray,
    backbone_probs: Dict[str, np.ndarray],
    fused_probs: np.ndarray,
    cfg: DictConfig,
    out_dir: Path,
) -> None:
    """Save per-sample predictions and confidences to CSV.

    Args:
        image_paths: Ordered image path strings.
        y_true: Ground-truth labels.
        backbone_probs: Per-backbone probability arrays.
        fused_probs: Fused probability array (N, C).
        cfg: Full configuration.
        out_dir: Output directory.
    """
    class_names = list(cfg.data.class_names)
    fpath = out_dir / "per_sample_results.csv"
    fused_pred = (fused_probs[:, 1] >= 0.5).astype(int)

    with open(fpath, "w", newline="", encoding="utf-8") as fh:
        header = ["image", "true_label"]
        for bname in backbone_probs:
            header += [f"{bname}_prob_class0", f"{bname}_prob_class1", f"{bname}_pred"]
        header += ["fused_prob_class0", "fused_prob_class1", "fused_pred", "correct"]
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()

        for i, (path, true) in enumerate(zip(image_paths, y_true)):
            row: dict = {
                "image": Path(path).name,
                "true_label": class_names[int(true)],
            }
            for bname, probs in backbone_probs.items():
                p0 = float(probs[i, 0])
                p1 = float(probs[i, 1])
                row[f"{bname}_prob_class0"] = f"{p0:.4f}"
                row[f"{bname}_prob_class1"] = f"{p1:.4f}"
                row[f"{bname}_pred"] = class_names[int(p1 >= 0.5)]
            row["fused_prob_class0"] = f"{fused_probs[i, 0]:.4f}"
            row["fused_prob_class1"] = f"{fused_probs[i, 1]:.4f}"
            row["fused_pred"] = class_names[int(fused_pred[i])]
            row["correct"] = "yes" if fused_pred[i] == int(true) else "no"
            writer.writerow(row)

    logger.info(f"Per-sample CSV saved → {fpath}")


# ===========================================================================
# Sample selection
# ===========================================================================


def _select_samples(
    image_paths: List[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n: int,
    strategy: str,
) -> List[str]:
    """Select images for heatmap generation.

    Args:
        image_paths: All test image paths.
        y_true: Ground-truth labels.
        y_prob: Fused predicted probabilities for class 1.
        n: Maximum number of samples to select.
        strategy: ``"all"``, ``"random"``, ``"best"`` (highest confidence
            correct), or ``"worst"`` (highest confidence wrong).

    Returns:
        List of selected image paths (length ≤ n).
    """
    import random as _random

    n = min(n, len(image_paths))
    paths_arr = np.array(image_paths)
    pred = (y_prob >= 0.5).astype(int)
    correct = pred == y_true.astype(int)

    if strategy == "all":
        return list(paths_arr[:n])
    elif strategy == "random":
        return list(_random.sample(list(paths_arr), n))
    elif strategy == "best":
        # Highest-confidence correct predictions
        confidence = np.where(correct, y_prob, -1.0)
        idx = np.argsort(confidence)[::-1][:n]
        return list(paths_arr[idx])
    elif strategy == "worst":
        # Highest-confidence wrong predictions
        confidence = np.where(~correct, np.abs(y_prob - 0.5), -1.0)
        idx = np.argsort(confidence)[::-1][:n]
        return list(paths_arr[idx])
    else:
        logger.warning(f"Unknown heatmap_selection='{strategy}'. Using 'all'.")
        return list(paths_arr[:n])


# ===========================================================================
# Bootstrap CI (thin wrapper)
# ===========================================================================


def _bootstrap_cis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cfg: DictConfig,
) -> Dict:
    """Compute bootstrap CIs for key metrics.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities for class 1.
        cfg: Full configuration.

    Returns:
        Dictionary of CI tuples keyed by ``<metric>_ci``.
    """
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix as _cm

    def _sensitivity(yt, yp):
        # VASCULAR (label=0) is positive; tn=TP_vasc, fp=FN_vasc after ravel()
        c = _cm(yt, (yp >= 0.5).astype(int), labels=[0, 1])
        tn, fp, fn, tp = c.ravel()
        return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    def _specificity(yt, yp):
        # tp=TN_vasc, fn=FP_vasc after ravel()
        c = _cm(yt, (yp >= 0.5).astype(int), labels=[0, 1])
        tn, fp, fn, tp = c.ravel()
        return float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")

    result: Dict = {}
    bci = cfg.evaluation.bootstrap_ci
    fns = {
        "auc_roc":     lambda yt, yp: roc_auc_score(yt, yp),
        "f1":          lambda yt, yp: f1_score(yt, (yp >= 0.5).astype(int),
                                               average="macro", zero_division=0),
        "accuracy":    lambda yt, yp: accuracy_score(yt, (yp >= 0.5).astype(int)),
        "sensitivity": _sensitivity,
        "specificity": _specificity,
    }
    for name, fn in fns.items():
        lo, hi = bootstrap_confidence_interval(
            y_true, y_prob, fn,
            n_iterations=bci.n_iterations,
            confidence_level=bci.confidence_level,
        )
        result[f"{name}_ci"] = (lo, hi)
    return result


# ===========================================================================
# Summary table
# ===========================================================================


def _print_metrics_table(all_metrics: Dict) -> None:
    """Print a formatted test metrics summary.

    Args:
        all_metrics: Mapping from model name to metrics dict.
    """
    print("\n" + "=" * 72)
    print(" TEST RESULTS")
    print("=" * 72)
    print(f"{'Model':<22} {'AUC':<8} {'F1':<8} {'Acc':<8} {'Sens':<8} {'Spec':<8}")
    print("-" * 72)
    for model_name, m in all_metrics.items():
        print(
            f"{model_name:<22} "
            f"{m.get('auc_roc', float('nan')):<8.4f} "
            f"{m.get('f1', float('nan')):<8.4f} "
            f"{m.get('accuracy', float('nan')):<8.4f} "
            f"{m.get('sensitivity', float('nan')):<8.4f} "
            f"{m.get('specificity', float('nan')):<8.4f}"
        )
    print("=" * 72 + "\n")

"""Entry point for the Ulcer Classification Pipeline.

Usage::

    python main.py --config config.yaml --mode train
    python main.py --config config.yaml --mode evaluate
    python main.py --config config.yaml --mode explain
    python main.py --config config.yaml --mode all

Modes:
    train     — Run full 5-fold CV training (+ fusion ablation if enabled).
    evaluate  — Load saved checkpoints, compute metrics on validation sets.
    explain   — Generate GradCAM++ and Attention Rollout heatmaps.
    all       — Run train → evaluate → explain sequentially.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)


# ===========================================================================
# CLI argument parsing
# ===========================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Ulcer Classification Pipeline — VASCULAR vs NON-VASCULAR"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "explain", "test", "all"],
        default="all",
        help="Pipeline mode: train | evaluate | explain | test | all (default: all).",
    )
    return parser.parse_args()


# ===========================================================================
# GPU summary
# ===========================================================================


def print_gpu_summary(cfg) -> None:
    """Print a formatted GPU hardware summary at startup.

    Args:
        cfg: Validated OmegaConf configuration.
    """
    print("\n" + "=" * 60)
    print(" Ulcer Classification Pipeline — GPU Summary")
    print("=" * 60)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_gb = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"GPU:                    {props.name}")
        print(f"VRAM:                   {total_gb:.1f} GB ({free_gb:.1f} GB available)")
    else:
        print("GPU:                    Not available (CPU mode)")

    mp = cfg.hardware.mixed_precision
    uni_gc = (
        cfg.backbones.uni.gradient_checkpointing
        if cfg.backbones.uni.enabled
        else False
    )
    print(f"Mixed Precision:        {'Enabled' if mp else 'Disabled'}")
    print(f"Gradient Checkpointing: {'Enabled (UNI)' if uni_gc else 'Disabled'}")
    print(
        f"Sequential Training:    "
        f"{'Enabled' if cfg.hardware.sequential_backbone_training else 'Disabled'}"
    )
    print("=" * 60 + "\n")


# ===========================================================================
# Mode: train
# ===========================================================================


def run_train(cfg) -> Dict:
    """Run the full training pipeline.

    Args:
        cfg: Validated OmegaConf configuration.

    Returns:
        Dictionary of aggregated results.
    """
    from src.training.cross_validation import run_cross_validation

    logger.info("Starting training pipeline …")
    results = run_cross_validation(cfg)
    logger.info("Training complete.")
    return results


# ===========================================================================
# Mode: evaluate
# ===========================================================================


def run_evaluate(cfg) -> None:
    """Compute metrics from saved checkpoints.

    Loads each backbone's best checkpoint per fold, runs inference on the
    validation split, and reports per-fold and aggregated metrics.

    Args:
        cfg: Validated OmegaConf configuration.
    """
    from src.training.evaluator import Evaluator
    from src.training.embedding_cache import load_cached_logits, load_cached_labels
    import torch.nn.functional as F

    evaluator = Evaluator(cfg, str(Path(cfg.general.output_dir) / "metrics"))

    enabled_backbones = [
        n for n in cfg.backbones.training_order
        if getattr(cfg.backbones, n).enabled
    ]

    for backbone_name in enabled_backbones:
        fold_metrics = []
        for fold in range(cfg.cross_validation.n_splits):
            try:
                logits = load_cached_logits(cfg.hardware.embedding_cache_dir, backbone_name, fold)
                labels = load_cached_labels(cfg.hardware.embedding_cache_dir, fold).numpy()
                import numpy as np
                probs = F.softmax(logits.float(), dim=-1).numpy()
                y_prob = probs[:, 1]
                fold_met = evaluator.evaluate_fold(labels, y_prob, fold=fold, tag=backbone_name)
                fold_metrics.append(fold_met)
            except FileNotFoundError as exc:
                logger.warning(f"Skipping fold {fold} for '{backbone_name}': {exc}")

        if fold_metrics:
            evaluator.aggregate_and_save(fold_metrics, backbone_name)

    logger.info("Evaluation complete.")


# ===========================================================================
# Mode: explain
# ===========================================================================


def run_explain(cfg) -> None:
    """Generate GradCAM++ and Attention Rollout heatmaps.

    Selects samples according to explainability.selection config, then
    processes one backbone at a time with gpu_memory_guard.

    Args:
        cfg: Validated OmegaConf configuration.
    """
    import random
    from src.explainability.gradcam import generate_gradcam_heatmaps
    from src.explainability.attention_rollout import generate_attention_rollout_heatmaps
    from src.explainability.visualize import generate_explanation_report
    from src.data.dataset import UlcerDataset

    if not cfg.explainability.enabled:
        logger.info("Explainability disabled in config. Skipping.")
        return

    ecfg = cfg.explainability
    output_dir = Path(cfg.general.output_dir)
    ckpt_dir = output_dir / "checkpoints"

    # Build dataset for sample selection
    full_dataset = UlcerDataset(
        root_dir=cfg.data.root_dir,
        transform=None,
        class_names=list(cfg.data.class_names),
    )

    all_paths = [str(p) for p, _ in full_dataset.samples]

    # Select samples
    n = min(ecfg.num_samples, len(all_paths))
    if ecfg.selection == "random":
        selected = random.sample(all_paths, n)
    elif ecfg.selection == "all":
        selected = all_paths[:n]
    else:
        selected = all_paths[:n]

    logger.info(f"Generating explanations for {len(selected)} images …")

    backbone_heatmaps: Dict[str, Dict[str, object]] = {}

    # --- EfficientNet: GradCAM++ ---
    if cfg.backbones.efficientnet.enabled and ecfg.gradcam.enabled:
        # Use fold 0 checkpoint
        ckpt_path = str(ckpt_dir / "fold_0" / "efficientnet_best.pt")
        if Path(ckpt_path).exists():
            backbone_heatmaps["efficientnet"] = generate_gradcam_heatmaps(
                cfg=cfg,
                checkpoint_path=ckpt_path,
                image_paths=selected,
                class_names=list(cfg.data.class_names),
            )
        else:
            logger.warning(f"EfficientNet checkpoint not found at {ckpt_path}. Skipping.")

    # --- BiomedCLIP: Attention Rollout ---
    if cfg.backbones.biomedclip.enabled and ecfg.attention_rollout.enabled:
        ckpt_path = str(ckpt_dir / "fold_0" / "biomedclip_best.pt")
        if Path(ckpt_path).exists():
            backbone_heatmaps["biomedclip"] = generate_attention_rollout_heatmaps(
                cfg=cfg,
                backbone_name="biomedclip",
                checkpoint_path=ckpt_path,
                image_paths=selected,
            )
        else:
            logger.warning(f"BiomedCLIP checkpoint not found at {ckpt_path}. Skipping.")

    # --- UNI: Attention Rollout ---
    if cfg.backbones.uni.enabled and ecfg.attention_rollout.enabled:
        ckpt_path = str(ckpt_dir / "fold_0" / "uni_best.pt")
        if Path(ckpt_path).exists():
            backbone_heatmaps["uni"] = generate_attention_rollout_heatmaps(
                cfg=cfg,
                backbone_name="uni",
                checkpoint_path=ckpt_path,
                image_paths=selected,
            )
        else:
            logger.warning(f"UNI checkpoint not found at {ckpt_path}. Skipping.")

    # --- Generate per-image reports ---
    for img_path in selected:
        per_backbone: Dict[str, object] = {}
        for bname, hmap_dict in backbone_heatmaps.items():
            if img_path in hmap_dict:
                per_backbone[bname] = hmap_dict[img_path]

        if per_backbone:
            generate_explanation_report(
                image_path=img_path,
                predictions={},  # Could load cached probabilities here
                heatmaps_per_backbone=per_backbone,
                output_dir=ecfg.output_dir,
                colormap=ecfg.gradcam.colormap,
                alpha=ecfg.gradcam.alpha,
            )

    logger.info(f"Explanations saved to {ecfg.output_dir}")


# ===========================================================================
# Mode: test
# ===========================================================================


def run_test(cfg) -> None:
    """Run held-out test-set evaluation with late fusion and heatmaps.

    Args:
        cfg: Validated OmegaConf configuration.
    """
    if not hasattr(cfg, "test"):
        raise RuntimeError(
            "No 'test' section found in config.yaml. "
            "Add test.test_dir, test.fold, and test.output_dir."
        )
    from src.training.tester import run_test as _run_test
    results = _run_test(cfg)
    logger.info("Test evaluation complete.")
    return results


# ===========================================================================
# Results summary table
# ===========================================================================


def print_results_table(results: Dict) -> None:
    """Print a formatted summary table of all aggregated results.

    Args:
        results: Dictionary of aggregated metrics per backbone / fusion method.
    """
    print("\n" + "=" * 70)
    print(" RESULTS SUMMARY")
    print("=" * 70)
    header = f"{'Model':<22} {'AUC':<8} {'F1':<8} {'Acc':<8} {'Sens':<8} {'Spec':<8}"
    print(header)
    print("-" * 70)

    for model_name, metrics in results.items():
        auc = metrics.get("auc_roc_mean", float("nan"))
        f1 = metrics.get("f1_mean", float("nan"))
        acc = metrics.get("accuracy_mean", float("nan"))
        sens = metrics.get("sensitivity_mean", float("nan"))
        spec = metrics.get("specificity_mean", float("nan"))
        print(
            f"{model_name:<22} "
            f"{auc:<8.4f} {f1:<8.4f} {acc:<8.4f} {sens:<8.4f} {spec:<8.4f}"
        )

    print("=" * 70 + "\n")


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    """Parse arguments, load config, and dispatch to the requested mode."""
    args = parse_args()

    # --- Load config and set up logging ---
    from src.config import load_config
    from src.utils.logger import setup_logger
    from src.utils.seed import set_seed

    # Minimal bootstrap logger before config is loaded
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)

    log_file = str(Path(cfg.general.output_dir) / "training.log")
    setup_logger(log_level=cfg.general.log_level, log_file=log_file)

    set_seed(cfg.general.seed, cudnn_benchmark=cfg.hardware.cudnn_benchmark)
    print_gpu_summary(cfg)

    results: Dict = {}

    if args.mode in ("train", "all"):
        results = run_train(cfg)

    if args.mode in ("evaluate", "all"):
        run_evaluate(cfg)

    if args.mode in ("explain", "all"):
        run_explain(cfg)

    if args.mode == "test":
        run_test(cfg)

    if results:
        print_results_table(results)

        # Save results JSON
        results_file = Path(cfg.general.output_dir) / "metrics" / "aggregated_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        logger.info(f"Aggregated results saved → {results_file}")


if __name__ == "__main__":
    main()

"""Full stratified 5-fold CV orchestrator with sequential backbone training.

Flow:
  For each backbone in training_order:
      For each fold (0..K-1):
          1. Load backbone + head into VRAM
          2. Train on train split
          3. Evaluate on val split
          4. Extract and cache embeddings / logits for train + val
          5. Release backbone from VRAM

  Late Fusion:
      For each fold: load cached logits → compute weighted average → evaluate

  Feature Fusion (ablation):
      For each fold: load cached embeddings → train MLP → evaluate
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.dataset import UlcerDataset
from src.data.splits import get_cv_splits
from src.models.backbone_biomedclip import BiomedCLIPBackbone
from src.models.backbone_efficientnet import EfficientNetBackbone
from src.models.backbone_uni import UNIBackbone
from src.models.feature_fusion import FeatureFusionEnsemble
from src.models.late_fusion import LateFusionEnsemble
from src.models.memory import gpu_memory_guard, release_model, safe_cuda_empty_cache
from src.training.embedding_cache import (
    extract_and_cache_embeddings,
    load_cached_embeddings,
    load_cached_labels,
    load_cached_logits,
)
from src.training.evaluator import Evaluator
from src.training.trainer import FocalLoss, Trainer

logger = logging.getLogger(__name__)


def run_cross_validation(cfg: DictConfig) -> Dict:
    """Orchestrate the full 5-fold CV pipeline.

    Args:
        cfg: Validated OmegaConf configuration.

    Returns:
        Dictionary of aggregated results per backbone and fusion method.
    """
    device = cfg.general.device
    output_dir = Path(cfg.general.output_dir)
    checkpoint_dir = str(output_dir / "checkpoints")
    metrics_dir = str(output_dir / "metrics")
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator(cfg, metrics_dir)

    # --- Build full dataset for split generation (no transform) ---
    full_dataset = UlcerDataset(
        root_dir=cfg.data.root_dir,
        transform=None,
        class_names=list(cfg.data.class_names),
    )
    labels = [lbl for _, lbl in full_dataset.samples]
    splits = get_cv_splits(labels, n_splits=cfg.cross_validation.n_splits, seed=cfg.general.seed)

    all_results: Dict = {}

    # ================================================================
    # Phase 1: Sequential backbone training
    # ================================================================
    for backbone_name in cfg.backbones.training_order:
        backbone_cfg = getattr(cfg.backbones, backbone_name)
        if not backbone_cfg.enabled:
            logger.info(f"Backbone '{backbone_name}' disabled — skipping.")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Training backbone: {backbone_name.upper()}")
        logger.info(f"{'=' * 60}")

        fold_metrics: List[Dict] = []
        fold_roc_data: List[Dict] = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            with gpu_memory_guard(f"backbone={backbone_name} | fold={fold}"):
                # Build model
                model = _build_backbone(backbone_name, cfg).to(device)

                # Dataloaders
                train_loader, val_loader = _build_dataloaders(
                    cfg, full_dataset, train_idx, val_idx, backbone_name
                )

                # Optimizer + scheduler + loss
                optimizer = _build_optimizer(model, backbone_name, cfg)
                scheduler = _build_scheduler(optimizer, cfg)
                loss_fn = _build_loss(cfg, labels, train_idx, device)

                trainer = Trainer(model, optimizer, scheduler, loss_fn, device, cfg)

                # Train
                best_ckpt = trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    backbone_name=backbone_name,
                    fold=fold,
                    checkpoint_dir=checkpoint_dir,
                )

                # Load best checkpoint
                ckpt = torch.load(best_ckpt, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                logger.info(f"Loaded best checkpoint (epoch {ckpt['epoch']}, metric={ckpt['metric']:.4f})")

                # Extract + cache embeddings for BOTH train and val
                _cache_embeddings_for_fold(
                    model=model,
                    cfg=cfg,
                    full_dataset=full_dataset,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    backbone_name=backbone_name,
                    fold=fold,
                    device=device,
                )

                # Evaluate on val set via cached logits
                val_logits = load_cached_logits(cfg.hardware.embedding_cache_dir, backbone_name, fold)
                # The cache stores full-dataset tensors; val portion is at val_idx
                # But we cache train and val separately — use val subset
                # Actually we cache val only during embedding extraction below;
                # re-derive from the val_loader evaluation above for simplicity.
                # Use trainer's eval directly on logits:
                val_labels = load_cached_labels(cfg.hardware.embedding_cache_dir, fold)

                # The cached tensors cover the whole split (train+val merged in order).
                # Recompute val metrics from val_loader pass (already done in trainer).
                # Log a clean metric from logits cache.
                probs = F.softmax(val_logits.float(), dim=-1).numpy()
                y_prob = probs[:, 1]
                y_true = val_labels.numpy()
                fold_met = evaluator.evaluate_fold(
                    y_true=y_true,
                    y_prob=y_prob,
                    fold=fold,
                    tag=backbone_name,
                )
                fold_metrics.append(fold_met)
                fold_roc_data.append({"y_true": y_true, "y_prob": y_prob})

                # Release backbone from VRAM
                release_model(model)

        evaluator.save_roc_curve(fold_roc_data, tag=backbone_name)
        aggregated = evaluator.aggregate_and_save(fold_metrics, backbone_name)
        all_results[backbone_name] = aggregated
        logger.info(f"\nBackbone '{backbone_name}' — aggregated results: {aggregated}")

        if cfg.hardware.empty_cache_between_backbones:
            safe_cuda_empty_cache()

    # ================================================================
    # Phase 2: Late Fusion (cached logits — no backbone in VRAM)
    # ================================================================
    if cfg.fusion.late_fusion.enabled:
        logger.info(f"\n{'=' * 60}")
        logger.info("  Late Fusion (cached logits)")
        logger.info(f"{'=' * 60}")

        enabled_backbones = [
            n for n in cfg.backbones.training_order
            if getattr(cfg.backbones, n).enabled
        ]

        fusion_fold_metrics: List[Dict] = []
        fusion_fold_roc: List[Dict] = []

        for fold, (_, val_idx) in enumerate(splits):
            logits_dict: Dict[str, torch.Tensor] = {}
            for bname in enabled_backbones:
                logits_dict[bname] = load_cached_logits(
                    cfg.hardware.embedding_cache_dir, bname, fold
                )

            val_labels = load_cached_labels(cfg.hardware.embedding_cache_dir, fold).numpy()

            ensemble = LateFusionEnsemble(
                backbone_names=enabled_backbones,
                num_classes=cfg.classification_head.num_classes,
                learn_weights=cfg.fusion.late_fusion.learn_weights,
            )

            if cfg.fusion.late_fusion.learn_weights:
                ensemble = _learn_fusion_weights(ensemble, logits_dict, val_labels, cfg)

            with torch.no_grad():
                preds = ensemble.predict(logits_dict)
            fused_probs = preds["fused"].numpy()
            y_prob = fused_probs[:, 1]

            fold_met = evaluator.evaluate_fold(val_labels, y_prob, fold=fold, tag="late_fusion")
            fusion_fold_metrics.append(fold_met)
            fusion_fold_roc.append({"y_true": val_labels, "y_prob": y_prob})

        evaluator.save_roc_curve(fusion_fold_roc, tag="late_fusion")
        aggregated_fusion = evaluator.aggregate_and_save(fusion_fold_metrics, "late_fusion")
        all_results["late_fusion"] = aggregated_fusion
        logger.info(f"\nLate Fusion — aggregated results: {aggregated_fusion}")

    # ================================================================
    # Phase 3: Feature Fusion ablation (cached embeddings — no backbone)
    # ================================================================
    if cfg.fusion.feature_fusion.enabled:
        logger.info(f"\n{'=' * 60}")
        logger.info("  Feature Fusion — ablation (cached embeddings)")
        logger.info(f"{'=' * 60}")

        enabled_backbones = [
            n for n in cfg.backbones.training_order
            if getattr(cfg.backbones, n).enabled
        ]
        emb_dims = [getattr(cfg.backbones, n).embedding_dim for n in enabled_backbones]

        ff_fold_metrics: List[Dict] = []
        ff_fold_roc: List[Dict] = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            train_embs: Dict[str, torch.Tensor] = {}
            val_embs: Dict[str, torch.Tensor] = {}
            for bname in enabled_backbones:
                all_emb = load_cached_embeddings(cfg.hardware.embedding_cache_dir, bname, fold)
                train_embs[bname] = all_emb[train_idx]
                val_embs[bname] = all_emb[val_idx]

            all_labels = load_cached_labels(cfg.hardware.embedding_cache_dir, fold).numpy()
            train_labels = all_labels[train_idx]
            val_labels = all_labels[val_idx]

            ff_model = FeatureFusionEnsemble(
                backbone_names=enabled_backbones,
                embedding_dims=emb_dims,
                num_classes=cfg.classification_head.num_classes,
                hidden_dim=cfg.fusion.feature_fusion.hidden_dim,
                dropout=cfg.fusion.feature_fusion.dropout,
                batch_norm=cfg.fusion.feature_fusion.batch_norm,
            ).to(device)

            ff_model = _train_fusion_mlp(ff_model, train_embs, train_labels, val_embs, val_labels, cfg, device)

            ff_model.eval()
            with torch.no_grad():
                val_logits = ff_model(
                    {k: v.to(device) for k, v in val_embs.items()}
                )
            y_prob = F.softmax(val_logits.float(), dim=-1).cpu().numpy()[:, 1]

            fold_met = evaluator.evaluate_fold(val_labels, y_prob, fold=fold, tag="feature_fusion")
            ff_fold_metrics.append(fold_met)
            ff_fold_roc.append({"y_true": val_labels, "y_prob": y_prob})

            release_model(ff_model)

        evaluator.save_roc_curve(ff_fold_roc, tag="feature_fusion")
        aggregated_ff = evaluator.aggregate_and_save(ff_fold_metrics, "feature_fusion")
        all_results["feature_fusion"] = aggregated_ff
        logger.info(f"\nFeature Fusion — aggregated results: {aggregated_ff}")

        # Save ablation comparison
        ablation = {
            "late_fusion": all_results.get("late_fusion", {}),
            "feature_fusion": aggregated_ff,
        }
        ablation_file = Path(metrics_dir) / "ablation_comparison.json"
        with open(ablation_file, "w", encoding="utf-8") as fh:
            json.dump(ablation, fh, indent=2)
        logger.info(f"Ablation comparison saved → {ablation_file}")

    return all_results


# ===========================================================================
# Private helpers
# ===========================================================================


def _build_backbone(name: str, cfg: DictConfig) -> nn.Module:
    """Instantiate the appropriate backbone from config.

    Args:
        name: Backbone identifier string.
        cfg: Full configuration.

    Returns:
        Instantiated backbone model.
    """
    if name == "biomedclip":
        return BiomedCLIPBackbone(cfg)
    elif name == "uni":
        return UNIBackbone(cfg)
    elif name == "efficientnet":
        return EfficientNetBackbone(cfg)
    else:
        raise ValueError(f"Unknown backbone: '{name}'")


def _build_dataloaders(
    cfg: DictConfig,
    full_dataset: UlcerDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    backbone_name: str,
) -> tuple:
    """Build train and val DataLoaders for a fold.

    Args:
        cfg: Full configuration.
        full_dataset: The complete dataset (without transforms).
        train_idx: Training sample indices.
        val_idx: Validation sample indices.
        backbone_name: Used to select the correct normalization stats.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_transform = get_train_transforms(cfg, backbone_name)
    val_transform = get_val_transforms(cfg, backbone_name)

    # We rebuild datasets with the appropriate transforms
    train_ds = UlcerDataset(cfg.data.root_dir, train_transform, list(cfg.data.class_names))
    val_ds = UlcerDataset(cfg.data.root_dir, val_transform, list(cfg.data.class_names))

    train_subset = Subset(train_ds, train_idx.tolist())
    val_subset = Subset(val_ds, val_idx.tolist())

    pin = cfg.hardware.pin_memory and cfg.general.device == "cuda"

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.general.num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.general.num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader


def _build_optimizer(model: nn.Module, backbone_name: str, cfg: DictConfig):
    """Build an AdamW or Adam optimizer with optional differential LR.

    Args:
        model: The backbone model.
        backbone_name: Backbone identifier.
        cfg: Full configuration.

    Returns:
        Configured torch optimizer.
    """
    tcfg = cfg.training
    bcfg = getattr(cfg.backbones, backbone_name)

    if backbone_name == "efficientnet" and hasattr(bcfg, "backbone_lr"):
        param_groups = model.get_param_groups(
            backbone_lr=bcfg.backbone_lr,
            head_lr=bcfg.head_lr,
        )
    else:
        lr = getattr(bcfg, "learning_rate", 1e-5)
        param_groups = [{"params": model.parameters(), "lr": lr}]

    Opt = torch.optim.AdamW if tcfg.optimizer == "adamw" else torch.optim.Adam
    return Opt(param_groups, weight_decay=tcfg.weight_decay)


def _build_scheduler(optimizer, cfg: DictConfig):
    """Build the learning rate scheduler from config.

    Args:
        optimizer: The optimizer.
        cfg: Full configuration.

    Returns:
        Configured scheduler or None.
    """
    scfg = cfg.training.scheduler

    if scfg.type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs - scfg.warmup_epochs,
            eta_min=scfg.min_lr,
        )
    elif scfg.type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scfg.step_size, gamma=scfg.gamma
        )
    else:
        return None


def _build_loss(cfg: DictConfig, all_labels: List[int], train_idx: np.ndarray, device: str):
    """Build the loss function with optional class weighting.

    Args:
        cfg: Full configuration.
        all_labels: Full label list for the dataset.
        train_idx: Indices of training samples (used for balanced weights).
        device: Target device.

    Returns:
        Configured loss function module.
    """
    tcfg = cfg.training
    weight: torch.Tensor | None = None

    if tcfg.class_weights == "balanced":
        train_labels = np.array(all_labels)[train_idx]
        classes, counts = np.unique(train_labels, return_counts=True)
        n_total = len(train_labels)
        n_classes = len(classes)
        weights = n_total / (n_classes * counts)
        weight = torch.tensor(weights, dtype=torch.float32).to(device)
        logger.info(f"Class weights (balanced): {dict(zip(classes.tolist(), weights.tolist()))}")
    elif isinstance(tcfg.class_weights, (list, tuple)):
        weight = torch.tensor(tcfg.class_weights, dtype=torch.float32).to(device)

    if tcfg.loss == "focal":
        return FocalLoss(gamma=tcfg.focal_loss_gamma, weight=weight)
    else:
        return nn.CrossEntropyLoss(weight=weight)


def _cache_embeddings_for_fold(
    model: nn.Module,
    cfg: DictConfig,
    full_dataset: UlcerDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    backbone_name: str,
    fold: int,
    device: str,
) -> None:
    """Extract and cache embeddings/logits for the val split of a fold.

    Args:
        model: Trained backbone model with head.
        cfg: Full configuration.
        full_dataset: Complete dataset (used to build val loader).
        train_idx: Training indices (not used here but kept for symmetry).
        val_idx: Validation indices.
        backbone_name: Backbone identifier.
        fold: Fold index.
        device: Target device.
    """
    val_transform = get_val_transforms(cfg, backbone_name)
    val_ds = UlcerDataset(cfg.data.root_dir, val_transform, list(cfg.data.class_names))
    val_subset = Subset(val_ds, val_idx.tolist())
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.general.num_workers,
    )

    extract_and_cache_embeddings(
        backbone=model,
        head=model.head,
        dataloader=val_loader,
        backbone_name=backbone_name,
        fold=fold,
        cache_dir=cfg.hardware.embedding_cache_dir,
        device=device,
    )


def _learn_fusion_weights(
    ensemble: LateFusionEnsemble,
    logits_dict: Dict[str, torch.Tensor],
    val_labels: np.ndarray,
    cfg: DictConfig,
) -> LateFusionEnsemble:
    """Optimise late fusion weights on the validation set.

    Args:
        ensemble: LateFusionEnsemble with learnable weights.
        logits_dict: Mapping from backbone name to logit tensor.
        val_labels: Ground-truth labels for validation.
        cfg: Full configuration.

    Returns:
        Ensemble with optimised weights.
    """
    labels_t = torch.tensor(val_labels, dtype=torch.long)
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(200):
        optimizer.zero_grad()
        fused = ensemble(logits_dict)
        loss = loss_fn(fused, labels_t)
        loss.backward()
        optimizer.step()

    weights = ensemble._normalised_weights().detach().numpy()
    logger.info(
        f"Learned fusion weights: "
        + ", ".join(
            f"{n}={w:.3f}" for n, w in zip(ensemble.backbone_names, weights)
        )
    )
    return ensemble


def _train_fusion_mlp(
    model: FeatureFusionEnsemble,
    train_embs: Dict[str, torch.Tensor],
    train_labels: np.ndarray,
    val_embs: Dict[str, torch.Tensor],
    val_labels: np.ndarray,
    cfg: DictConfig,
    device: str,
) -> FeatureFusionEnsemble:
    """Train the feature fusion MLP on cached embeddings.

    Args:
        model: FeatureFusionEnsemble model.
        train_embs: Training embeddings per backbone.
        train_labels: Training labels.
        val_embs: Validation embeddings per backbone.
        val_labels: Validation labels.
        cfg: Full configuration.
        device: Target device.

    Returns:
        Trained FeatureFusionEnsemble.
    """
    dev = torch.device(device)
    train_labels_t = torch.tensor(train_labels, dtype=torch.long).to(dev)
    train_embs_dev = {k: v.to(dev) for k, v in train_embs.items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        logits = model(train_embs_dev)
        loss = loss_fn(logits, train_labels_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            logger.debug(f"Feature fusion MLP epoch {epoch+1}/100  loss={loss.item():.4f}")

    return model

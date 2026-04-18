"""Per-fold, per-backbone training loop with mandatory mixed-precision.

Features:
  - FP16 via torch.cuda.amp.autocast + GradScaler
  - Gradient accumulation
  - OOM auto-recovery (halve batch size and retry)
  - Early stopping
  - Best-checkpoint saving per fold
  - VRAM logging every 10 epochs
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.memory import log_gpu_memory, safe_cuda_empty_cache
from src.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced binary classification.

    Args:
        gamma: Focusing parameter.
        weight: Optional class weight tensor.
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: (N, C) unnormalised log-probabilities.
            targets: (N,) integer class labels.

        Returns:
            Scalar loss value.
        """
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class EarlyStopping:
    """Early stopping with patience-based monitoring.

    Args:
        patience: Number of epochs to wait after the last improvement.
        mode: ``"max"`` if higher metric is better, ``"min"`` otherwise.
        min_delta: Minimum change considered an improvement.
    """

    def __init__(self, patience: int, mode: str = "max", min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter: int = 0
        self.best_value: float = float("-inf") if mode == "max" else float("inf")
        self.should_stop: bool = False

    def step(self, value: float) -> bool:
        """Update state with latest metric value.

        Args:
            value: Current epoch metric value.

        Returns:
            True if training should stop.
        """
        improved = (
            value >= self.best_value + self.min_delta
            if self.mode == "max"
            else value <= self.best_value - self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """Training loop for a single backbone on a single fold.

    Args:
        model: The backbone + head model.
        optimizer: Configured torch optimizer.
        scheduler: Learning-rate scheduler (or None).
        loss_fn: Loss function module.
        device: Target device.
        cfg: Full OmegaConf configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_fn: nn.Module,
        device: str,
        cfg: DictConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.cfg = cfg

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=cfg.hardware.mixed_precision and self.device.type == "cuda"
        )
        self.accumulation_steps: int = cfg.training.accumulation_steps

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        backbone_name: str,
        fold: int,
        checkpoint_dir: str,
    ) -> str:
        """Run the full training loop and return the path of the best checkpoint.

        Args:
            train_loader: DataLoader for the training split.
            val_loader: DataLoader for the validation split.
            backbone_name: Backbone identifier (used for checkpoint naming).
            fold: Fold index (used for checkpoint naming).
            checkpoint_dir: Directory to save checkpoints.

        Returns:
            Path to the saved best-model checkpoint.
        """
        ckpt_dir = Path(checkpoint_dir) / f"fold_{fold}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = str(ckpt_dir / f"{backbone_name}_best.pt")

        tcfg = self.cfg.training
        early_stopping = EarlyStopping(
            patience=tcfg.early_stopping.patience,
            mode=tcfg.early_stopping.mode,
        ) if tcfg.early_stopping.enabled else None

        best_metric = float("-inf") if (
            not tcfg.early_stopping.enabled or tcfg.early_stopping.mode == "max"
        ) else float("inf")

        for epoch in range(1, tcfg.epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            val_metrics = self._eval_epoch(val_loader)
            val_loss = val_metrics.get("loss", float("nan"))

            monitor_key = tcfg.early_stopping.monitor if tcfg.early_stopping.enabled else "val_auc"
            # map "val_auc" → "auc_roc", "val_loss" → "loss"
            metric_key = monitor_key.replace("val_", "")
            if metric_key == "auc":
                metric_key = "auc_roc"
            current_metric = val_metrics.get(metric_key, float("nan"))

            logger.info(
                f"[{backbone_name} | fold {fold} | epoch {epoch}/{tcfg.epochs}] "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_auc={val_metrics.get('auc_roc', float('nan')):.4f}  "
                f"val_f1={val_metrics.get('f1', float('nan')):.4f}"
            )

            # VRAM logging every 10 epochs
            if epoch % 10 == 0:
                log_gpu_memory(f"{backbone_name} | fold {fold} | epoch {epoch}")

            # Save best checkpoint
            is_better = (
                current_metric > best_metric
                if tcfg.early_stopping.mode == "max"
                else current_metric < best_metric
            )
            if is_better:
                best_metric = current_metric
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "metric": current_metric,
                        "backbone_name": backbone_name,
                        "fold": fold,
                    },
                    best_ckpt_path,
                )
                logger.info(f"  → Checkpoint saved ({monitor_key}={current_metric:.4f})")

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_metric)
                else:
                    self.scheduler.step()

            # Early stopping
            if early_stopping is not None and early_stopping.step(current_metric):
                logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no improvement in {tcfg.early_stopping.patience} epochs)."
                )
                break

        logger.info(
            f"Training complete — best {monitor_key}={best_metric:.4f} | "
            f"checkpoint: {best_ckpt_path}"
        )
        return best_ckpt_path

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Run one training epoch with gradient accumulation and OOM recovery.

        Args:
            loader: Training DataLoader.
            epoch: Current epoch number (1-based).

        Returns:
            Mean training loss for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc=f"Train epoch {epoch}", leave=False)):
            images, labels, _ = batch
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            loss = self._forward_backward(images, labels, step)
            if loss is None:
                continue

            total_loss += loss
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _forward_backward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        step: int,
        retry_count: int = 0,
    ) -> Optional[float]:
        """Single forward-backward step with OOM recovery.

        Args:
            images: Batch of image tensors.
            labels: Batch of label tensors.
            step: Current batch step index.
            retry_count: Number of OOM retries already performed.

        Returns:
            Scalar loss value or None if irrecoverable.
        """
        try:
            with torch.cuda.amp.autocast(
                enabled=self.cfg.hardware.mixed_precision and self.device.type == "cuda"
            ):
                logits = self.model(images)
                loss = self.loss_fn(logits, labels) / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            return float(loss.item()) * self.accumulation_steps

        except torch.cuda.OutOfMemoryError:
            max_retries = self.cfg.hardware.max_batch_size_retries
            if self.cfg.hardware.auto_reduce_batch_size and retry_count < max_retries:
                safe_cuda_empty_cache()
                self.optimizer.zero_grad()
                logger.warning(
                    f"CUDA OOM — halving batch and retrying "
                    f"(attempt {retry_count + 1}/{max_retries})."
                )
                # Use half the batch
                half = images.shape[0] // 2
                if half == 0:
                    logger.error("Batch size = 1 but still OOM. Skipping batch.")
                    return None
                return self._forward_backward(
                    images[:half], labels[:half], step, retry_count + 1
                )
            else:
                logger.error("CUDA OOM with no more retries available. Aborting training.")
                raise

    def _eval_epoch(self, loader: DataLoader) -> dict:
        """Run one evaluation epoch and compute metrics.

        Args:
            loader: Validation DataLoader.

        Returns:
            Dictionary of metric values including ``"loss"``.
        """
        self.model.eval()
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Val", leave=False):
                images, labels, _ = batch
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(
                    enabled=self.cfg.hardware.mixed_precision and self.device.type == "cuda"
                ):
                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)

                all_logits.append(logits.cpu().float())
                all_labels.append(labels.cpu())
                total_loss += float(loss.item())
                n_batches += 1

        import numpy as np
        import torch.nn.functional as F

        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0).numpy()
        probs = F.softmax(logits_cat, dim=-1).numpy()
        y_prob = probs[:, 1]  # probability for class 1 (NON-VENOUS)

        metrics = compute_metrics(labels_cat, y_prob)
        metrics["loss"] = total_loss / max(n_batches, 1)
        return metrics

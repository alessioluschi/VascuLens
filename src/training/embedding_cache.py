"""Embedding and logit extraction with disk caching.

After each backbone is trained on a fold, this module runs a forward
pass over the full dataset and saves:
  - embeddings: (N, embedding_dim)  →  {cache_dir}/{backbone}_fold{k}_embeddings.pt
  - logits:     (N, num_classes)    →  {cache_dir}/{backbone}_fold{k}_logits.pt
  - labels:     (N,)                →  {cache_dir}/fold{k}_labels.pt
  - paths:      list of str         →  {cache_dir}/fold{k}_paths.json

Late fusion and feature fusion read these cached tensors directly — no
backbone needs to be in VRAM during fusion training.
"""

import json
import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_and_cache_embeddings(
    backbone: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    backbone_name: str,
    fold: int,
    cache_dir: str,
    device: str,
) -> None:
    """Extract embeddings and logits from a trained backbone and save to disk.

    For each sample in the dataloader a forward pass is run and the
    following tensors / metadata are saved:

    * ``{cache_dir}/{backbone_name}_fold{fold}_embeddings.pt`` — (N, emb_dim)
    * ``{cache_dir}/{backbone_name}_fold{fold}_logits.pt``     — (N, num_classes)
    * ``{cache_dir}/fold{fold}_labels.pt``                     — (N,)
    * ``{cache_dir}/fold{fold}_paths.json``                    — list[str]

    Uses FP16 autocast for extraction.  No gradients are computed.

    Args:
        backbone: Feature extractor model (must expose ``get_embedding``).
        head: Classification head to produce logits.
        dataloader: DataLoader yielding (image, label, path) tuples.
        backbone_name: Identifier string (e.g. ``"biomedclip"``).
        fold: Fold index (0-based).
        cache_dir: Directory where .pt / .json files are written.
        device: Target device string (``"cuda"`` or ``"cpu"``).
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    emb_file = cache_path / f"{backbone_name}_fold{fold}_embeddings.pt"
    logits_file = cache_path / f"{backbone_name}_fold{fold}_logits.pt"
    labels_file = cache_path / f"fold{fold}_labels.pt"
    paths_file = cache_path / f"fold{fold}_paths.json"

    backbone.eval()
    head.eval()

    all_embeddings: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_paths: list[str] = []

    dev = torch.device(device)

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc=f"Extracting embeddings [{backbone_name} | fold {fold}]",
            leave=False,
        ):
            images, labels, paths = batch
            images = images.to(dev, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=dev.type == "cuda"):
                embeddings = backbone.get_embedding(images)
                logits = head(embeddings)

            all_embeddings.append(embeddings.cpu().float())
            all_logits.append(logits.cpu().float())
            all_labels.append(labels.cpu())
            all_paths.extend(list(paths))

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    torch.save(embeddings_tensor, emb_file)
    logger.info(f"Saved embeddings → {emb_file}  shape={tuple(embeddings_tensor.shape)}")

    torch.save(logits_tensor, logits_file)
    logger.info(f"Saved logits     → {logits_file}  shape={tuple(logits_tensor.shape)}")

    # Labels and paths are shared across backbones — only write once per fold
    if not labels_file.exists():
        torch.save(labels_tensor, labels_file)
        logger.info(f"Saved labels     → {labels_file}  shape={tuple(labels_tensor.shape)}")

    if not paths_file.exists():
        with open(paths_file, "w", encoding="utf-8") as fh:
            json.dump(all_paths, fh)
        logger.info(f"Saved paths      → {paths_file}  ({len(all_paths)} entries)")


def load_cached_logits(
    cache_dir: str,
    backbone_name: str,
    fold: int,
) -> torch.Tensor:
    """Load cached logits for a given backbone and fold.

    Args:
        cache_dir: Directory containing the cached .pt files.
        backbone_name: Backbone identifier string.
        fold: Fold index.

    Returns:
        Logit tensor of shape (N, num_classes).
    """
    path = Path(cache_dir) / f"{backbone_name}_fold{fold}_logits.pt"
    if not path.exists():
        raise FileNotFoundError(f"Cached logits not found: {path}")
    return torch.load(path, map_location="cpu")


def load_cached_embeddings(
    cache_dir: str,
    backbone_name: str,
    fold: int,
) -> torch.Tensor:
    """Load cached embeddings for a given backbone and fold.

    Args:
        cache_dir: Directory containing the cached .pt files.
        backbone_name: Backbone identifier string.
        fold: Fold index.

    Returns:
        Embedding tensor of shape (N, embedding_dim).
    """
    path = Path(cache_dir) / f"{backbone_name}_fold{fold}_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(f"Cached embeddings not found: {path}")
    return torch.load(path, map_location="cpu")


def load_cached_labels(cache_dir: str, fold: int) -> torch.Tensor:
    """Load cached labels for a fold.

    Args:
        cache_dir: Directory containing the cached .pt files.
        fold: Fold index.

    Returns:
        Label tensor of shape (N,).
    """
    path = Path(cache_dir) / f"fold{fold}_labels.pt"
    if not path.exists():
        raise FileNotFoundError(f"Cached labels not found: {path}")
    return torch.load(path, map_location="cpu")


def load_cached_paths(cache_dir: str, fold: int) -> list[str]:
    """Load cached image paths for a fold.

    Args:
        cache_dir: Directory containing the cached .json files.
        fold: Fold index.

    Returns:
        List of image path strings.
    """
    path = Path(cache_dir) / f"fold{fold}_paths.json"
    if not path.exists():
        raise FileNotFoundError(f"Cached paths not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

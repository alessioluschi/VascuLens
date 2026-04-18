"""Data augmentation and preprocessing transforms.

Train transforms apply all enabled augmentations from config.
Val/Test transforms apply only resize + center-crop + normalize.

Normalization constants per backbone:
  - EfficientNet-B0: ImageNet mean/std (standard timm default)
  - BiomedCLIP ViT-B/16: same as OpenCLIP CLIP normalization (0.5, 0.5, 0.5)
  - UNI ViT-L/16: ImageNet mean/std (timm default for UNI)
"""

import logging
from typing import Dict

from omegaconf import DictConfig
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-backbone normalization constants
# ---------------------------------------------------------------------------
_NORM_STATS: Dict[str, Dict[str, list]] = {
    "efficientnet": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "biomedclip": {
        # OpenCLIP / BiomedCLIP normalization
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
    "uni": {
        # UNI uses ImageNet stats
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}


def get_train_transforms(cfg: DictConfig, backbone_name: str) -> transforms.Compose:
    """Build the training augmentation pipeline from config values.

    Args:
        cfg: Full OmegaConf configuration.
        backbone_name: One of ``"efficientnet"``, ``"biomedclip"``, ``"uni"``.

    Returns:
        torchvision Compose transform for training.
    """
    aug = cfg.data.augmentation
    size = cfg.data.image_size
    norm = _get_norm(backbone_name)

    t: list = []

    if aug.random_resized_crop.enabled:
        t.append(
            transforms.RandomResizedCrop(
                size,
                scale=(aug.random_resized_crop.scale_min, aug.random_resized_crop.scale_max),
                antialias=True,
            )
        )
    else:
        t.append(transforms.Resize((size, size), antialias=True))

    if aug.horizontal_flip:
        t.append(transforms.RandomHorizontalFlip())

    if aug.vertical_flip:
        t.append(transforms.RandomVerticalFlip())

    if aug.rotation_degrees > 0:
        t.append(transforms.RandomRotation(degrees=aug.rotation_degrees))

    if aug.affine.enabled:
        t.append(
            transforms.RandomAffine(
                degrees=0,
                translate=tuple(aug.affine.translate),
                shear=aug.affine.shear,
            )
        )

    if aug.colour_jitter.enabled:
        t.append(
            transforms.ColorJitter(
                brightness=aug.colour_jitter.brightness,
                contrast=aug.colour_jitter.contrast,
                saturation=aug.colour_jitter.saturation,
                hue=aug.colour_jitter.hue,
            )
        )

    t.append(transforms.ToTensor())
    t.append(norm)

    logger.debug(f"Train transforms for '{backbone_name}': {t}")
    return transforms.Compose(t)


def get_val_transforms(cfg: DictConfig, backbone_name: str) -> transforms.Compose:
    """Build the validation / test preprocessing pipeline.

    Only applies resize → center-crop → to-tensor → normalize.

    Args:
        cfg: Full OmegaConf configuration.
        backbone_name: One of ``"efficientnet"``, ``"biomedclip"``, ``"uni"``.

    Returns:
        torchvision Compose transform for validation/testing.
    """
    size = cfg.data.image_size
    norm = _get_norm(backbone_name)

    return transforms.Compose(
        [
            transforms.Resize(int(size * 1.143), antialias=True),  # ~256 for 224
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            norm,
        ]
    )


def _get_norm(backbone_name: str) -> transforms.Normalize:
    """Return the Normalize transform for a given backbone.

    Args:
        backbone_name: Backbone identifier string.

    Returns:
        torchvision Normalize transform.
    """
    stats = _NORM_STATS.get(backbone_name.lower(), _NORM_STATS["efficientnet"])
    return transforms.Normalize(mean=stats["mean"], std=stats["std"])

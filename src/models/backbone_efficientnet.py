"""EfficientNet-B0 backbone wrapper.

Loaded via timm with ``num_classes=0`` to expose the 1280-dim feature
vector.  Supports full fine-tuning with a differential learning rate:
backbone parameters get ``backbone_lr`` and the attached classification
head gets ``head_lr``.
"""

import logging
from typing import Dict, List

import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.classification_head import ClassificationHead

logger = logging.getLogger(__name__)


class EfficientNetBackbone(nn.Module):
    """EfficientNet-B0 feature extractor + classification head.

    Args:
        cfg: Full OmegaConf configuration.

    Attributes:
        backbone: The timm EfficientNet-B0 feature extractor.
        head: Classification head.
        embedding_dim: Dimensionality of the output embedding (1280).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        ecfg = cfg.backbones.efficientnet
        hcfg = cfg.classification_head

        self.embedding_dim: int = ecfg.embedding_dim

        # num_classes=0 removes the built-in classifier and returns features
        self.backbone = timm.create_model(
            ecfg.model_name,
            pretrained=ecfg.pretrained,
            num_classes=0,
        )
        logger.info(
            f"Loaded EfficientNet-B0 ({ecfg.model_name}), "
            f"pretrained={ecfg.pretrained}, embedding_dim={self.embedding_dim}"
        )

        self.head = ClassificationHead(
            input_dim=self.embedding_dim,
            hidden_dim=hcfg.hidden_dim,
            num_classes=hcfg.num_classes,
            dropout=hcfg.dropout,
            activation=hcfg.activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch of images.

        Args:
            x: Input tensor of shape (N, 3, H, W).

        Returns:
            Logit tensor of shape (N, num_classes).
        """
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            features = self.backbone(x)
            logits = self.head(features)
        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 1280-dim feature embedding without the classification head.

        Args:
            x: Input tensor of shape (N, 3, H, W).

        Returns:
            Embedding tensor of shape (N, 1280).
        """
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            return self.backbone(x)

    def get_param_groups(self, backbone_lr: float, head_lr: float) -> List[Dict]:
        """Return parameter groups with differential learning rates.

        Args:
            backbone_lr: Learning rate for backbone parameters.
            head_lr: Learning rate for classification head parameters.

        Returns:
            List of parameter group dicts compatible with torch optimizers.
        """
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]

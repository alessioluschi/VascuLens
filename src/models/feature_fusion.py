"""Feature fusion ensemble (ablation study) operating on cached embeddings.

Concatenates cached embeddings from all enabled backbones
(512 + 1024 + 1280 = 2816-dim) and passes them through a small MLP.

This module does NOT hold any backbone references.  The full embedding
matrix for ~200 samples fits comfortably in RAM, and the MLP itself
occupies < 0.5 GB of VRAM during training.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FeatureFusionEnsemble(nn.Module):
    """Concatenation + MLP ensemble trained on cached embeddings.

    Architecture::

        concat(embed_1, embed_2, ...) →
        Linear(total_dim → hidden_dim) →
        [BatchNorm] → ReLU → Dropout →
        Linear(hidden_dim → num_classes)

    Args:
        backbone_names: Ordered list of backbone identifiers.
        embedding_dims: Corresponding embedding dimensions for each backbone.
        num_classes: Number of output classes.
        hidden_dim: Hidden layer size (from config).
        dropout: Dropout probability.
        batch_norm: Whether to include a BatchNorm layer.
    """

    def __init__(
        self,
        backbone_names: List[str],
        embedding_dims: List[int],
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        batch_norm: bool,
    ) -> None:
        super().__init__()
        self.backbone_names = backbone_names
        total_dim = sum(embedding_dims)

        layers: list[nn.Module] = [nn.Linear(total_dim, hidden_dim)]

        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        layers.extend(
            [
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            ]
        )
        self.mlp = nn.Sequential(*layers)

        logger.info(
            f"FeatureFusionEnsemble: {total_dim} → {hidden_dim} "
            f"(BN={batch_norm}) → {num_classes}"
        )

    def forward(self, embeddings_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate embeddings and return logits.

        Args:
            embeddings_dict: Mapping from backbone name to embedding tensor
                of shape (N, embedding_dim).

        Returns:
            Logit tensor of shape (N, num_classes).
        """
        parts: list[torch.Tensor] = []
        for name in self.backbone_names:
            if name not in embeddings_dict:
                raise KeyError(f"Missing embeddings for backbone '{name}'.")
            parts.append(embeddings_dict[name])

        concatenated = torch.cat(parts, dim=-1)
        return self.mlp(concatenated)

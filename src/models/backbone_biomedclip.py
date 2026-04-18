"""BiomedCLIP ViT-B/16 vision encoder backbone.

Loaded via open_clip from HuggingFace Hub.  Only the vision encoder is
used; the text encoder is discarded.  The CLS token embedding (512-dim)
is exposed as the feature representation.

Selective layer freezing: all layers are frozen initially, then the last
N transformer blocks are unfrozen according to config.
"""

import logging
from typing import List

import open_clip
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.classification_head import ClassificationHead

logger = logging.getLogger(__name__)

_MODEL_TAG = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class BiomedCLIPBackbone(nn.Module):
    """BiomedCLIP ViT-B/16 vision encoder + classification head.

    Args:
        cfg: Full OmegaConf configuration.

    Attributes:
        vision_encoder: The ViT vision encoder from BiomedCLIP.
        head: Classification head.
        embedding_dim: Dimensionality of the CLS token embedding (512).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        bcfg = cfg.backbones.biomedclip
        hcfg = cfg.classification_head

        self.embedding_dim: int = bcfg.embedding_dim

        logger.info(f"Loading BiomedCLIP from {_MODEL_TAG} …")
        model, _, _ = open_clip.create_model_and_transforms(_MODEL_TAG)
        self.vision_encoder: nn.Module = model.visual
        logger.info("BiomedCLIP vision encoder loaded.")

        # Selective freezing
        if bcfg.freeze_layers:
            self._freeze_all()
            self._unfreeze_last_n(bcfg.unfreeze_last_n_layers)

        self.head = ClassificationHead(
            input_dim=self.embedding_dim,
            hidden_dim=hcfg.hidden_dim,
            num_classes=hcfg.num_classes,
            dropout=hcfg.dropout,
            activation=hcfg.activation,
        )

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch of images.

        Args:
            x: Input tensor of shape (N, 3, H, W).

        Returns:
            Logit tensor of shape (N, num_classes).
        """
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            embedding = self._extract_embedding(x)
            logits = self.head(embedding)
        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the CLS token embedding (512-dim) without the head.

        Args:
            x: Input tensor of shape (N, 3, H, W).

        Returns:
            Embedding tensor of shape (N, 512).
        """
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            return self._extract_embedding(x)

    def _extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Internal: run vision encoder and extract CLS token."""
        out = self.vision_encoder(x)
        # open_clip vision encoders return (N, embedding_dim) or pooled output
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    # ------------------------------------------------------------------ #
    # Layer freezing utilities                                             #
    # ------------------------------------------------------------------ #

    def _freeze_all(self) -> None:
        """Freeze all vision encoder parameters."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        logger.debug("All BiomedCLIP vision encoder parameters frozen.")

    def _unfreeze_last_n(self, n: int) -> None:
        """Unfreeze the last N transformer blocks of the vision encoder.

        Args:
            n: Number of transformer blocks to unfreeze from the end.
        """
        # Locate transformer blocks; open_clip ViT exposes them as
        # vision_encoder.transformer.resblocks
        blocks: List[nn.Module] = []
        try:
            blocks = list(self.vision_encoder.transformer.resblocks)
        except AttributeError:
            logger.warning(
                "Could not locate transformer.resblocks — "
                "trying to unfreeze entire vision_encoder."
            )
            for param in self.vision_encoder.parameters():
                param.requires_grad = True
            return

        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

        # Also unfreeze the final projection layer if present
        for attr in ("proj", "ln_post"):
            layer = getattr(self.vision_encoder, attr, None)
            if layer is not None:
                for param in layer.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.vision_encoder.parameters())
        logger.info(
            f"BiomedCLIP: unfroze last {n} blocks — "
            f"{trainable:,} / {total:,} parameters trainable."
        )

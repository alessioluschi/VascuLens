"""UNI ViT-L/16 backbone (MahmoodLab/uni).

Loaded via timm from HuggingFace Hub.  Gradient checkpointing is
MANDATORY for RTX 3080 (10 GB VRAM) — ViT-L activations exceed 4 GB in
FP32 at batch_size=16 without checkpointing.

IMPORTANT: UNI requires accepting the model license on HuggingFace and
authenticating via ``huggingface-cli login`` before use.
"""

import logging
import os
from typing import List, Optional

import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.models.classification_head import ClassificationHead

logger = logging.getLogger(__name__)


def _hf_authenticate(cfg: DictConfig) -> None:
    """Authenticate with HuggingFace Hub before loading gated models.

    Token resolution order:
    1. ``huggingface.token`` in config.yaml
    2. ``HF_TOKEN`` environment variable

    If neither is found, a warning is logged and login is skipped (the
    model load will fail unless the user is already logged in via the
    CLI cache).

    Args:
        cfg: Full OmegaConf configuration.
    """
    from huggingface_hub import login as hf_login

    token: Optional[str] = None

    # 1. Config file
    try:
        raw = OmegaConf.select(cfg, "huggingface.token")
        if raw:
            token = str(raw)
    except Exception:
        pass

    # 2. Environment variable
    if not token:
        token = os.environ.get("HF_TOKEN") or None

    if token:
        hf_login(token=token, add_to_git_credential=False)
        logger.info("HuggingFace Hub: authenticated via token.")
    else:
        logger.warning(
            "No HuggingFace token found. "
            "Set huggingface.token in config.yaml or the HF_TOKEN environment variable. "
            "UNI model loading will fail if you have not already logged in via the CLI."
        )


class UNIBackbone(nn.Module):
    """UNI ViT-L/16 feature extractor + classification head.

    Args:
        cfg: Full OmegaConf configuration.

    Attributes:
        backbone: The timm UNI ViT-L model.
        head: Classification head.
        embedding_dim: Dimensionality of the CLS token embedding (1024).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        ucfg = cfg.backbones.uni
        hcfg = cfg.classification_head

        self.embedding_dim: int = ucfg.embedding_dim

        logger.info(f"Loading UNI from hf-hub:{ucfg.model_name} …")
        _hf_authenticate(cfg)

        self.backbone: nn.Module = timm.create_model(
            f"hf-hub:{ucfg.model_name}",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
            num_classes=0,   # removes classifier, returns CLS embedding
        )
        logger.info("UNI loaded successfully.")

        # --- MANDATORY gradient checkpointing for ViT-L on 10 GB VRAM ---
        if ucfg.gradient_checkpointing:
            try:
                self.backbone.set_grad_checkpointing(enable=True)
                logger.info(
                    "Gradient checkpointing ENABLED for UNI "
                    "(trades ~30% compute for ~50% activation VRAM reduction)."
                )
            except AttributeError:
                logger.warning(
                    "Could not enable gradient checkpointing via set_grad_checkpointing(). "
                    "UNI may exceed VRAM budget on RTX 3080."
                )

        # Selective layer freezing
        if ucfg.freeze_layers:
            self._freeze_all()
            self._unfreeze_last_n(ucfg.unfreeze_last_n_layers)

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
            embedding = self.backbone(x)
            logits = self.head(embedding)
        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the CLS token embedding (1024-dim) without the head.

        Args:
            x: Input tensor of shape (N, 3, H, W).

        Returns:
            Embedding tensor of shape (N, 1024).
        """
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            return self.backbone(x)

    # ------------------------------------------------------------------ #
    # Layer freezing utilities                                             #
    # ------------------------------------------------------------------ #

    def _freeze_all(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.debug("All UNI backbone parameters frozen.")

    def _unfreeze_last_n(self, n: int) -> None:
        """Unfreeze the last N transformer blocks.

        Args:
            n: Number of blocks to unfreeze from the end.
        """
        blocks: List[nn.Module] = []
        # timm ViT exposes blocks as model.blocks
        try:
            blocks = list(self.backbone.blocks)
        except AttributeError:
            logger.warning(
                "Could not locate backbone.blocks — unfreezing entire backbone."
            )
            for param in self.backbone.parameters():
                param.requires_grad = True
            return

        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze final norm and head layers
        for attr in ("norm", "fc_norm", "head_drop"):
            layer = getattr(self.backbone, attr, None)
            if layer is not None:
                for param in layer.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        logger.info(
            f"UNI: unfroze last {n} blocks — "
            f"{trainable:,} / {total:,} parameters trainable."
        )

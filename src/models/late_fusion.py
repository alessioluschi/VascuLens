"""Late fusion ensemble operating on cached per-backbone logits.

This module does NOT hold any backbone references and does NOT load
backbones into VRAM.  It operates entirely on tensors loaded from disk,
so its VRAM footprint is near zero.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LateFusionEnsemble(nn.Module):
    """Weighted average of per-backbone softmax probabilities.

    When ``learn_weights=True``, the combination weights are a learnable
    softmax-normalised parameter vector optimised on the validation set.
    Otherwise equal weights (1/N) are used.

    Args:
        backbone_names: Ordered list of backbone identifiers.
        num_classes: Number of output classes.
        learn_weights: Whether to learn the fusion weights.
    """

    def __init__(
        self,
        backbone_names: List[str],
        num_classes: int,
        learn_weights: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_names = backbone_names
        self.num_classes = num_classes
        self.learn_weights = learn_weights

        n = len(backbone_names)
        if learn_weights:
            # Raw (pre-softmax) weights — initialised to uniform
            self.raw_weights = nn.Parameter(torch.zeros(n))
        else:
            self.register_buffer("raw_weights", torch.zeros(n))

    def _normalised_weights(self) -> torch.Tensor:
        """Return softmax-normalised fusion weights."""
        return F.softmax(self.raw_weights, dim=0)

    def forward(
        self,
        logits_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute fused class probabilities from per-backbone logits.

        Args:
            logits_dict: Mapping from backbone name to logit tensor of
                shape (N, num_classes).

        Returns:
            Fused probability tensor of shape (N, num_classes).
        """
        weights = self._normalised_weights()
        fused: Optional[torch.Tensor] = None

        for i, name in enumerate(self.backbone_names):
            if name not in logits_dict:
                raise KeyError(f"Missing logits for backbone '{name}'.")
            probs = F.softmax(logits_dict[name].float(), dim=-1)
            weighted = weights[i] * probs
            fused = weighted if fused is None else fused + weighted

        return fused  # type: ignore[return-value]

    def predict(
        self,
        logits_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return fused probabilities and per-backbone probabilities.

        Args:
            logits_dict: Mapping from backbone name to logit tensor.

        Returns:
            Dictionary with keys:
              - ``"fused"``: Fused probability tensor (N, num_classes).
              - ``"<backbone_name>"``: Per-backbone softmax probabilities.
              - ``"weights"``: Current fusion weights (n_backbones,).
        """
        result: Dict[str, torch.Tensor] = {}
        for name, logits in logits_dict.items():
            result[name] = F.softmax(logits.float(), dim=-1)

        result["fused"] = self.forward(logits_dict)
        result["weights"] = self._normalised_weights().detach()
        return result

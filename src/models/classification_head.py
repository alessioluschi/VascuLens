"""Shared classification head architecture used by all backbones."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """Two-layer MLP classification head.

    Architecture::

        Linear(input_dim → hidden_dim) → Activation → Dropout → Linear(hidden_dim → num_classes)

    Args:
        input_dim: Dimensionality of the input embedding.
        hidden_dim: Number of hidden units.
        num_classes: Number of output classes.
        dropout: Dropout probability.
        activation: Activation function name: ``"relu"`` or ``"gelu"``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if activation.lower() == "relu":
            act_fn: nn.Module = nn.ReLU(inplace=True)
        elif activation.lower() == "gelu":
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: '{activation}'. Use 'relu' or 'gelu'.")

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        logger.debug(
            f"ClassificationHead: {input_dim} → {hidden_dim} ({activation}) "
            f"→ dropout({dropout}) → {num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classification head.

        Args:
            x: Input embedding tensor of shape (N, input_dim).

        Returns:
            Logit tensor of shape (N, num_classes).
        """
        return self.net(x)

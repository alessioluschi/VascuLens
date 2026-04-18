"""Reproducibility utilities — set all random seeds."""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, cudnn_benchmark: bool = False) -> None:
    """Set all random seeds for full reproducibility.

    Args:
        seed: The integer seed value.
        cudnn_benchmark: Value for torch.backends.cudnn.benchmark.
            Set to False for reproducibility, True for speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = cudnn_benchmark
    logger.info(f"Random seed set to {seed} | cudnn.benchmark={cudnn_benchmark}")

"""Configuration loader and hardware validator for the Ulcer Classification Pipeline."""

import logging
import os
from pathlib import Path

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Required top-level config sections
_REQUIRED_SECTIONS = [
    "general",
    "hardware",
    "data",
    "cross_validation",
    "backbones",
    "classification_head",
    "fusion",
    "training",
    "evaluation",
    "explainability",
]


def load_config(config_path: str) -> DictConfig:
    """Load, validate, and return the project configuration.

    Reads config.yaml, resolves relative paths against the project root,
    validates hardware constraints (VRAM, FP16 support), and prints a
    startup summary.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        OmegaConf DictConfig supporting dot-access (cfg.training.epochs).

    Raises:
        FileNotFoundError: If config_path does not exist.
        ValueError: If required config sections are missing.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg: DictConfig = OmegaConf.create(raw)

    # --- Validate required sections ---
    missing = [s for s in _REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"Config is missing required sections: {missing}")

    # --- Resolve relative paths against project root ---
    project_root = config_path.parent
    cfg.general.output_dir = str(project_root / cfg.general.output_dir)
    cfg.hardware.embedding_cache_dir = str(project_root / cfg.hardware.embedding_cache_dir)
    cfg.data.root_dir = str(project_root / cfg.data.root_dir)
    cfg.explainability.output_dir = str(project_root / cfg.explainability.output_dir)

    if cfg.data.colour_normalization.reference_image is not None:
        cfg.data.colour_normalization.reference_image = str(
            project_root / cfg.data.colour_normalization.reference_image
        )

    # --- Validate hardware ---
    _validate_hardware(cfg)

    # --- Validate HuggingFace token (only when UNI is enabled) ---
    _validate_huggingface(cfg)

    # --- Print startup summary ---
    _print_config_summary(cfg)

    return cfg


def _validate_hardware(cfg: DictConfig) -> None:
    """Validate GPU availability and mixed-precision support.

    Args:
        cfg: The loaded configuration (mutated in place for fallbacks).
    """
    if not torch.cuda.is_available():
        if cfg.general.device == "cuda":
            logger.warning(
                "CUDA not available but device='cuda' in config. Falling back to CPU."
            )
            OmegaConf.update(cfg, "general.device", "cpu", merge=True)
        return

    props = torch.cuda.get_device_properties(0)
    actual_vram_gb = props.total_memory / (1024 ** 3)
    configured_vram_gb = cfg.hardware.vram_gb

    if actual_vram_gb < configured_vram_gb:
        logger.warning(
            f"Actual VRAM ({actual_vram_gb:.1f} GB) < configured VRAM "
            f"({configured_vram_gb} GB). Consider reducing training.batch_size."
        )

    # FP16 requires compute capability >= 7.0
    if cfg.hardware.mixed_precision and props.major < 7:
        logger.warning(
            f"GPU compute capability {props.major}.{props.minor} < 7.0 — "
            f"FP16 not fully supported. Disabling mixed_precision."
        )
        OmegaConf.update(cfg, "hardware.mixed_precision", False, merge=True)


def _validate_huggingface(cfg: DictConfig) -> None:
    """Warn early if UNI is enabled but no HuggingFace token is available.

    Checks ``huggingface.token`` in the config and the ``HF_TOKEN``
    environment variable.  Does not raise — the error will surface later
    when timm tries to download the gated model.

    Args:
        cfg: The validated configuration.
    """
    if not cfg.backbones.uni.enabled:
        return

    token = OmegaConf.select(cfg, "huggingface.token") or os.environ.get("HF_TOKEN")
    if not token:
        logger.warning(
            "backbones.uni.enabled=true but no HuggingFace token found. "
            "Set huggingface.token in config.yaml or the HF_TOKEN environment variable, "
            "otherwise UNI model loading will fail."
        )


def _print_config_summary(cfg: DictConfig) -> None:
    """Print a human-readable startup summary of the loaded configuration.

    Args:
        cfg: The validated configuration.
    """
    print("\n" + "=" * 60)
    print(" Ulcer Classification Pipeline — Config Summary")
    print("=" * 60)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_mem_gb = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
        total_mem_gb = props.total_memory / (1024 ** 3)
        print(f"GPU:                {props.name}")
        print(f"VRAM:               {total_mem_gb:.1f} GB ({free_mem_gb:.1f} GB available)")
    else:
        print("GPU:                Not available (CPU mode)")

    mp = cfg.hardware.mixed_precision
    uni_gc = (
        cfg.backbones.uni.gradient_checkpointing
        if cfg.backbones.uni.enabled
        else False
    )
    print(f"Mixed Precision:    {'Enabled' if mp else 'Disabled'}")
    print(f"Gradient Ckpt:      {'Enabled (UNI)' if uni_gc else 'Disabled'}")
    print(
        f"Sequential Train:   "
        f"{'Enabled' if cfg.hardware.sequential_backbone_training else 'Disabled'}"
    )
    print(f"Output dir:         {cfg.general.output_dir}")
    print(f"Data dir:           {cfg.data.root_dir}")
    print(f"CV folds:           {cfg.cross_validation.n_splits}")
    print(f"Epochs:             {cfg.training.epochs}")
    print(f"Batch size:         {cfg.training.batch_size}")

    if cfg.backbones.uni.enabled:
        hf_token = OmegaConf.select(cfg, "huggingface.token") or os.environ.get("HF_TOKEN")
        token_status = "Configured" if hf_token else "NOT SET — UNI will fail"
        print(f"HF Token (UNI):     {token_status}")

    print("=" * 60 + "\n")

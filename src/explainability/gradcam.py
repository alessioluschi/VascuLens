"""GradCAM++ heatmap generation for EfficientNet-B0.

Loads the EfficientNet checkpoint into VRAM, generates heatmaps for all
selected samples, then releases the model.  Uses gpu_memory_guard.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models.backbone_efficientnet import EfficientNetBackbone
from src.models.memory import gpu_memory_guard, release_model

logger = logging.getLogger(__name__)


def _get_cam_class(method: str):
    """Return the pytorch-grad-cam class for the given method name.

    Args:
        method: One of ``"gradcam"``, ``"gradcam++"``, ``"scorecam"``.

    Returns:
        pytorch-grad-cam CAM class.
    """
    mapping = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "scorecam": ScoreCAM,
    }
    cls = mapping.get(method.lower())
    if cls is None:
        raise ValueError(f"Unknown CAM method '{method}'. Choose from: {list(mapping.keys())}")
    return cls


def _auto_detect_target_layer(model: EfficientNetBackbone) -> list:
    """Auto-detect the last convolutional layer of EfficientNet-B0.

    Args:
        model: EfficientNetBackbone instance.

    Returns:
        List containing the target layer module.
    """
    # timm EfficientNet-B0: last conv block is in backbone.blocks[-1][-1]
    try:
        target_layer = model.backbone.blocks[-1][-1]
        logger.info(f"Auto-detected GradCAM target layer: {type(target_layer).__name__}")
        return [target_layer]
    except (AttributeError, IndexError):
        logger.warning(
            "Could not auto-detect target layer. Falling back to last conv2d in backbone."
        )
        for module in reversed(list(model.backbone.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return [module]
        raise RuntimeError("No Conv2d layer found in EfficientNet backbone.")


def generate_gradcam_heatmaps(
    cfg: DictConfig,
    checkpoint_path: str,
    image_paths: List[str],
    class_names: List[str],
) -> Dict[str, np.ndarray]:
    """Load EfficientNet, generate GradCAM++ heatmaps, then release.

    Args:
        cfg: Full OmegaConf configuration.
        checkpoint_path: Path to the best EfficientNet checkpoint.
        image_paths: List of image paths to generate heatmaps for.
        class_names: Ordered class names (used for target class selection).

    Returns:
        Dictionary mapping image path to heatmap array (H, W) in [0, 1].
    """
    device = cfg.general.device
    gcfg = cfg.explainability.gradcam
    method = gcfg.method
    cam_cls = _get_cam_class(method)

    heatmaps: Dict[str, np.ndarray] = {}

    with gpu_memory_guard(f"GradCAM++ ({method}) — EfficientNet"):
        model = EfficientNetBackbone(cfg).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Determine target layer
        if gcfg.target_layer == "auto":
            target_layers = _auto_detect_target_layer(model)
        else:
            target_layers = [dict(model.backbone.named_modules())[gcfg.target_layer]]

        cam = cam_cls(model=model, target_layers=target_layers)

        from src.data.augmentation import get_val_transforms
        transform = get_val_transforms(cfg, "efficientnet")

        for img_path in image_paths:
            try:
                pil_img = Image.open(img_path).convert("RGB")
                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.cuda.amp.autocast(enabled=device == "cuda"):
                    targets = None  # Use top predicted class
                    grayscale_cam = cam(
                        input_tensor=input_tensor,
                        targets=targets,
                    )

                heatmaps[img_path] = grayscale_cam[0]  # (H, W) in [0, 1]
                logger.debug(f"GradCAM++ heatmap generated for {Path(img_path).name}")

            except Exception as exc:
                logger.warning(f"GradCAM++ failed for {img_path}: {exc}")

        del cam
        release_model(model)

    logger.info(f"Generated {len(heatmaps)} GradCAM++ heatmaps.")
    return heatmaps

"""Heatmap overlay and explanation report generation."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.5,
) -> np.ndarray:
    """Apply colormap to heatmap and blend with the original image.

    Args:
        image: Original RGB image as uint8 array (H, W, 3).
        heatmap: Grayscale heatmap in [0, 1] of shape (H, W).
        colormap: Matplotlib colormap name (e.g. ``"jet"``).
        alpha: Overlay transparency (0 = original only, 1 = heatmap only).

    Returns:
        Blended overlay as uint8 array (H, W, 3).
    """
    # Resize heatmap to match image if needed
    h, w = image.shape[:2]
    if heatmap.shape != (h, w):
        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (w, h), Image.BILINEAR
        )
        heatmap = np.array(heatmap_pil, dtype=np.float32) / 255.0

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)  # (H, W, 3)

    # Blend
    overlay = (image * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    return overlay


def save_side_by_side(
    image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    save_path: str,
    title: str = "",
    colormap: str = "jet",
) -> None:
    """Save a side-by-side figure: original | heatmap | overlay.

    Args:
        image: Original RGB image (H, W, 3).
        heatmap: Grayscale heatmap in [0, 1] (H, W).
        overlay: Blended overlay image (H, W, 3).
        save_path: Output file path.
        title: Figure title.
        colormap: Colormap for heatmap display.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap=colormap, vmin=0, vmax=1)
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_explanation_report(
    image_path: str,
    predictions: Dict[str, float],
    heatmaps_per_backbone: Dict[str, np.ndarray],
    output_dir: str,
    colormap: str = "jet",
    alpha: float = 0.5,
) -> None:
    """Create and save a multi-row explanation figure for one image.

    Produces one row per backbone showing:
        original | heatmap | overlay

    The figure is annotated with predicted class, confidence, and backbone name.

    Args:
        image_path: Path to the source image.
        predictions: Mapping from backbone name to predicted class probability.
        heatmaps_per_backbone: Mapping from backbone name to heatmap array.
        output_dir: Directory to save the report figure.
        colormap: Matplotlib colormap.
        alpha: Overlay transparency.
    """
    backbone_names = list(heatmaps_per_backbone.keys())
    n_rows = len(backbone_names)

    if n_rows == 0:
        logger.warning(f"No heatmaps available for {image_path}. Skipping report.")
        return

    img_pil = Image.open(image_path).convert("RGB")
    img_arr = np.array(img_pil, dtype=np.uint8)

    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing

    for row_idx, bname in enumerate(backbone_names):
        heatmap = heatmaps_per_backbone[bname]
        overlay = overlay_heatmap(img_arr, heatmap, colormap=colormap, alpha=alpha)

        pred_str = f"{predictions.get(bname, float('nan')):.3f}"
        row_label = f"{bname}  (conf={pred_str})"

        axes[row_idx, 0].imshow(img_arr)
        axes[row_idx, 0].set_title(f"Original\n[{bname}]")
        axes[row_idx, 0].set_ylabel(row_label, fontsize=9)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(heatmap, cmap=colormap, vmin=0, vmax=1)
        axes[row_idx, 1].set_title("Heatmap")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title("Overlay")
        axes[row_idx, 2].axis("off")

    stem = Path(image_path).stem
    fig.suptitle(f"Explanation Report — {stem}", fontsize=13)
    fig.tight_layout()

    out_dir = Path(output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "combined.png"
    fig.savefig(report_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Explanation report saved → {report_path}")

    # Also save individual backbone heatmaps
    img_save_path = out_dir / "original.png"
    img_pil.save(img_save_path)

    for bname, heatmap in heatmaps_per_backbone.items():
        overlay = overlay_heatmap(img_arr, heatmap, colormap=colormap, alpha=alpha)
        if bname == "efficientnet":
            fname = out_dir / "efficientnet_gradcam.png"
        elif bname == "biomedclip":
            fname = out_dir / "biomedclip_attention.png"
        elif bname == "uni":
            fname = out_dir / "uni_attention.png"
        else:
            fname = out_dir / f"{bname}_heatmap.png"

        save_side_by_side(
            image=img_arr,
            heatmap=heatmap,
            overlay=overlay,
            save_path=str(fname),
            title=f"{bname} — {Path(image_path).name}",
            colormap=colormap,
        )

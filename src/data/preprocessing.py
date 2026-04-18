"""Colour normalization for histological / clinical ulcer images.

Supports Macenko and Reinhard stain normalization.  Uses ``staintools``
when available and falls back to a pure NumPy implementation otherwise.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import staintools; fall back to pure NumPy implementations.
# ---------------------------------------------------------------------------
try:
    import staintools  # type: ignore

    _HAS_STAINTOOLS = True
    logger.info("staintools found — using library implementations.")
except ImportError:
    _HAS_STAINTOOLS = False
    logger.warning(
        "staintools not found. Falling back to pure NumPy implementations of "
        "Macenko / Reinhard normalization."
    )


# ===========================================================================
# Public API
# ===========================================================================


def normalize_and_cache(
    image_paths: List[Path],
    method: str,
    reference_image_path: Optional[str],
    cache_dir: str,
) -> List[Path]:
    """Apply colour normalization to all images and save results to cache_dir.

    If a reference image path is not provided, the image with median
    brightness in the dataset is selected automatically.

    Args:
        image_paths: List of source image paths.
        method: Normalization method: ``"macenko"`` or ``"reinhard"``.
        reference_image_path: Path to the reference image, or ``None``.
        cache_dir: Directory where normalized images will be saved.

    Returns:
        List of paths to the normalized (cached) images, parallel to image_paths.
        If an image fails normalization the original path is returned in its place.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Build or load the normalizer
    normalizer = _build_normalizer(
        method=method,
        image_paths=image_paths,
        reference_image_path=reference_image_path,
    )

    normalized_paths: List[Path] = []
    for src in image_paths:
        dst = cache_path / src.name
        if dst.exists():
            normalized_paths.append(dst)
            continue

        try:
            img_rgb = _load_rgb(src)
            norm_rgb = normalizer.transform(img_rgb)
            Image.fromarray(norm_rgb).save(dst)
            normalized_paths.append(dst)
        except Exception as exc:
            logger.warning(
                f"Normalization failed for {src.name}: {exc}. Using original."
            )
            normalized_paths.append(src)

    logger.info(
        f"Colour normalization ({method}) complete. "
        f"Cached to {cache_dir}."
    )
    return normalized_paths


# ===========================================================================
# Internal helpers
# ===========================================================================


def _load_rgb(path: Path) -> np.ndarray:
    """Load an image as a uint8 RGB NumPy array."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _select_reference(image_paths: List[Path]) -> Path:
    """Return the image with median brightness as the normalization reference."""
    brightnesses: List[float] = []
    for p in image_paths:
        try:
            arr = _load_rgb(p)
            brightnesses.append(float(np.mean(arr)))
        except Exception:
            brightnesses.append(0.0)

    median_val = float(np.median(brightnesses))
    diffs = [abs(b - median_val) for b in brightnesses]
    ref_path = image_paths[int(np.argmin(diffs))]
    logger.info(f"Auto-selected reference image: {ref_path.name} (brightness ≈ {median_val:.1f})")
    return ref_path


def _build_normalizer(
    method: str,
    image_paths: List[Path],
    reference_image_path: Optional[str],
):
    """Construct and fit a normalizer from the specified method and reference.

    Args:
        method: ``"macenko"`` or ``"reinhard"``.
        image_paths: All dataset image paths (used only if reference is None).
        reference_image_path: Explicit reference image path or None.

    Returns:
        A fitted normalizer object with a ``.transform(img_rgb)`` method.
    """
    if reference_image_path is not None:
        ref_path = Path(reference_image_path)
    else:
        ref_path = _select_reference(image_paths)

    ref_img = _load_rgb(ref_path)

    if _HAS_STAINTOOLS:
        return _StainToolsNormalizer(method=method, reference=ref_img)
    else:
        if method == "macenko":
            return _NumpyMacenkoNormalizer(reference=ref_img)
        else:
            return _NumpyReinhardNormalizer(reference=ref_img)


# ---------------------------------------------------------------------------
# staintools wrapper
# ---------------------------------------------------------------------------


class _StainToolsNormalizer:
    """Thin wrapper around staintools normalizers."""

    def __init__(self, method: str, reference: np.ndarray) -> None:
        if method == "macenko":
            self._norm = staintools.MacenkoStainNormalizer()
        else:
            self._norm = staintools.ReinhardColorNormalizer()

        self._norm.fit(reference)

    def transform(self, img: np.ndarray) -> np.ndarray:
        """Normalize img to match the reference stain."""
        return self._norm.transform(img)


# ---------------------------------------------------------------------------
# Pure NumPy fallback — Macenko (simplified SVD-based approach)
# ---------------------------------------------------------------------------


class _NumpyMacenkoNormalizer:
    """Simplified Macenko stain normalization (NumPy-only fallback)."""

    _Io: float = 240.0
    _beta: float = 0.15
    _alpha: float = 1.0

    def __init__(self, reference: np.ndarray) -> None:
        self._stain_matrix = self._get_stain_matrix(reference)

    def transform(self, img: np.ndarray) -> np.ndarray:
        """Normalize stain of img to the reference."""
        try:
            stain_mat_src = self._get_stain_matrix(img)
            concentrations = self._get_concentrations(img, stain_mat_src)
            return self._reconstruct(concentrations, self._stain_matrix, img.shape)
        except Exception as exc:
            logger.debug(f"Macenko transform failed: {exc}")
            raise

    # -- internal ------------------------------------------------------------

    def _od(self, img: np.ndarray) -> np.ndarray:
        """Convert RGB to optical density."""
        img = img.astype(np.float64)
        img = np.clip(img, 1, 254)
        return -np.log(img / self._Io)

    def _get_stain_matrix(self, img: np.ndarray) -> np.ndarray:
        od = self._od(img)
        od_hat = od.reshape(-1, 3)
        # Remove near-transparent pixels
        od_hat = od_hat[np.any(od_hat > self._beta, axis=1)]
        if od_hat.shape[0] < 6:
            return np.eye(2, 3)
        _, _, V = np.linalg.svd(od_hat, full_matrices=False)
        V = V[:2, :]
        # Project onto plane and find extreme angles
        proj = od_hat @ V.T
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        min_phi = np.percentile(phi, self._alpha)
        max_phi = np.percentile(phi, 100 - self._alpha)
        v1 = V[0] * np.cos(min_phi) + V[1] * np.sin(min_phi)
        v2 = V[0] * np.cos(max_phi) + V[1] * np.sin(max_phi)
        stain_matrix = np.array([v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)])
        return stain_matrix

    def _get_concentrations(self, img: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        od = self._od(img).reshape(-1, 3)
        conc, _ = np.linalg.lstsq(stain_matrix.T, od.T, rcond=None)
        return conc.T

    def _reconstruct(
        self, concentrations: np.ndarray, stain_matrix: np.ndarray, shape: tuple
    ) -> np.ndarray:
        od_norm = concentrations @ stain_matrix
        img_norm = self._Io * np.exp(-od_norm)
        img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)
        return img_norm.reshape(shape)


# ---------------------------------------------------------------------------
# Pure NumPy fallback — Reinhard colour transfer
# ---------------------------------------------------------------------------


class _NumpyReinhardNormalizer:
    """Reinhard colour-space transfer (NumPy-only fallback)."""

    def __init__(self, reference: np.ndarray) -> None:
        lab_ref = self._rgb2lab(reference)
        self._mean_ref = lab_ref.reshape(-1, 3).mean(axis=0)
        self._std_ref = lab_ref.reshape(-1, 3).std(axis=0) + 1e-6

    def transform(self, img: np.ndarray) -> np.ndarray:
        """Transfer colour statistics from reference to img."""
        lab = self._rgb2lab(img)
        mean_src = lab.reshape(-1, 3).mean(axis=0)
        std_src = lab.reshape(-1, 3).std(axis=0) + 1e-6
        lab_norm = (lab - mean_src) / std_src * self._std_ref + self._mean_ref
        return self._lab2rgb(lab_norm)

    # -- internal colour space conversions (simplified) ----------------------

    @staticmethod
    def _rgb2lab(img: np.ndarray) -> np.ndarray:
        """Very lightweight RGB → Lab approximation via PIL."""
        from PIL import Image as PilImage
        pil = PilImage.fromarray(img.astype(np.uint8)).convert("LAB")
        return np.array(pil, dtype=np.float32)

    @staticmethod
    def _lab2rgb(lab: np.ndarray) -> np.ndarray:
        from PIL import Image as PilImage
        lab_clipped = np.clip(lab, 0, 255).astype(np.uint8)
        pil = PilImage.fromarray(lab_clipped, mode="LAB").convert("RGB")
        return np.array(pil, dtype=np.uint8)

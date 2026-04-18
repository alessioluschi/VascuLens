"""PyTorch Dataset for binary ulcer image classification (VENOUS / NON-VENOUS)."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# Extensions accepted when scanning class subdirectories
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class UlcerDataset(Dataset):
    """Binary image dataset with folder structure root_dir/CLASS_NAME/*.{jpg,png}.

    Labels are assigned according to the position of the class name in
    ``class_names``: the first entry maps to label 0, the second to label 1.

    Args:
        root_dir: Root directory containing one subdirectory per class.
        transform: torchvision transform applied to each PIL image.
        class_names: Ordered list of class folder names, e.g. ["VENOUS", "NON-VENOUS"].

    Attributes:
        samples: List of (image_path, label) tuples.
        class_to_idx: Mapping from class name to integer label.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose],
        class_names: List[str],
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        self.samples: List[Tuple[Path, int]] = []
        self._scan_directory()
        self._log_class_distribution()

    def _scan_directory(self) -> None:
        """Scan root_dir for images in each class subdirectory."""
        for class_name, label in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(
                    f"Class directory not found: {class_dir}. "
                    f"No samples will be loaded for class '{class_name}'."
                )
                continue

            found = 0
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in _IMAGE_EXTENSIONS:
                    self.samples.append((path, label))
                    found += 1

            logger.info(f"Class '{class_name}' (label={label}): {found} images found.")

        if not self.samples:
            logger.warning(
                f"No images found in {self.root_dir}. "
                f"Expected subdirectories: {self.class_names}"
            )

    def _log_class_distribution(self) -> None:
        """Log the overall class distribution."""
        counts = {name: 0 for name in self.class_names}
        for _, label in self.samples:
            name = self.class_names[label]
            counts[name] += 1

        total = len(self.samples)
        dist_str = ", ".join(
            f"{name}: {cnt} ({100 * cnt / max(total, 1):.1f}%)"
            for name, cnt in counts.items()
        )
        logger.info(f"Dataset total: {total} images | {dist_str}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, str]:
        """Load and return a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label, image_path_str).
        """
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label, str(path)

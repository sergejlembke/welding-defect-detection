from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
cv2.setNumThreads(0)
import torch
from torch.utils.data import Dataset


# noinspection PyTypeChecker
def get_default_transforms(img_size: int = 224, is_train: bool = True) -> A.Compose:
    """Build default Albumentations transformation pipelines for training or evaluation.

    Args:
        img_size: Target image height and width in pixels.
        is_train: If True, applies data augmentations (flips, noise, brightness contrast).
            If False, applies only resizing and normalization.

    Returns:
        Composed Albumentations transformation pipeline.
    """
    if is_train:
        return A.Compose(  # type: ignore[return-value]
            [
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.GaussNoise(p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    return A.Compose(  # type: ignore[return-value]
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


class WeldingDefectDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Custom PyTorch Dataset for loading and processing welding seam inspection images."""

    def __init__(
        self,
        image_paths: list[str | Path],
        labels: list[int],
        transform: A.Compose | None = None,
    ) -> None:
        """Initialize dataset with image file paths, target labels, and augmentations.

        Args:
            image_paths: List of file paths pointing to inspection images.
            labels: List of binary classification labels (0 = OK, 1 = Defect).
            transform: Optional Albumentations transformation pipeline.
        """
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have equal length.")

        self.image_paths = [Path(p) for p in image_paths]
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform

    def __len__(self) -> int:
        """Return total number of dataset samples."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and transform an image sample by index.

        Args:
            index: Sample index.

        Returns:
            Tuple containing the transformed image tensor (C, H, W) and label tensor.
        """
        img_path = str(self.image_paths[index])
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Failed to load image at path: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0

        label = self.labels[index]
        return image, label
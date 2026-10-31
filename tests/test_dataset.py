from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pytest
import torch

from src.dataset import WeldingDefectDataset, get_default_transforms


def test_get_default_transforms() -> None:
    """Verify that transform generators return valid Albumentations Compose instances."""
    train_transform = get_default_transforms(img_size=224, is_train=True)
    val_transform = get_default_transforms(img_size=224, is_train=False)

    assert isinstance(train_transform, A.Compose)
    assert isinstance(val_transform, A.Compose)


def test_dataset_mismatched_length() -> None:
    """Verify that initializing dataset with mismatched paths and labels raises ValueError."""
    with pytest.raises(ValueError, match="must have equal length"):
        WeldingDefectDataset(image_paths=["path1.png"], labels=[0, 1])


def test_dataset_item_loading(tmp_path: Path) -> None:
    """Verify dataset item retrieval, transform application, and output tensor shapes."""
    # Generate a temporary dummy image file on disk
    img_path = tmp_path / "test_welding_seam.png"
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), dummy_img)

    transform = get_default_transforms(img_size=224, is_train=False)
    dataset = WeldingDefectDataset(
        image_paths=[img_path], labels=[1], transform=transform
    )

    assert len(dataset) == 1

    img_tensor, label_tensor = dataset[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(label_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 224, 224)
    assert label_tensor.item() == 1.0


def test_dataset_missing_file(tmp_path: Path) -> None:
    """Verify that loading a non-existent image file raises FileNotFoundError."""
    missing_path = tmp_path / "non_existent.png"
    dataset = WeldingDefectDataset(image_paths=[missing_path], labels=[0])

    with pytest.raises(FileNotFoundError, match="Failed to load image"):
        _ = dataset[0]
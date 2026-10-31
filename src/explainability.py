import argparse
from pathlib import Path

import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import torch.nn as nn

from src.dataset import get_default_transforms
from src.models import WeldingDefectClassifier


def get_target_layer(model: WeldingDefectClassifier) -> list[nn.Module]:
    """Identify and return the final convolutional target layer for Grad-CAM.

    Args:
        model: Instantiated classifier model.

    Returns:
        List containing target PyTorch layer module.
    """
    model_name = model.model_name.lower()

    if "resnet" in model_name:
        return [model.backbone.layer4[-1]]
    elif "convnext" in model_name:
        return [model.backbone.stages[-1]]
    elif "efficientnet" in model_name:
        return [model.backbone.conv_head]
    else:
        raise ValueError(
            f"Automatic target layer determination not implemented for '{model_name}'."
        )


def generate_gradcam_overlay(
    model: nn.Module,
    target_layers: list[nn.Module],
    image_tensor: torch.Tensor,
    original_image_rgb: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Compute Grad-CAM activation heatmap and blend over the input image.

    Args:
        model: Trained PyTorch evaluation model.
        target_layers: Convolutional layers targeted for activation maps.
        image_tensor: Preprocessed single image input tensor (C, H, W).
        original_image_rgb: Original resized image in RGB format (H, W, 3).
        device: Computation target device (CPU or CUDA).

    Returns:
        Tuple of blended RGB visualization overlay and predicted defect probability.
    """
    cam = GradCAM(model=model, target_layers=target_layers)
    input_tensor = image_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    image_float = original_image_rgb.astype(np.float32) / 255.0

    visualization = show_cam_on_image(
        image_float, grayscale_cam, use_rgb=True, image_weight=0.6
    )

    return visualization, prob


def run_explainability_pipeline(
    image_path: str | Path,
    model_path: str | Path | None = None,
    model_name: str = "resnet50",
    output_dir: str | Path = "outputs/gradcam",
    img_size: int = 224,
) -> None:
    """Run full explainability workflow on an image and export side-by-side comparison.

    Args:
        image_path: Path to target inspection image.
        model_path: Optional path to trained model weights checkpoint (.pt).
        model_name: timm backbone architecture identifier.
        output_dir: Directory where exported visualization images will be saved.
        img_size: Target resolution for model input.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WeldingDefectClassifier(model_name=model_name, freeze_backbone=False)
    if model_path and Path(model_path).exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device)
    target_layers = get_target_layer(model)

    cv_img = cv2.imread(str(image_path))
    if cv_img is None:
        raise FileNotFoundError(f"Failed to load image at: {image_path}")

    cv_img_resized = cv2.resize(cv_img, (img_size, img_size))
    rgb_img = cv2.cvtColor(cv_img_resized, cv2.COLOR_BGR2RGB)

    val_transform = get_default_transforms(img_size=img_size, is_train=False)
    transformed = val_transform(image=rgb_img)
    tensor_img = transformed["image"]

    overlay, prob = generate_gradcam_overlay(
        model, target_layers, tensor_img, rgb_img, device
    )

    comparison_image = np.hstack([rgb_img, overlay])
    save_filename = output_dir / f"gradcam_{image_path.stem}.png"
    cv2.imwrite(str(save_filename), cv2.cvtColor(comparison_image, cv2.COLOR_RGB2BGR))
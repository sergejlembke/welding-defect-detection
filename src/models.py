import timm
import torch
import torch.nn as nn


class WeldingDefectClassifier(nn.Module):
    """Flexible Computer Vision classifier wrapper around timm backbones."""

    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize classification model with optional layer freezing for linear probing.

        Args:
            model_name: Name of the timm architecture backbone.
            num_classes: Number of output units (1 for binary BCE loss).
            pretrained: Whether to load ImageNet pre-trained weights.
            freeze_backbone: If True, freezes feature extractor parameters (Linear Probing).
        """
        super().__init__()
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        if freeze_backbone:
            self._freeze_feature_extractor()

    def _freeze_feature_extractor(self) -> None:
        """Freeze all feature extractor layers while keeping classification head trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        head = self.backbone.get_classifier()
        if head is not None:
            for param in head.parameters():
                param.requires_grad = True
        else:
            raise AttributeError(
                f"Could not identify classification head for model '{self.model_name}'."
            )

    def set_freeze_backbone(self, freeze: bool) -> None:
        """Dynamically switch between frozen feature extractor and full fine-tuning.

        Args:
            freeze: If True, freezes backbone layers. If False, unfreezes all parameters.
        """
        self.freeze_backbone = freeze
        if freeze:
            self._freeze_feature_extractor()
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass through backbone network.

        Args:
            x: Input image tensor batch of shape (B, C, H, W).

        Returns:
            Raw logit output tensor of shape (B, num_classes).
        """
        return self.backbone(x)

    def get_parameter_summary(self) -> dict[str, int]:
        """Calculate and return total, trainable, and frozen parameter counts.

        Returns:
            Dictionary containing 'total_params', 'trainable_params', and 'frozen_params'.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
        }
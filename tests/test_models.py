import torch

from src.models import WeldingDefectClassifier


def test_linear_probing_parameter_freeze():
    model = WeldingDefectClassifier(model_name="resnet50", freeze_backbone=True)
    summary = model.get_parameter_summary()

    assert summary["trainable_params"] < summary["total_params"]
    assert summary["trainable_params"] > 0  # Nur Head ist trainierbar


def test_forward_pass_shape():
    model = WeldingDefectClassifier(model_name="resnet50", freeze_backbone=True)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    assert output.shape == (2, 1)
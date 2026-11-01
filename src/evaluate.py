import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    precision_recall_fscore_support,
)
import torch
from torch.utils.data import DataLoader

from src.dataset import WeldingDefectDataset, get_default_transforms
from src.models import WeldingDefectClassifier


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    """Plot and save confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["OK (0)", "Defect (1)"]
    )
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title("Confusion Matrix - Welding Defect Inspection")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Info] Saved confusion matrix plot to: {output_path}")


def plot_pr_curve(
    precisions: np.ndarray,
    recalls: np.ndarray,
    pr_auc: float,
    output_path: Path,
) -> None:
    """Plot and save Precision-Recall curve visualization."""
    plt.figure(figsize=(7, 5))
    plt.plot(
        recalls,
        precisions,
        color="#2b5c8f",
        lw=2,
        label=f"PR Curve (PR-AUC = {pr_auc:.4f})",
    )
    plt.xlabel("Recall (Defect Detection Rate)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Test Evaluation")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Info] Saved PR curve plot to: {output_path}")


def main() -> None:
    """Main evaluation script routine."""
    parser = argparse.ArgumentParser(
        description="Evaluation Script for Welding Defect Detection Model"
    )
    # Angepasster Default-Pfad passend zum neuen Namensschema
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/best_resnet50_finetune.pt",
        help="Path to trained model checkpoint file (z.B. outputs/best_resnet50_linear.pt oder outputs/best_resnet50_finetune.pt)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        help="timm backbone model architecture name",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/test.csv",
        help="Path to test dataset CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval",
        help="Directory to store evaluation plots and metrics.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Decision threshold for defect probability classification",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="DataLoader batch size"
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Input image dimension"
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Running evaluation on device: {device}")

    # 1. Load Test Metadata from CSV
    test_df = pd.read_csv(args.test_csv)
    test_paths = [Path(p) for p in test_df["image_path"]]
    test_labels = test_df["label"].to_numpy(dtype=np.int64)

    # 2. Dataset & DataLoader Setup
    test_transform = get_default_transforms(
        img_size=args.img_size, is_train=False
    )
    test_dataset = WeldingDefectDataset(
        test_paths, test_labels, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # 3. Initialize & Load Model Checkpoint
    model = WeldingDefectClassifier(
        model_name=args.model_name, freeze_backbone=False
    )

    checkpoint_path = Path(args.model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{args.model_path}'. "
            "Please run training or provide a valid --model_path."
        )

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Perform Inference on Test Dataset
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_targets.extend(labels.numpy())

    y_true = np.array(all_targets)
    y_probs = np.array(all_probs)
    y_pred = (y_probs >= args.threshold).astype(int)

    # 5. Compute Metrics & Artifacts
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    precisions_curve, recalls_curve, _ = precision_recall_curve(
        y_true, y_probs
    )
    pr_auc = auc(recalls_curve, precisions_curve)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "test_samples": int(len(y_true)),
        "decision_threshold": float(args.threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "pr_auc": float(pr_auc),
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
            "false_negatives": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
            "true_positives": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
        },
    }

    # Save Metrics JSON
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[Info] Saved evaluation metrics to: {metrics_file}")

    # 6. Plot & Save Visualizations
    plot_confusion_matrix(cm, output_path / "confusion_matrix.png")
    plot_pr_curve(
        precisions_curve,
        recalls_curve,
        float(pr_auc),
        output_path / "pr_curve.png",
    )

    print(
        f"\n[Success] Evaluation complete!\n"
        f"  PR-AUC: {pr_auc:.4f} | Recall: {recall:.4f} | "
        f"Precision: {precision:.4f} | F1: {f1:.4f}"
    )


if __name__ == "__main__":
    main()
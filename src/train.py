import os

from pathlib import Path
import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import WeldingDefectDataset, get_default_transforms
from src.models import WeldingDefectClassifier

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

def create_dataloaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader]:
    """Lädt train.csv und val.csv und erstellt PyTorch DataLoader."""
    project_root = Path.cwd()
    train_csv = project_root / "data/train.csv"
    val_csv = project_root / "data/val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            "data/train.csv oder data/val.csv nicht gefunden. "
            "Bitte zuerst prepare_yolo_dataset.py ausführen."
        )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    img_size = getattr(cfg.dataset, "img_size", 224)
    batch_size = getattr(cfg.dataset, "batch_size", 32)
    num_workers = getattr(cfg.dataset, "num_workers", 0)

    train_tf = get_default_transforms(img_size=img_size, is_train=True)
    val_tf = get_default_transforms(img_size=img_size, is_train=False)

    train_ds = WeldingDefectDataset(
        train_df["image_path"].tolist(),
        train_df["label"].tolist(),
        transform=train_tf,
    )
    val_ds = WeldingDefectDataset(
        val_df["image_path"].tolist(),
        val_df["label"].tolist(),
        transform=val_tf,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    global_step: int,
    log_interval: int = 10,
) -> tuple[float, int]:
    """Trainiert das Modell für eine Epoche."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train]",
        leave=True,
        unit="batch",
    )

    for batch_idx, (images, labels) in enumerate(pbar, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        global_step += 1

        current_avg_loss = running_loss / total_samples

        pbar.set_postfix(
            {
                "batch": f"{batch_idx}/{len(dataloader)}",
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{current_avg_loss:.4f}",
            }
        )

        # MLflow Schreibzugriffe reduzieren (NFS/SQLite entlasten)
        if global_step % log_interval == 0:
            mlflow.log_metric("train_batch_loss", loss.item(), step=global_step)
            mlflow.log_metric(
                "train_running_loss", current_avg_loss, step=global_step
            )

    epoch_loss = running_loss / total_samples
    return epoch_loss, global_step


def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluiert das Modell auf dem Validierungsdatensatz."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze(-1)
            loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            correct += (preds == labels).sum().item()
            total_samples += batch_size

    val_loss = running_loss / total_samples
    val_acc = correct / total_samples if total_samples > 0 else 0.0

    return val_loss, val_acc


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f"[Info] Starte Training mit Config:\n{cfg}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Nutze Device: {device}")

    # 1. Dateibasiertes MLflow-Tracking setzen
    tracking_uri = getattr(cfg.mlflow, "tracking_uri", "file:./mlruns")
    experiment_name = getattr(
        cfg.mlflow, "experiment_name", "Welding_Defect_Inspection"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # 2. DataLoader & Modell initialisieren
    train_loader, val_loader = create_dataloaders(cfg)

    model_name = getattr(
        cfg.model, "model_name", getattr(cfg.model, "name", "resnet50")
    )
    lr = getattr(cfg.model, "learning_rate", 1e-4)

    model = WeldingDefectClassifier(
        model_name=model_name, freeze_backbone=False
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pt"

    best_val_loss = float("inf")
    global_step = 0

    # 3. Training Loop
    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_name": model_name,
                "learning_rate": lr,
                "epochs": cfg.epochs,
                "device": str(device),
            }
        )

        for epoch in range(1, cfg.epochs + 1):
            train_loss, global_step = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                total_epochs=cfg.epochs,
                global_step=global_step,
            )

            val_loss, val_acc = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )

            print(
                f"Epoch {epoch:02d}/{cfg.epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            mlflow.log_metric("train_epoch_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            # Bestes Modell speichern
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"--> Bestes Modell gespeichert ({best_model_path}) mit Val Loss: {val_loss:.4f}"
                )


if __name__ == "__main__":
    main()
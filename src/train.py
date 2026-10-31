import mlflow
import torch
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    global_step: int,
) -> tuple[float, int]:
    """Train model for one epoch with tqdm console tracking and step-level MLflow logging.

    Args:
        model: PyTorch model to train.
        dataloader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Target execution device (cpu/cuda).
        epoch: Current epoch index (1-based).
        total_epochs: Total number of planned epochs.
        global_step: Accumulated step counter across all epochs.

    Returns:
        Tuple containing (average epoch loss, updated global step counter).
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    # Initialize tqdm progress bar for terminal output
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

        # 1. Update terminal progress bar with current step and loss values
        pbar.set_postfix(
            {
                "batch": f"{batch_idx}/{len(dataloader)}",
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{current_avg_loss:.4f}",
            }
        )

        # 2. Log batch metrics to MLflow for live UI step tracking
        mlflow.log_metric("train_batch_loss", loss.item(), step=global_step)
        mlflow.log_metric(
            "train_running_loss", current_avg_loss, step=global_step
        )

    epoch_loss = running_loss / total_samples
    return epoch_loss, global_step
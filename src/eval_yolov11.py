from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
from ultralytics import YOLO

# Import custom logger module
from src.utils.logger import setup_logger


def find_latest_checkpoint(runs_dir: Path) -> Path:
    """Locate the best weights file from the most recent training run directory.

    Args:
        runs_dir (Path): Base directory containing training output runs.

    Returns:
        Path: Path to the best.pt weights file, or fallback path.
    """
    if not runs_dir.exists():
        return Path("outputs/yolov11_runs/latest/weights/best.pt")

    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if run_dirs:
        latest_run = run_dirs[-1]
        best_weights = latest_run / "weights" / "best.pt"
        if best_weights.exists():
            return best_weights

    return Path("outputs/yolov11_runs/latest/weights/best.pt")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run model evaluation pipeline using Ultralytics YOLOv11 and log metrics to MLflow."""
    log_path = Path("outputs/logs/eval_yolov11.log")
    logger = setup_logger(log_path, logger_name="eval_yolov11")

    logger.info("Initializing YOLOv11 evaluation pipeline...")
    logger.info(f"Loaded Hydra Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Resolve evaluation weights path dynamically or from configuration
    custom_weights = cfg.get("eval_weights", None)
    if custom_weights:
        weights_path = Path(custom_weights)
    else:
        weights_path = find_latest_checkpoint(Path("outputs/yolov11_runs"))

    logger.info(f"Target weights file for evaluation: {weights_path}")

    if not weights_path.exists():
        logger.error(f"Weights file not found at: {weights_path}")
        raise FileNotFoundError(f"Weights file not found at specified path: {weights_path}")

    # Load YOLO model
    logger.info(f"Loading pretrained weights into YOLO model from {weights_path}...")
    model = YOLO(str(weights_path))

    # Configure MLflow tracking environment
    tracking_uri = cfg.mlflow.tracking_uri
    experiment_name = cfg.mlflow.experiment_name
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow Tracking URI: {tracking_uri} | Experiment: {experiment_name}")

    with mlflow.start_run(run_name="yolov11x_evaluation", nested=True):
        # Run validation split evaluation
        logger.info("Starting validation split evaluation...")
        metrics = model.val(
            data=cfg.dataset.data_yaml,
            imgsz=cfg.imgsz,
            batch=cfg.batch_size,
            split="val",
            project="outputs/yolov11_eval",
            name="latest_eval",
            exist_ok=True,
        )

        # Extract quantitative performance metrics
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)

        logger.info(
            f"Evaluation Metrics -> mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f} | "
            f"Precision: {precision:.4f} | Recall: {recall:.4f}"
        )

        # Log metrics and execution artifacts to MLflow
        mlflow.log_metrics(
            {
                "eval_mAP50": map50,
                "eval_mAP50_95": map50_95,
                "eval_precision": precision,
                "eval_recall": recall,
            }
        )
        mlflow.log_artifact(str(log_path))

    logger.info(f"Evaluation finished successfully. Execution log saved to {log_path}")


if __name__ == "__main__":
    main()
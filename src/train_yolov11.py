from datetime import datetime
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

# Import central logger module
from src.utils.logger import setup_logger


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Execute multi-GPU training of YOLOv11 for welding defect detection with MLflow logging."""
    log_path = Path("outputs/logs/train_yolov11.log")
    logger = setup_logger(log_path, logger_name="train_yolov11")

    logger.info("Initializing YOLOv11 multi-GPU training execution...")
    logger.info(f"Loaded Hydra Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Configure MLflow tracking server and experiment
    tracking_uri = cfg.mlflow.tracking_uri
    experiment_name = cfg.mlflow.experiment_name
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow Tracking URI: {tracking_uri} | Experiment: {experiment_name}")

    # Resolve target device specification for Distributed Data Parallel (DDP)
    device = list(cfg.device) if hasattr(cfg.device, "__iter__") and not isinstance(cfg.device, str) else cfg.device

    # Define unique run identifier and target project output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    project_dir = Path("outputs/yolov11_runs")

    logger.info(f"Loading pretrained architecture: {cfg.model.name}...")
    model = YOLO(cfg.model.name)

    # Extract hyperparameter dictionary from Hydra configuration and narrow type for PyCharm
    raw_hyperparams = OmegaConf.to_container(cfg.model.hyperparameters, resolve=True)
    if not isinstance(raw_hyperparams, dict):
        raise TypeError(f"Expected hyperparameters to be a dict, got {type(raw_hyperparams).__name__}")
    hyperparams: dict = raw_hyperparams

    logger.info(f"Starting execution for run '{run_name}' on target device(s): {device}...")

    with mlflow.start_run(run_name=f"yolov11x_training_{run_name}"):
        # Launch Ultralytics multi-GPU training process
        model.train(
            data=cfg.dataset.data_yaml,
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch_size,
            device=device,
            project=str(project_dir),
            name=run_name,
            exist_ok=False,
            plots=True,
            save=True,
            val=True,
            **hyperparams,
        )

        # Log training configuration parameters to MLflow
        mlflow.log_params(
            {
                "epochs": cfg.epochs,
                "imgsz": cfg.imgsz,
                "batch_size": cfg.batch_size,
                "device": str(device),
                "model_name": cfg.model.name,
                **hyperparams,
            }
        )

        # Log generated model artifacts and execution log to MLflow
        run_output_dir = project_dir / run_name
        if run_output_dir.exists():
            logger.info(f"Logging Ultralytics run artifacts from {run_output_dir} to MLflow...")
            mlflow.log_artifacts(str(run_output_dir), artifact_path="ultralytics_outputs")

        mlflow.log_artifact(str(log_path))

    logger.info(f"Training run '{run_name}' completed successfully. Execution log saved to {log_path}")


if __name__ == "__main__":
    # noinspection PyNoValue
    main()
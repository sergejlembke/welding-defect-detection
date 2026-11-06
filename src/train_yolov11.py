import os
from datetime import datetime
from pathlib import Path
import hydra
import mlflow
from omegaconf import DictConfig, ListConfig, OmegaConf
from ultralytics import YOLO

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # noinspection PyNoValueForParameter
    print(f"[INFO] Starting YOLOv11 multi-GPU training with config:\n{cfg}")

    project_root = Path.cwd()
    data_yaml_path = project_root / cfg.dataset.data_yaml

    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML file not found at: {data_yaml_path}"
        )

    # Setup MLflow tracking
    tracking_uri = getattr(
        cfg.mlflow, "tracking_uri", "sqlite:///outputs/mlflow.db"
    )
    experiment_name = getattr(
        cfg.mlflow, "experiment_name", "Welding_Defect_Inspection"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    model_name = cfg.model.name
    output_dir = project_root / "outputs" / "yolov11_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"

    device_arg = (
        list(cfg.device)
        if isinstance(cfg.device, (list, ListConfig))
        else cfg.device
    )

    # Extract hyperparameter dictionary from Hydra config
    hyperparams = (
        OmegaConf.to_container(cfg.model.hyperparameters, resolve=True)
        if hasattr(cfg.model, "hyperparameters")
        else {}
    )

    model = YOLO(model_name)

    with mlflow.start_run(run_name=f"yolo11_{run_name}"):
        mlflow.log_params(
            {
                "framework": "ultralytics_yolo11",
                "model_name": model_name,
                "epochs": cfg.epochs,
                "imgsz": cfg.imgsz,
                "batch_size": cfg.batch_size,
                "device": str(device_arg),
                **hyperparams,
            }
        )

        # Train model and return final metrics summary
        results = model.train(
            data=str(data_yaml_path),
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch_size,
            device=device_arg,
            project=str(output_dir),
            name=run_name,
            exist_ok=False,
            save=True,
            plots=True,
            **hyperparams,
        )

        # Log metrics to MLflow directly from results object
        if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
            for key, val in results.results_dict.items():
                if isinstance(val, (int, float)):
                    clean_key = (
                        f"final_{key}"
                        .replace("(", "_")
                        .replace(")", "_")
                        .replace("/", "_")
                    )
                    mlflow.log_metric(clean_key, float(val))

        best_weights = output_dir / run_name / "weights" / "best.pt"
        if best_weights.exists():
            mlflow.log_param("best_weight_path", str(best_weights))
            print(f"[SUCCESS] Best YOLOv11 checkpoint saved at: {best_weights}")


if __name__ == "__main__":
    # noinspection PyNoValueForParameter
    main()
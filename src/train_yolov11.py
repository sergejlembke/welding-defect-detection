import os
from pathlib import Path
import hydra
import mlflow
from omegaconf import DictConfig, ListConfig
from ultralytics import YOLO

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


@hydra.main(
    version_base=None, config_path="../configs", config_name="yolov11_config"
)
def main(cfg: DictConfig) -> None:
    print(f"[Info] Starte YOLOv11 Multi-GPU Training mit Config:\n{cfg}")

    project_root = Path.cwd()
    data_yaml_path = project_root / cfg.dataset.data_yaml

    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML nicht gefunden unter: {data_yaml_path}"
        )

    # MLflow-Tracking Setup
    tracking_uri = getattr(
        cfg.mlflow, "tracking_uri", "sqlite:///outputs/mlflow.db"
    )
    experiment_name = getattr(
        cfg.mlflow, "experiment_name", "Welding_Defect_Inspection"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    model_name = getattr(cfg.model, "name", "yolo11x.pt")
    output_dir = project_root / "outputs" / "yolov11_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device Parameter für Ultralytics Multi-GPU aufbereiten
    device_arg = (
        list(cfg.device)
        if isinstance(cfg.device, (list, ListConfig))
        else cfg.device
    )

    # YOLOv11 Model initialisieren (lädt Gewichte automatisch beim ersten Aufruf)
    model = YOLO(model_name)

    with mlflow.start_run(run_name=f"yolo11_{model_name.replace('.pt', '')}"):
        mlflow.log_params(
            {
                "framework": "ultralytics_yolo11",
                "model_name": model_name,
                "epochs": cfg.epochs,
                "imgsz": cfg.imgsz,
                "batch_size": cfg.batch_size,
                "device": str(device_arg),
                "data_yaml": str(data_yaml_path),
            }
        )

        results = model.train(
            data=str(data_yaml_path),
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch_size,
            device=device_arg,
            project=str(output_dir),
            name="run",
            exist_ok=True,
            save=True,
            plots=True,
        )

        # Metriken in MLflow protokollieren
        if hasattr(results, "results_dict"):
            for key, val in results.results_dict.items():
                if isinstance(val, (int, float)):
                    clean_key = (
                        key.replace("(", "_")
                        .replace(")", "_")
                        .replace("/", "_")
                    )
                    mlflow.log_metric(clean_key, float(val))

        best_weights = output_dir / "run" / "weights" / "best.pt"
        if best_weights.exists():
            mlflow.log_param("best_weight_path", str(best_weights))
            print(
                f"--> Bestes YOLOv11 Modell gespeichert unter: {best_weights}"
            )


if __name__ == "__main__":
    main()
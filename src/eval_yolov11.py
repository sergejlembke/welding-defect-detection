import os
from datetime import datetime
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
    print(f"[Info] Starte YOLOv11 Evaluierung mit Config:\n{cfg}")

    project_root = Path.cwd()
    yolo_runs_dir = project_root / "outputs" / "yolov11_runs"

    # Get weights of latest run dynamically
    if hasattr(cfg, "weights_path") and cfg.weights_path:
        best_weights = project_root / cfg.weights_path
        run_name_tag = best_weights.parent.parent.name
    else:
        all_runs = sorted(
            [d for d in yolo_runs_dir.glob("run_*") if d.is_dir()],
            key=os.path.getmtime,
            reverse=True,
        )

        if not all_runs:
            raise FileNotFoundError(
                f"No training runs (run_*) in {yolo_runs_dir} found."
                "Execute train_yolov11.py first."
            )

        latest_run = all_runs[0]
        best_weights = latest_run / "weights" / "best.pt"
        run_name_tag = latest_run.name

    if not best_weights.exists():
        raise FileNotFoundError(
            f"Model weights not found at: {best_weights}."
        )

    print(f"[Info] Use model weights from run: '{run_name_tag}' ({best_weights})")

    # Create unique eval folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_run_name = f"eval_{run_name_tag}_{timestamp}"
    output_dir = project_root / "outputs" / "yolov11_eval" / eval_run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images_dir = (
        project_root
        / "data/raw_welding_dataset/The Welding Defect Dataset - v2/The Welding Defect Dataset - v2/test/images"
    )

    if not test_images_dir.exists():
        raise FileNotFoundError(
            f"Test image directory not found at: {test_images_dir}"
        )

    # Load model with trained weights
    model = YOLO(str(best_weights))

    # MLflow-Tracking Setup
    tracking_uri = getattr(
        cfg.mlflow, "tracking_uri", "sqlite:///outputs/mlflow.db"
    )
    experiment_name = getattr(
        cfg.mlflow, "experiment_name", "Welding_Defect_Inspection"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    conf_threshold = getattr(cfg, "conf_threshold", 0.35)

    device_arg = (
        list(cfg.device)
        if isinstance(cfg.device, (list, ListConfig))
        else cfg.device
    )

    with mlflow.start_run(run_name=f"eval_{run_name_tag}"):
        mlflow.log_params(
            {
                "eval_weights": str(best_weights),
                "conf_threshold": conf_threshold,
                "eval_run_name": eval_run_name,
                "test_images_dir": str(test_images_dir),
            }
        )

        print(
            f"[Info] Generate Bounding-Box Visualizations (Conf Threshold: {conf_threshold})..."
        )
        model.predict(
            source=str(test_images_dir),
            conf=conf_threshold,
            save=True,
            project=str(output_dir),
            name="visualizations",
            exist_ok=True,
        )

        data_yaml_path = project_root / cfg.dataset.data_yaml
        if data_yaml_path.exists():
            print("[Info] Calculate test-metrics on official split...")
            metrics = model.val(
                data=str(data_yaml_path),
                split="test",
                conf=conf_threshold,
                device=device_arg,
                project=str(output_dir),
                name="val_metrics",
                exist_ok=True,
            )

            if hasattr(metrics, "results_dict"):
                for key, val in metrics.results_dict.items():
                    if isinstance(val, (int, float)):
                        clean_key = (
                            f"test_{key}"
                            .replace("(", "_")
                            .replace(")", "_")
                            .replace("/", "_")
                        )
                        mlflow.log_metric(clean_key, float(val))

        saved_path = output_dir / "visualizations"
        print(
            f"[Success] Evaluation completed! Results are saved at: {saved_path}"
        )


if __name__ == "__main__":
    main()
import os
from pathlib import Path
import hydra
import mlflow
from omegaconf import DictConfig
from ultralytics import YOLO

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


@hydra.main(
    version_base=None, config_path="../configs", config_name="yolov11_config"
)
def main(cfg: DictConfig) -> None:
    print(f"[Info] Starte YOLOv11 Evaluierung mit Config:\n{cfg}")

    project_root = Path.cwd()

    # 1. Gewichte und Pfade auflösen
    weights_path = getattr(
        cfg, "weights_path", "outputs/yolov11_runs/run/weights/best.pt"
    )
    best_weights = project_root / weights_path

    if not best_weights.exists():
        raise FileNotFoundError(
            f"Modellgewichte nicht gefunden unter: {best_weights}. "
            "Führe zuerst train_yolov11.py aus oder passe weights_path an."
        )

    test_images_dir = (
        project_root
        / "data/raw_welding_dataset/The Welding Defect Dataset - v2/The Welding Defect Dataset - v2/test/images"
    )

    if not test_images_dir.exists():
        raise FileNotFoundError(
            f"Testbild-Ordner nicht gefunden unter: {test_images_dir}"
        )

    output_dir = project_root / "outputs" / "yolov11_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Modell mit trainierten Gewichten laden
    print(f"[Info] Lade beste Gewichte aus: {best_weights}")
    model = YOLO(str(best_weights))

    # 3. MLflow Tracking Setup
    tracking_uri = getattr(
        cfg.mlflow, "tracking_uri", "sqlite:///outputs/mlflow.db"
    )
    experiment_name = getattr(
        cfg.mlflow, "experiment_name", "Welding_Defect_Inspection"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    conf_threshold = getattr(cfg, "conf_threshold", 0.25)

    with mlflow.start_run(run_name="yolov11_test_evaluation"):
        mlflow.log_params(
            {
                "eval_weights": str(best_weights),
                "conf_threshold": conf_threshold,
                "test_images_dir": str(test_images_dir),
            }
        )

        # 4. Inferenz auf den Testbildern & Bounding-Box Visualisierung speichern
        print(
            f"[Info] Generiere Bounding-Box Visualisierungen (Conf Threshold: {conf_threshold})..."
        )
        model.predict(
            source=str(test_images_dir),
            conf=conf_threshold,
            save=True,
            project=str(output_dir),
            name="visualizations",
            exist_ok=True,
        )

        # 5. quantitative Test-Metriken (mAP, Precision, Recall) berechnen
        data_yaml_path = project_root / cfg.dataset.data_yaml
        if data_yaml_path.exists():
            print("[Info] Berechne Test-Metriken auf dem offiziellen Split...")
            metrics = model.val(
                data=str(data_yaml_path),
                split="test",
                conf=conf_threshold,
                device=cfg.device,
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
            f"[Success] Evaluierung abgeschlossen! Bounding-Box-Bilder liegen unter: {saved_path}"
        )


if __name__ == "__main__":
    main()
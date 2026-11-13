from __future__ import annotations

import json
import os
from pathlib import Path
import inspect

import hydra
import mlflow
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from omegaconf import DictConfig
from ultralytics import YOLO

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class YoloV11WithFeatureMap(nn.Module):
    """Wrapper um das fusionierte YOLOv11x-Modell für Dual-Output (Detections + P4 Feature Map)."""

    def __init__(self, detection_model: nn.Module) -> None:
        super().__init__()
        self.model = detection_model
        self._p4_features: torch.Tensor | None = None

        # Detect-Head ist das letzte Modul; detect_layer.f enthält [P3_idx, P4_idx, P5_idx]
        detect_layer = self.model.model[-1]
        p4_layer_idx = detect_layer.f[1]  # P4 (Stride 16 -> 40x40 bei imgsz=640)
        self.model.model[p4_layer_idx].register_forward_hook(self._hook_p4)

    def _hook_p4(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self._p4_features = output

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        preds = self.model(x)
        detections = preds[0] if isinstance(preds, (tuple, list)) else preds
        if self._p4_features is None:
            raise RuntimeError("P4-Feature-Map konnte im Forward-Pass nicht abgegriffen werden.")
        return detections, self._p4_features


def resolve_weights_path(run_dir: Path) -> Path:
    """Findet best.pt im Run-Verzeichnis (unterstützt direkte & verschachtelte Ultralytics-Ordner)."""
    candidates = [
        run_dir / "weights" / "best.pt",
        *sorted(run_dir.glob("**/weights/best.pt")),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Keine 'best.pt' unter {run_dir} gefunden.")


def export_to_onnx(
    weights_path: Path,
    export_dir: Path,
    imgsz: int = 640,
    opset: int = 17,
    mlflow_tracking_uri: str | None = None,
) -> tuple[Path, Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / "yolov11x_weld_cam.onnx"
    meta_path = export_dir / "model_meta.json"

    logger.info(f"Lade Modell-Weights von: {weights_path}")
    yolo = YOLO(str(weights_path))

    # Klassen-Mapping direkt aus dem trainierten Modell auslesen ('Bad Weld', 'Good Weld', 'Defect')
    raw_names: dict[int, str] = yolo.names
    sorted_ids = sorted(raw_names.keys())
    normalized_names = [
        raw_names[i].strip().lower().replace(" ", "_") for i in sorted_ids
    ]
    logger.info(f"Erkanntes Klassen-Mapping: {dict(zip(sorted_ids, normalized_names))}")

    # Conv + BatchNorm fusionieren für schnellere CPU-Inferenz in ONNX Runtime
    yolo.fuse()
    pytorch_model = yolo.model.eval().cpu()
    wrapped_model = YoloV11WithFeatureMap(pytorch_model).eval()

    dummy_input = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float32)

    with torch.no_grad():
        ref_detections, ref_features = wrapped_model(dummy_input)

    logger.info(f"Exportiere ONNX (opset={opset}, imgsz={imgsz}) nach: {onnx_path}")
    export_kwargs: dict[str, object] = {
        "opset_version": opset,
        "input_names": ["images"],
        "output_names": ["detections", "feature_map"],
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(onnx_path),
        **export_kwargs,
    )

    # Numerischer Verifikationslauf mit ONNX Runtime
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_detections, ort_features = session.run(
        ["detections", "feature_map"], {"images": dummy_input.numpy()}
    )

    max_diff = float(np.max(np.abs(ref_detections.numpy() - ort_detections)))
    if not np.allclose(ref_detections.numpy(), ort_detections, atol=1e-3):
        raise RuntimeError(f"ONNX-Abweichung zu hoch (max_diff={max_diff:.6f})!")

    logger.info(
        f"ONNX-Verifikation erfolgreich | detections={ort_detections.shape}, "
        f"feature_map={ort_features.shape}, max_diff={max_diff:.2e}"
    )

    # Metadaten für den CNSD-Inferenz-Worker speichern
    meta = {
        "model_version": "yolov11x-weld-v1.0-onnx",
        "source_weights": str(weights_path),
        "input_size": imgsz,
        "opset": opset,
        "raw_class_names": {str(k): v for k, v in raw_names.items()},
        "normalized_class_names": normalized_names,
        "feature_map_shape": list(ort_features.shape),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Optionales Logging in deine bestehende MLflow SQLite-DB
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("welding-defect-onnx-export")
        with mlflow.start_run(run_name="export_yolov11x_onnx"):
            mlflow.log_params({"imgsz": imgsz, "opset": opset, "classes": normalized_names})
            mlflow.log_metric("onnx_max_abs_diff", max_diff)
            mlflow.log_artifact(str(onnx_path), artifact_path="onnx_bundle")
            mlflow.log_artifact(str(meta_path), artifact_path="onnx_bundle")
        logger.info(f"ONNX-Bundle in MLflow geloggt ({mlflow_tracking_uri}).")

    return onnx_path, meta_path


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = Path(cfg.get("run_dir", "outputs/yolov11_runs/run_20261106_124707"))
    export_dir = Path(cfg.get("export_dir", "outputs/onnx_export"))
    imgsz = int(cfg.get("imgsz", 640))
    mlflow_uri = cfg.get("mlflow", {}).get("tracking_uri", "sqlite:///outputs/mlflow.db")

    weights_path = resolve_weights_path(run_dir)
    export_to_onnx(
        weights_path=weights_path,
        export_dir=export_dir,
        imgsz=imgsz,
        mlflow_tracking_uri=mlflow_uri,
    )


if __name__ == "__main__":
    # Ermöglicht auch direkten Aufruf ohne Hydra-Override-Pflicht via CLI-Env-Vars
    if "HYDRA_DIRECT" in os.environ:
        export_to_onnx(
            weights_path=resolve_weights_path(
                Path(os.getenv("RUN_DIR", "outputs/yolov11_runs/run_20261106_124707"))
            ),
            export_dir=Path(os.getenv("EXPORT_DIR", "outputs/onnx_export")),
            imgsz=int(os.getenv("IMGSZ", "640")),
            mlflow_tracking_uri=os.getenv("MLFLOW_URI", "sqlite:///outputs/mlflow.db"),
        )
    else:
        main()
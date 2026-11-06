import os
from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import AsyncGenerator

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import torch
from ultralytics import YOLO


class ServiceSettings(BaseSettings):
    """Runtime configuration driven by environment variables."""

    model_path: str = ""  # If empty, automatically resolves latest run
    decision_threshold: float = 0.35
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = ServiceSettings()


def resolve_model_weights() -> Path:
    """Dynamically find the latest trained YOLOv11 checkpoint if not explicitly set."""
    if settings.model_path:
        return Path(settings.model_path)

    project_root = Path.cwd()
    yolo_runs_dir = project_root / "outputs" / "yolov11_runs"

    if yolo_runs_dir.exists():
        all_runs = sorted(
            [d for d in yolo_runs_dir.glob("run_*") if d.is_dir()],
            key=os.path.getmtime,
            reverse=True,
        )
        if all_runs:
            latest_weights = all_runs[0] / "weights" / "best.pt"
            if latest_weights.exists():
                return latest_weights

    # Fallback default location
    return project_root / "outputs" / "yolov11_runs" / "run" / "weights" / "best.pt"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager handling model loading and GPU memory cleanup."""
    weights_path = resolve_model_weights()

    try:
        if weights_path.exists():
            model = YOLO(str(weights_path))
            print(f"[INFO] Loaded YOLOv11 checkpoint from: {weights_path}")
        else:
            print(f"[WARNING] Checkpoint not found at {weights_path}. Loading default yolo11x.pt")
            model = YOLO("yolo11x.pt")

        app.state.model = model
        app.state.weights_path = str(weights_path)
        print(f"[SUCCESS] YOLOv11 inference service ready on device: {settings.device}")
    except Exception as e:
        app.state.model = None
        print(f"[ERROR] Failed to load YOLOv11 model: {e}")

    yield

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[INFO] Application shutdown: Cleared GPU memory cache.")


app = FastAPI(
    title="Welding Defect Inspection API",
    description="YOLOv11 Object Detection Service for automated welding seam quality control.",
    version="2.0.0",
    lifespan=lifespan,
)


class DetectionBox(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]  # Bounding box coordinates [x1, y1, x2, y2]


class PredictionResponse(BaseModel):
    filename: str
    is_defect: bool
    detected_objects_count: int
    detections: list[DetectionBox]
    decision_threshold: float
    latency_ms: float


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, str | float]:
    if getattr(app.state, "model", None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="YOLOv11 model is not loaded",
        )
    return {
        "status": "healthy",
        "device": settings.device,
        "active_weights": getattr(app.state, "weights_path", "unknown"),
        "decision_threshold": settings.decision_threshold,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_welding_seam(file: UploadFile = File(...)) -> PredictionResponse:
    model: YOLO | None = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized.",
        )

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Please upload JPEG or PNG.",
        )

    start_time = time.perf_counter()

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError("Failed to decode image byte stream.")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Execute object detection with configured threshold
        results = model.predict(
            source=img_rgb,
            conf=settings.decision_threshold,
            device=settings.device,
            verbose=False,
        )

        detections: list[DetectionBox] = []
        is_defect = False

        if results and len(results) > 0:
            boxes = results[0].boxes
            names = results[0].names

            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = str(names.get(cls_id, f"class_{cls_id}"))
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                if cls_name.lower() in ["defect", "bad weld", "bad_weld"]:
                    is_defect = True

                detections.append(
                    DetectionBox(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=round(conf, 4),
                        bbox=[round(coord, 2) for coord in xyxy],
                    )
                )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return PredictionResponse(
            filename=file.filename or "unknown",
            is_defect=is_defect,
            detected_objects_count=len(detections),
            detections=detections,
            decision_threshold=settings.decision_threshold,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )
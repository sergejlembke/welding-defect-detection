from contextlib import asynccontextmanager
import time
from typing import AsyncGenerator

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel
import torch

from src.dataset import get_default_transforms
from src.models import WeldingDefectClassifier

# Aktualisierter Pfad zum spezifischen Checkpoint
MODEL_PATH = "outputs/best_resnet50_finetune.pt"
MODEL_NAME = "resnet50"
DECISION_THRESHOLD = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_transform = get_default_transforms(img_size=224, is_train=False)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager handling application startup and shutdown events."""
    try:
        model = WeldingDefectClassifier(model_name=MODEL_NAME, freeze_backbone=False)
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"[Info] Checkpoint loaded from: {MODEL_PATH}")
        except FileNotFoundError:
            print("[Warning] Checkpoint not found. Running with ImageNet initialization.")

        model.to(device)
        model.eval()
        app.state.model = model
        print(f"[Success] Model initialized successfully on device: {device}")
    except Exception as e:
        app.state.model = None
        print(f"[Error] Failed to load model: {e}")

    yield

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Info] Application shutdown: Cleared GPU memory cache.")


app = FastAPI(
    title="Welding Defect Inspection API",
    description="Inference Service for automated welding seam quality control.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    filename: str
    is_defect: bool
    defect_probability: float
    decision_threshold: float
    latency_ms: float


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, str]:
    if getattr(app.state, "model", None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    return {"status": "healthy", "device": str(device)}


@app.post("/predict", response_model=PredictionResponse)
async def predict_welding_seam(file: UploadFile = File(...)) -> PredictionResponse:
    model: WeldingDefectClassifier | None = getattr(app.state, "model", None)
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
        transformed = val_transform(image=img_rgb)
        image_tensor = transformed["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probability = torch.sigmoid(output).item()

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        is_defect = probability >= DECISION_THRESHOLD

        return PredictionResponse(
            filename=file.filename or "unknown",
            is_defect=is_defect,
            defect_probability=round(probability, 4),
            decision_threshold=DECISION_THRESHOLD,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )
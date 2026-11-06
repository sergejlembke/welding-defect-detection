import io
from PIL import Image
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Verify that the health check endpoint returns status 200 and device info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "active_weights" in data


def test_predict_invalid_file_format() -> None:
    """Verify that non-image file uploads are rejected with HTTP 400 Bad Request."""
    files = {"file": ("test.txt", b"invalid text payload", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Invalid image format" in response.json()["detail"]


def test_predict_valid_image_payload() -> None:
    """Verify inference pipeline processes valid PNG image and returns standard response schema."""
    # Generate a dummy RGB image in memory
    img = Image.new("RGB", (224, 224), color="gray")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    files = {"file": ("test_welding_seam.png", img_bytes, "image/png")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_welding_seam.png"
    assert isinstance(data["is_defect"], bool)
    assert isinstance(data["detected_objects_count"], int)
    assert isinstance(data["detections"], list)
    assert "latency_ms" in data
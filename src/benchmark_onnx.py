from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_VAL_DIR = Path(
    "data/raw_welding_dataset/The Welding Defect Dataset - v2/"
    "The Welding Defect Dataset - v2/valid/images"
)


def generate_side_by_side_cam(
    image_bgr: np.ndarray,
    feature_map: np.ndarray,
    bbox_xyxy: list[int],
    input_size: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    alpha: float = 0.45,
) -> np.ndarray:
    """Erzeugt ein Side-by-Side-Bild (Links: Original-Crop | Rechts: Eigen-CAM-Overlay)."""
    xmin, ymin, xmax, ymax = bbox_xyxy
    crop_bgr = image_bgr[ymin:ymax, xmin:xmax]
    crop_h, crop_w = crop_bgr.shape[:2]

    lb_x1 = xmin * scale + pad_x
    lb_y1 = ymin * scale + pad_y
    lb_x2 = xmax * scale + pad_x
    lb_y2 = ymax * scale + pad_y

    _, channels, feat_h, feat_w = feature_map.shape
    fx1 = max(0, int(np.floor((lb_x1 / input_size) * feat_w)) - 1)
    fy1 = max(0, int(np.floor((lb_y1 / input_size) * feat_h)) - 1)
    fx2 = min(feat_w, int(np.ceil((lb_x2 / input_size) * feat_w)) + 1)
    fy2 = min(feat_h, int(np.ceil((lb_y2 / input_size) * feat_h)) + 1)

    if fx2 - fx1 < 2:
        fx1 = max(0, min(feat_w - 2, fx1))
        fx2 = fx1 + 2
    if fy2 - fy1 < 2:
        fy1 = max(0, min(feat_h - 2, fy1))
        fy2 = fy1 + 2

    local_feat = feature_map[0, :, fy1:fy2, fx1:fx2]
    sub_h, sub_w = local_feat.shape[1], local_feat.shape[2]
    flat = local_feat.reshape(channels, sub_h * sub_w).astype(np.float32)
    flat -= flat.mean(axis=1, keepdims=True)

    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    cam = vt[0].reshape(sub_h, sub_w)

    if np.sum(cam * flat.mean(axis=0).reshape(sub_h, sub_w)) < 0:
        cam = -cam

    cam = np.maximum(cam, 0.0)
    max_val = float(cam.max())
    if max_val > 1e-8:
        cam = cam / max_val

    cam_resized = cv2.resize(cam, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(np.clip(cam_resized * 255.0, 0, 255))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay_bgr = cv2.addWeighted(crop_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    return np.hstack([crop_bgr, overlay_bgr])


def preprocess_letterbox(
    image_bgr: np.ndarray, input_size: int
) -> tuple[np.ndarray, float, int, int]:
    orig_h, orig_w = image_bgr.shape[:2]
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))

    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2

    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(tensor), scale, pad_x, pad_y


def resolve_val_dir(val_dir: Path) -> Path:
    if val_dir.is_dir():
        return val_dir
    candidates = list(Path("data").rglob("valid/images")) + list(Path("data").rglob("val/images"))
    if not candidates:
        raise FileNotFoundError(f"Validierungsordner nicht gefunden: {val_dir}")
    return candidates[0]


def run_benchmark(
    export_dir: Path = Path("outputs/onnx_export"),
    val_dir: Path = DEFAULT_VAL_DIR,
    out_dir: Path = Path("outputs/cam_preview"),
    min_confidence: float = 0.40,
    auto_pass_confidence: float = 0.75,  # Empirisch auf Validierungs-Scores kalibriert
    max_images: int = 25,
    clean_old_previews: bool = True,
) -> None:
    resolved_val_dir = resolve_val_dir(val_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if clean_old_previews:
        for old_file in out_dir.glob("*.png"):
            old_file.unlink()

    meta = json.loads((export_dir / "model_meta.json").read_text(encoding="utf-8"))
    class_names: list[str] = meta["normalized_class_names"]
    input_size: int = int(meta["input_size"])

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(export_dir / "yolov11x_weld_cam.onnx"),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    # Warmup-Run für stabile CPU-Latenzmessung
    session.run(
        ["detections", "feature_map"],
        {input_name: np.zeros((1, 3, input_size, input_size), dtype=np.float32)},
    )

    image_paths = sorted(
        [p for p in resolved_val_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )[:max_images]

    latencies_ms: list[float] = []
    triage_counts = {"defects": 0, "manual-review": 0, "auto_passed": 0}

    for img_path in image_paths:
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        orig_h, orig_w = image_bgr.shape[:2]

        t0 = time.perf_counter()
        tensor, scale, pad_x, pad_y = preprocess_letterbox(image_bgr, input_size)
        detections, feature_map = session.run(
            ["detections", "feature_map"], {input_name: tensor}
        )
        predictions = np.squeeze(detections).T

        boxes_xywh, confidences, class_ids = [], [], []
        for row in predictions:
            scores = row[4:]
            cls_id = int(np.argmax(scores))
            score = float(scores[cls_id])
            if score < min_confidence:
                continue
            cx, cy, w, h = row[:4]
            xmin = (cx - w / 2.0 - pad_x) / scale
            ymin = (cy - h / 2.0 - pad_y) / scale
            boxes_xywh.append([float(xmin), float(ymin), float(w / scale), float(h / scale)])
            confidences.append(score)
            class_ids.append(cls_id)

        indices = (
            cv2.dnn.NMSBoxes(boxes_xywh, confidences, min_confidence, 0.45)
            if boxes_xywh
            else []
        )
        flat_indices = np.array(indices).flatten()

        img_summary: list[str] = []
        for idx_num, raw_idx in enumerate(flat_indices, start=1):
            x, y, w, h = boxes_xywh[raw_idx]
            bbox = [
                max(0, min(orig_w - 1, int(round(x)))),
                max(0, min(orig_h - 1, int(round(y)))),
                max(1, min(orig_w, int(round(x + w)))),
                max(1, min(orig_h, int(round(y + h)))),
            ]
            label = class_names[class_ids[raw_idx]]
            conf = confidences[raw_idx]

            if label in ("bad_weld", "defect"):
                bucket = "defects"
            elif label == "good_weld" and conf < auto_pass_confidence:
                bucket = "manual-review"
            else:
                bucket = "auto_passed"

            triage_counts[bucket] += 1
            img_summary.append(f"{label}({conf:.2f}->{bucket})")

            if bucket in ("defects", "manual-review"):
                side_by_side = generate_side_by_side_cam(
                    image_bgr, feature_map, bbox, input_size, scale, pad_x, pad_y
                )
                out_file = out_dir / f"{bucket}_{img_path.stem}_#{idx_num}_{label}_{conf:.2f}.png"
                cv2.imwrite(str(out_file), side_by_side)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        details = ", ".join(img_summary) if img_summary else "keine Treffer >= min_confidence"
        print(f"{img_path.name[:42]:<42} | {elapsed_ms:6.1f} ms | {details}")

    if latencies_ms:
        print("\n--- Benchmark- & Triage-Zusammenfassung ---")
        print(f"Getestete Bilder:      {len(latencies_ms)}")
        print(f"CPU-Latenz Ø / Median: {np.mean(latencies_ms):.1f} ms / {np.median(latencies_ms):.1f} ms")
        print(f"CPU-Latenz P95:        {np.percentile(latencies_ms, 95):.1f} ms")
        print(
            f"Triage-Ergebnis:       defects={triage_counts['defects']}, "
            f"manual-review={triage_counts['manual-review']}, "
            f"auto_passed={triage_counts['auto_passed']}"
        )


if __name__ == "__main__":
    run_benchmark()
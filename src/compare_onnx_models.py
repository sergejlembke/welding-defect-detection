from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import onnxruntime as ort

from src.benchmark_onnx import DEFAULT_VAL_DIR, preprocess_letterbox, resolve_val_dir


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def load_ground_truth(label_path: Path, orig_w: int, orig_h: int) -> list[tuple[int, list[float]]]:
    if not label_path.is_file():
        return []
    gts: list[tuple[int, list[float]]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        xmin = (cx - w / 2.0) * orig_w
        ymin = (cy - h / 2.0) * orig_h
        xmax = (cx + w / 2.0) * orig_w
        ymax = (cy + h / 2.0) * orig_h
        gts.append((cls_id, [xmin, ymin, xmax, ymax]))
    return gts


def compute_ap50(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_model(
    model_path: Path,
    image_paths: list[Path],
    labels_dir: Path,
    class_names: list[str],
    input_size: int,
    min_confidence: float = 0.25,
    triage_min_conf: float = 0.40,
    auto_pass_conf: float = 0.75,
) -> dict[str, object]:
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    # Warmup
    session.run(
        ["detections", "feature_map"],
        {input_name: np.zeros((1, 3, input_size, input_size), dtype=np.float32)},
    )

    latencies_ms: list[float] = []
    triage_counts = {"defects": 0, "manual-review": 0, "auto_passed": 0}
    preds_by_class: dict[int, list[tuple[float, int, list[float]]]] = {
        i: [] for i in range(len(class_names))
    }
    gts_by_class: dict[int, dict[int, list[list[float]]]] = {
        i: {} for i in range(len(class_names))
    }

    for img_idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]

        for cls_id, gt_box in load_ground_truth(
            labels_dir / f"{img_path.stem}.txt", orig_w, orig_h
        ):
            gts_by_class[cls_id].setdefault(img_idx, []).append(gt_box)

        t0 = time.perf_counter()
        tensor, scale, pad_x, pad_y = preprocess_letterbox(img, input_size)
        detections, _ = session.run(["detections", "feature_map"], {input_name: tensor})
        raw_preds = np.squeeze(detections).T

        boxes_xywh, confidences, class_ids = [], [], []
        for row in raw_preds:
            scores = row[4:]
            cls_id = int(np.argmax(scores))
            score = float(scores[cls_id])
            if score < min_confidence:
                continue
            cx, cy, w, h = row[:4]
            boxes_xywh.append([
                float((cx - w / 2.0 - pad_x) / scale),
                float((cy - h / 2.0 - pad_y) / scale),
                float(w / scale),
                float(h / scale),
            ])
            confidences.append(score)
            class_ids.append(cls_id)

        indices = (
            cv2.dnn.NMSBoxes(boxes_xywh, confidences, min_confidence, 0.45)
            if boxes_xywh
            else []
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        for raw_idx in np.array(indices).flatten():
            x, y, w, h = boxes_xywh[raw_idx]
            box_xyxy = [x, y, x + w, y + h]
            cls_id = class_ids[raw_idx]
            conf = confidences[raw_idx]
            preds_by_class[cls_id].append((conf, img_idx, box_xyxy))

            if conf >= triage_min_conf:
                label = class_names[cls_id]
                if label in ("bad_weld", "defect"):
                    triage_counts["defects"] += 1
                elif label == "good_weld" and conf < auto_pass_conf:
                    triage_counts["manual-review"] += 1
                else:
                    triage_counts["auto_passed"] += 1

    # mAP@0.5 Berechnung über alle Klassen
    ap50_per_class: dict[str, float] = {}
    for cls_id, cls_name in enumerate(class_names):
        cls_preds = sorted(preds_by_class[cls_id], key=lambda x: x[0], reverse=True)
        total_gts = sum(len(b) for b in gts_by_class[cls_id].values())
        if total_gts == 0 or not cls_preds:
            ap50_per_class[cls_name] = 0.0
            continue

        matched: dict[int, set[int]] = {}
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))

        for i, (_conf, img_idx, pred_box) in enumerate(cls_preds):
            gt_boxes = gts_by_class[cls_id].get(img_idx, [])
            best_iou, best_gt_idx = 0.0, -1
            for gt_i, gt_box in enumerate(gt_boxes):
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, gt_i
            if best_iou >= 0.50 and best_gt_idx not in matched.setdefault(img_idx, set()):
                tp[i] = 1.0
                matched[img_idx].add(best_gt_idx)
            else:
                fp[i] = 1.0

        recalls = np.cumsum(tp) / total_gts
        precisions = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-9)
        ap50_per_class[cls_name] = compute_ap50(recalls, precisions)

    mean_ms = float(np.mean(latencies_ms))
    return {
        "size_mb": model_path.stat().st_size / 1e6,
        "mean_ms": mean_ms,
        "median_ms": float(np.median(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "throughput_per_min": 60_000.0 / mean_ms,
        "map50": float(np.mean(list(ap50_per_class.values()))),
        "ap50_per_class": ap50_per_class,
        "triage": triage_counts,
    }


def main(max_images: int = 100) -> None:
    export_dir = Path("outputs/onnx_export")
    fp32_path = export_dir / "yolov11x_weld_cam.onnx"
    int8_path = export_dir / "yolov11x_weld_cam_int8.onnx"
    meta = json.loads((export_dir / "model_meta.json").read_text(encoding="utf-8"))

    val_dir = resolve_val_dir(DEFAULT_VAL_DIR)
    labels_dir = val_dir.parent / "labels"
    image_paths = sorted(
        [p for p in val_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )[:max_images]

    print(f"Starte A/B-Benchmark auf {len(image_paths)} Validierungsbildern (CPU)...", flush=True)
    res_fp32 = evaluate_model(
        fp32_path, image_paths, labels_dir, meta["normalized_class_names"], meta["input_size"]
    )
    res_int8 = evaluate_model(
        int8_path, image_paths, labels_dir, meta["normalized_class_names"], meta["input_size"]
    )

    speedup = res_fp32["mean_ms"] / res_int8["mean_ms"]  # type: ignore[operator]
    map_delta = (res_int8["map50"] - res_fp32["map50"]) * 100.0  # type: ignore[operator]

    print("\n================ VERGLEICH: FP32 vs. STATIC INT8 ================")
    print(f"{'Metrik':<28} | {'FP32 (Basis)':>14} | {'Static INT8':>14}")
    print("-" * 64)
    print(f"{'Modellgröße (MB)':<28} | {res_fp32['size_mb']:>13.1f} | {res_int8['size_mb']:>13.1f}")
    print(f"{'CPU-Latenz Ø (ms)':<28} | {res_fp32['mean_ms']:>13.1f} | {res_int8['mean_ms']:>13.1f}")
    print(f"{'CPU-Latenz Median (ms)':<28} | {res_fp32['median_ms']:>13.1f} | {res_int8['median_ms']:>13.1f}")
    print(f"{'CPU-Latenz P95 (ms)':<28} | {res_fp32['p95_ms']:>13.1f} | {res_int8['p95_ms']:>13.1f}")
    print(f"{'Durchsatz / Pod (Bilder/Min)':<28} | {res_fp32['throughput_per_min']:>13.1f} | {res_int8['throughput_per_min']:>13.1f}")
    print(f"{'mAP@0.5 (gesamt)':<28} | {res_fp32['map50']:>13.4f} | {res_int8['map50']:>13.4f}")
    for cls_name in meta["normalized_class_names"]:
        print(
            f"  AP@0.5 [{cls_name:<16}] | "
            f"{res_fp32['ap50_per_class'][cls_name]:>13.4f} | "  # type: ignore[index]
            f"{res_int8['ap50_per_class'][cls_name]:>13.4f}"    # type: ignore[index]
        )
    print("-" * 64)
    print(f"Speedup (INT8 vs. FP32):     {speedup:.2f}x schneller")
    print(f"mAP@0.5 Differenz:           {map_delta:+.2f} Prozentpunkte")
    print(f"Triage FP32:                 {res_fp32['triage']}")
    print(f"Triage INT8:                 {res_int8['triage']}")


if __name__ == "__main__":
    main()
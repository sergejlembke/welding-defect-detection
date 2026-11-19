"""Reproduce the Experiment 01 baseline metrics without modifying archived outputs."""

from __future__ import annotations

import argparse
import csv
import os
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from src.compare_onnx_models import evaluate_model

CLASS_NAMES = ["Bad Weld", "Good Weld", "Defect"]
REJECT_IDS = {0, 2}
CONF_THRESHOLDS = (0.05, 0.10, 0.25, 0.40)
TAU_GOOD = (0.40, 0.50, 0.60, 0.70, 0.80)
TAU_LOW = (0.05, 0.10, 0.20, 0.30, 0.40)
DATA_YAML = ROOT / "data/raw_welding_dataset/The Welding Defect Dataset - v2/The Welding Defect Dataset - v2/data.yaml"
WEIGHTS = ROOT / "outputs/yolov11_runs/run_20261106_124707/weights/best.pt"
OUT = ROOT / "docs/experiments/assets/exp_01"


def json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(intersection / max(area_a + area_b - intersection, 1e-12))


def load_labels(label_path: Path) -> list[tuple[int, np.ndarray]]:
    if not label_path.is_file():
        return []
    labels = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        labels.append((int(fields[0]), np.asarray([float(x) for x in fields[1:5]])))
    return labels


def split_dir(split: str) -> Path:
    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    relative = Path(data["test" if split == "test" else "val"])
    candidates = [
        (DATA_YAML.parent / relative).resolve(),
        (DATA_YAML.parent.parent / relative).resolve(),
        DATA_YAML.parent / ("test" if split == "test" else "valid") / "images",
        DATA_YAML.parent.parent / ("test" if split == "test" else "valid") / "images",
        *sorted((ROOT / "data").rglob(f"{'test' if split == 'test' else 'valid'}/images")),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not resolve {split} split from {DATA_YAML}")


def collect_predictions(
    model: YOLO, split: str, device: str, max_images: int | None = None
) -> list[dict[str, Any]]:
    image_dir = split_dir(split)
    label_dir = image_dir.parent / "labels"
    records: list[dict[str, Any]] = []
    source: str | list[str] = str(image_dir)
    if max_images is not None:
        images = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )[:max_images]
        source = [str(p) for p in images]
    results = model.predict(
        source=source,
        conf=0.001,
        iou=0.7,
        device=device,
        verbose=False,
        stream=True,
        save=False,
        batch=int(os.environ.get("BASELINE_BATCH", "128")),
    )
    for result in results:
        image_path = Path(result.path)
        boxes = result.boxes
        predictions = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            predictions = [
                {"class_id": int(c), "confidence": float(s), "box": b.tolist()}
                for b, s, c in zip(xyxy, conf, cls)
            ]
        height, width = result.orig_shape
        ground_truth = []
        for class_id, normalized in load_labels(label_dir / f"{image_path.stem}.txt"):
            cx, cy, w, h = normalized
            ground_truth.append(
                {
                    "class_id": class_id,
                    "box": [
                        float((cx - w / 2) * width),
                        float((cy - h / 2) * height),
                        float((cx + w / 2) * width),
                        float((cy + h / 2) * height),
                    ],
                }
            )
        records.append({"image": str(image_path), "predictions": predictions, "ground_truth": ground_truth})
    return records


def matched_recall(records: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    totals = {i: 0 for i in range(len(CLASS_NAMES))}
    hits = {i: 0 for i in range(len(CLASS_NAMES))}
    for record in records:
        predictions = [p for p in record["predictions"] if p["confidence"] >= threshold]
        for gt in record["ground_truth"]:
            class_id = gt["class_id"]
            totals[class_id] += 1
            if any(
                p["class_id"] == class_id and box_iou(np.asarray(p["box"]), np.asarray(gt["box"])) >= 0.5
                for p in predictions
            ):
                hits[class_id] += 1
    return {
        CLASS_NAMES[i]: hits[i] / totals[i] if totals[i] else 0.0
        for i in range(len(CLASS_NAMES))
    }


def gating_metrics(records: list[dict[str, Any]], tau_good: float, tau_low: float) -> dict[str, float]:
    positive = auto_clear = escapes = reject_hits = reject_total = 0
    for record in records:
        gt_reject = any(g["class_id"] in REJECT_IDS for g in record["ground_truth"])
        positive += int(gt_reject)
        reject_total += sum(g["class_id"] in REJECT_IDS for g in record["ground_truth"])
        reject_predictions = [
            p for p in record["predictions"] if p["class_id"] in REJECT_IDS and p["confidence"] >= tau_low
        ]
        good_predictions = [
            p for p in record["predictions"] if p["class_id"] == 1 and p["confidence"] >= tau_good
        ]
        auto_clear_now = bool(good_predictions) and not reject_predictions
        auto_clear += int(auto_clear_now)
        escapes += int(auto_clear_now and gt_reject)
        for gt in record["ground_truth"]:
            if gt["class_id"] in REJECT_IDS and any(
                box_iou(np.asarray(p["box"]), np.asarray(gt["box"])) >= 0.5 for p in reject_predictions
            ):
                reject_hits += 1
    n = len(records)
    return {
        "tau_good": tau_good,
        "tau_low": tau_low,
        "escape_rate": escapes / positive if positive else 0.0,
        "auto_clear_rate": auto_clear / n if n else 0.0,
        "hitl_review_rate": 1.0 - auto_clear / n if n else 0.0,
        "merged_reject_class_recall": reject_hits / reject_total if reject_total else 0.0,
        "images": n,
    }


def aggregate_metrics(model: YOLO, split: str, device: str, run_dir: Path) -> dict[str, Any]:
    metrics = model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=640,
        batch=int(os.environ.get("BASELINE_BATCH", "128")),
        conf=0.001,
        iou=0.7,
        device=device,
        plots=False,
        project=str(run_dir),
        name=split,
        exist_ok=True,
        verbose=False,
    )
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        precision, recall, ap50, ap5095 = metrics.box.class_result(i)
        per_class[name] = {"P": precision, "R": recall, "AP50": ap50, "AP50-95": ap5095}
    return {
        "split": split,
        "P": metrics.box.mp,
        "R": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "per_class": per_class,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_assets() -> None:
    source = ROOT / "outputs/yolov11_runs/run_20261106_124707"
    for name in ("results.png", "BoxPR_curve.png", "BoxR_curve.png", "confusion_matrix_normalized.png"):
        shutil.copy2(source / name, OUT / name)
    cam_files = sorted((ROOT / "outputs/cam_preview").glob("*.png"))
    selected = [next(p for p in cam_files if "defect_" in p.name)]
    selected.append(next(p for p in cam_files if "bad_weld_" in p.name))
    manual = [p for p in cam_files if "manual-review_" in p.name]
    selected.extend(manual[:2])
    for path in selected:
        shutil.copy2(path, OUT / f"cam_{path.name}")


def onnx_comparison(max_images: int | None = None) -> dict[str, Any]:
    export_dir = ROOT / "outputs/onnx_export"
    meta = json.loads((export_dir / "model_meta.json").read_text(encoding="utf-8"))
    image_dir = split_dir("val")
    images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if max_images is not None:
        images = images[:max_images]
    labels = image_dir.parent / "labels"
    common = {
        "image_paths": images,
        "labels_dir": labels,
        "class_names": meta["normalized_class_names"],
        "input_size": int(meta["input_size"]),
        "min_confidence": 0.001,
        "triage_min_conf": 0.40,
        "auto_pass_conf": 0.75,
    }
    return {
        "hardware": "local MacBook CPUExecutionProvider",
        "images": len(images),
        "fp32": evaluate_model(export_dir / "yolov11x_weld_cam.onnx", **common),
        "int8": evaluate_model(export_dir / "yolov11x_weld_cam_int8.onnx", **common),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps", help="Ultralytics device; MPS falls back to CPU.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit smoke-test inference locally.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    device = args.device
    try:
        import torch

        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"
    model = YOLO(str(WEIGHTS))
    aggregate = {}
    sweep_rows = []
    gating_rows = []
    for split in ("val", "test"):
        if args.max_images is None:
            aggregate[split] = aggregate_metrics(model, split, device, OUT / "_val_runs")
        else:
            aggregate[split] = {"status": "skipped_local_smoke_test"}
        records = collect_predictions(model, split, device, args.max_images)
        for threshold in CONF_THRESHOLDS:
            sweep_rows.append({"split": split, "conf": threshold, **matched_recall(records, threshold)})
        for tau_good in TAU_GOOD:
            for tau_low in TAU_LOW:
                gating_rows.append({"split": split, **gating_metrics(records, tau_good, tau_low)})
    (OUT / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2, default=json_default), encoding="utf-8")
    (OUT / "confidence_sweep.csv").unlink(missing_ok=True)
    write_csv(OUT / "confidence_sweep.csv", sweep_rows)
    (OUT / "gating_simulation.csv").unlink(missing_ok=True)
    write_csv(OUT / "gating_simulation.csv", gating_rows)
    (OUT / "run_metadata.json").write_text(
        json.dumps({"device": device, "weights": str(WEIGHTS), "data": str(DATA_YAML)}, indent=2),
        encoding="utf-8",
    )
    (OUT / "onnx_comparison.json").write_text(
        json.dumps(onnx_comparison(args.max_images), indent=2, default=json_default), encoding="utf-8"
    )
    copy_assets()
    print(json.dumps({"device": device, "aggregate": aggregate}, indent=2, default=json_default))


if __name__ == "__main__":
    main()

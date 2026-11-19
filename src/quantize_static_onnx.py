from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from src.benchmark_onnx import DEFAULT_VAL_DIR, preprocess_letterbox, resolve_val_dir


class WeldingCalibrationReader(CalibrationDataReader):
    """Liefert Validierungsbilder mit Live-Fortschrittsanzeige für die INT8-Kalibrierung."""

    def __init__(self, image_paths: list[Path], input_name: str, input_size: int) -> None:
        self.image_paths = image_paths
        self.input_name = input_name
        self.input_size = input_size
        self._index = 0
        self._start_time = time.perf_counter()

    def get_next(self) -> dict[str, np.ndarray] | None:
        while self._index < len(self.image_paths):
            path = self.image_paths[self._index]
            self._index += 1
            img = cv2.imread(str(path))
            if img is None:
                continue

            if self._index == 1 or self._index % 5 == 0 or self._index == len(self.image_paths):
                elapsed = time.perf_counter() - self._start_time
                print(
                    f"  Kalibriere Bild {self._index}/{len(self.image_paths)} ({elapsed:.1f}s)...",
                    flush=True,
                )

            tensor, _, _, _ = preprocess_letterbox(img, self.input_size)
            return {self.input_name: tensor}
        return None


import random
from collections import defaultdict


def select_stratified_calib_images(
        val_images_dir: Path, num_images: int = 36, seed: int = 42
) -> list[Path]:
    """
    Wählt reproduzierbar eine ausgewogene Mischung aus allen Klassen (bad_weld, good_weld, defect)
    sowie Mehrklassen-/Grenzfall-Bildern anhand der YOLO-Ground-Truth-Labels aus.
    """
    rng = random.Random(seed)
    labels_dir = val_images_dir.parent / "labels"
    all_images = sorted(
        [p for p in val_images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

    if not labels_dir.is_dir():
        rng.shuffle(all_images)
        return all_images[:num_images]

    buckets: dict[str, list[Path]] = defaultdict(list)
    for img_path in all_images:
        label_file = labels_dir / f"{img_path.stem}.txt"
        if not label_file.is_file():
            continue
        classes_in_img = {
            line.split()[0]
            for line in label_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if len(classes_in_img) > 1:
            # Bilder mit gleichzeitigen Gut- und Schlecht-Bereichen (ideale Grenzfälle)
            buckets["mixed_boundary"].append(img_path)
        elif "0" in classes_in_img:
            buckets["bad_weld"].append(img_path)
        elif "1" in classes_in_img:
            buckets["good_weld"].append(img_path)
        elif "2" in classes_in_img:
            buckets["defect"].append(img_path)

    for paths in buckets.values():
        rng.shuffle(paths)

    # Gleichmäßig im Round-Robin-Verfahren aus allen 4 Gruppen ziehen
    selected: list[Path] = []
    bucket_keys = ["mixed_boundary", "bad_weld", "good_weld", "defect"]
    idx = 0
    while len(selected) < num_images and any(buckets[k] for k in bucket_keys):
        key = bucket_keys[idx % len(bucket_keys)]
        if buckets[key]:
            selected.append(buckets[key].pop())
        idx += 1

    print(
        f"Stratifizierte Kalibrierungsauswahl ({len(selected)} Bilder): "
        f"Seed={seed}, Gruppen={bucket_keys}",
        flush=True,
    )
    return selected


def run_static_quantization(
        export_dir: Path = Path("outputs/onnx_export"),
        num_calib_images: int = 100,
) -> None:
    fp32_path = export_dir / "yolov11x_weld_cam.onnx"
    preproc_path = export_dir / "yolov11x_weld_cam_preproc.onnx"
    int8_path = export_dir / "yolov11x_weld_cam_int8.onnx"
    meta_path = export_dir / "model_meta.json"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    input_size = int(meta["input_size"])

    print("1/3 Führe ONNX Graph Pre-Processing aus...", flush=True)
    try:
        quant_pre_process(
            input_model_path=str(fp32_path),
            output_model_path=str(preproc_path),
            skip_symbolic_shape=False,
        )
    except Exception as exc:
        print(f"  Symbolic Shape Inference übersprungen ({exc}), nutze Standard-Pre-Processing...", flush=True)
        quant_pre_process(
            input_model_path=str(fp32_path),
            output_model_path=str(preproc_path),
            skip_symbolic_shape=True,
        )

    # Detect-Head (model.23) auf dem vorverarbeiteten Graphen ausnehmen (bleibt in FP32)
    model = onnx.load(str(preproc_path))
    detect_head_nodes = [
        node.name for node in model.graph.node if "model.23" in node.name and node.name
    ]
    print(f"2/3 Detect-Head-Knoten in FP32 belassen: {len(detect_head_nodes)}", flush=True)

    val_dir = resolve_val_dir(DEFAULT_VAL_DIR)
    calib_images = select_stratified_calib_images(val_dir, num_images=100, seed=42)

    reader = WeldingCalibrationReader(
        image_paths=calib_images,
        input_name="images",
        input_size=input_size,
    )

    print(f"3/3 Starte Static INT8 Quantisierung mit {len(calib_images)} Bildern...", flush=True)
    quantize_static(
        model_input=str(preproc_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        nodes_to_exclude=detect_head_nodes,
    )

    # Temporäre Pre-Processing-Datei wieder aufräumen
    if preproc_path.exists():
        preproc_path.unlink()

    print(
        f"\nFertig! Static INT8 Modell gespeichert: {int8_path} "
        f"({int8_path.stat().st_size / 1e6:.1f} MB)",
        flush=True,
    )


if __name__ == "__main__":
    run_static_quantization()
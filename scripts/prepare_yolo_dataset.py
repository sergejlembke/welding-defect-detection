import argparse
import os
from pathlib import Path
import sys
import pandas as pd
import yaml


def parse_yolo_dataset(
    raw_dir: str = "data/raw_welding_dataset",
    output_dir: str = "data",
) -> None:
    """Parse YOLO dataset annotations and store clean relative image paths in CSVs.

    Args:
        raw_dir: Root raw directory containing unzipped dataset.
        output_dir: Target directory for train.csv, val.csv, test.csv.
    """
    project_root = Path.cwd()
    raw_path = project_root / raw_dir
    output_path = project_root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(
            f"[Error] Directory '{raw_path}' does not exist. "
            "Please check path or run download script first."
        )
        sys.exit(1)

    # 1. Dynamisch data.yaml im gesamten raw_dir suchen
    yaml_files = list(raw_path.rglob("data.yaml"))
    defect_class_ids = set()

    if yaml_files:
        with open(yaml_files[0], "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
            names = data_cfg.get("names", [])

            for idx, name in enumerate(names):
                name_str = str(name).lower()
                if (
                    "bad" in name_str
                    or "defect" in name_str
                    or "crack" in name_str
                ):
                    defect_class_ids.add(idx)
        print(
            f"[Info] Found 'data.yaml' at: {yaml_files[0].relative_to(project_root)}"
        )
        print(
            f"[Info] Defect Class IDs detected from config: {defect_class_ids}"
        )
    else:
        defect_class_ids = {0, 1}

    # 2. Split-Ordner (train, valid/val, test) verarbeiten
    for split in ["train", "valid", "val", "test"]:
        # Suche nach Ordnern wie ".../train/images"
        images_dirs = [
            d for d in raw_path.rglob(f"{split}/images") if d.is_dir()
        ]

        if not images_dirs:
            continue

        img_folder = images_dirs[0]
        lbl_folder = img_folder.parent / "labels"

        image_paths = []
        labels = []

        for img_path in img_folder.glob("*.*"):
            if img_path.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
            ]:
                continue

            is_defect = 0
            if lbl_folder.exists():
                lbl_path = lbl_folder / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    with open(lbl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                if class_id in defect_class_ids:
                                    is_defect = 1
                                    break

            # Relativen Pfad ausgehend vom Projekt-Root erzwingen
            rel_path = os.path.relpath(img_path, start=project_root)
            image_paths.append(rel_path)
            labels.append(is_defect)

        if image_paths:
            df = pd.DataFrame({"image_path": image_paths, "label": labels})

            csv_name = (
                "val.csv" if split in ["valid", "val"] else f"{split}.csv"
            )
            out_csv = output_path / csv_name
            df.to_csv(out_csv, index=False)
            print(
                f"[Success] Saved {len(df)} samples to '{out_csv.relative_to(project_root)}' "
                f"(Defects: {df['label'].sum()}, OK: {len(df) - df['label'].sum()})"
            )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert YOLO dataset annotations into relative-path classification CSVs."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw_welding_dataset",
        help="Root directory where raw dataset is located",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Target directory to store train.csv, val.csv, test.csv",
    )
    args = parser.parse_args()

    parse_yolo_dataset(raw_dir=args.raw_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
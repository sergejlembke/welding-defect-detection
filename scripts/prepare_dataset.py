from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_welding_dataset(
    raw_dir: str = "data/raw_welding_dataset", output_dir: str = "data"
) -> None:
    """Scan raw image directory structure and create stratified train/val/test CSV splits."""
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths = []
    labels = []

    # Unterordner scannen (z.B. "ok" -> 0, "defect" -> 1)
    for path in raw_path.rglob("*"):
        if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
            parent_name = path.parent.name.lower()
            if "defect" in parent_name or "bad" in parent_name or "nio" in parent_name:
                label = 1
            else:
                label = 0

            image_paths.append(str(path))
            labels.append(label)

    if not image_paths:
        raise FileNotFoundError(
            f"Keine Bilder in '{raw_dir}' gefunden. Bitte Ordnerstruktur prüfen."
        )

    df = pd.DataFrame({"image_path": image_paths, "label": labels})

    # Stratified Split (Training / Val / Test)
    train_df, test_val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        test_val_df, test_size=0.5, random_state=42, stratify=test_val_df["label"]
    )

    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    print(f"[Success] Datensatz vorbereitet in '{output_dir}':")
    print(f"  Train: {len(train_df)} Samples (Defekte: {train_df['label'].sum()})")
    print(f"  Val:   {len(val_df)} Samples (Defekte: {val_df['label'].sum()})")
    print(f"  Test:  {len(test_df)} Samples (Defekte: {test_df['label'].sum()})")


if __name__ == "__main__":
    prepare_welding_dataset()
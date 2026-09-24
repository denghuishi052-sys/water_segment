#!/usr/bin/env python
"""Create a drop-in single-class flood dataset from flood-vs-water labels.

Class 0 (flood) remains foreground. Class 1 (natural_water) is deliberately
written as background so a one-class YOLOv8-seg model keeps the legacy ONNX
contract expected by the .NET platform.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_utils import ensure_dir, write_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single-class flood YOLO-seg dataset.")
    parser.add_argument("--input_processed", default="data/floodnet_flood_vs_water_1024/processed")
    parser.add_argument("--output_processed", default="data/floodnet_flood_only_1024/processed")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input_processed).resolve()
    output = Path(args.output_processed).resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Pass --overwrite to rebuild it.")
        shutil.rmtree(output)

    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        image_out = ensure_dir(output / "images" / split)
        label_out = ensure_dir(output / "labels" / split)
        mask_out = ensure_dir(output / "masks" / split)
        images = sorted((source / "images" / split).glob("*.jpg"))
        for image_path in tqdm(images, desc=f"flood-only {split}"):
            stem = image_path.stem
            shutil.copy2(image_path, image_out / image_path.name)
            mask = cv2.imread(str(source / "masks" / split / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(stem)
            flood_mask = (mask == 1).astype(np.uint8)
            cv2.imwrite(str(mask_out / f"{stem}.png"), flood_mask)
            source_lines = (source / "labels" / split / f"{stem}.txt").read_text(encoding="utf-8").splitlines()
            flood_lines = [line for line in source_lines if line.strip() and line.split(maxsplit=1)[0] == "0"]
            (label_out / f"{stem}.txt").write_text(
                "\n".join(flood_lines) + ("\n" if flood_lines else ""), encoding="utf-8"
            )
            rows.append({"split": split, "stem": stem, "flood_pixels": int(flood_mask.sum()), "flood_polygons": len(flood_lines)})

    write_yaml(output / "flood_only.yaml", {
        "path": str(output), "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "flood"},
    })
    report = ensure_dir(output.parent / "split_report") / "flood_only_report.csv"
    with report.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Dataset: {output}")
    print(f"YAML: {output / 'flood_only.yaml'}")
    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        print(f"{split}: {len(subset)} images, flood={sum(row['flood_pixels'] > 0 for row in subset)}")


if __name__ == "__main__":
    main()

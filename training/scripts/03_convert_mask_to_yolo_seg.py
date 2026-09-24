#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_utils import IMAGE_EXTS, ensure_dir, list_files, read_image, read_mask_binary, read_mask_multiclass, write_yaml
from src.mask_utils import binary_mask_to_polygons, write_yolo_seg_txt


def parse_args():
    p = argparse.ArgumentParser(description="Convert masks to YOLOv8 segmentation polygon labels.")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--class_id", type=int, default=0)
    p.add_argument("--class_name", default="waterlogging")
    p.add_argument("--num_classes", type=int, default=1,
                   help="Number of classes. Use >1 with --mask_mode multiclass.")
    p.add_argument("--class_names", nargs="+", default=None,
                   help="Per-class names for multiclass mode, e.g. --class_names background flooded_road water. "
                        "Length must equal --num_classes. Class 0 is treated as background and produces no labels.")
    p.add_argument("--min_area", type=int, default=20)
    p.add_argument("--epsilon_ratio", type=float, default=0.002)
    p.add_argument("--mask_mode", default="auto", choices=["red", "non_black", "grayscale", "auto", "multiclass"],
                   help="How to read masks. Use auto for binary processed PNGs; multiclass for {0,1,...} class-index PNGs.")
    p.add_argument("--foreground_rgb", default="128,0,0",
                   help="Foreground RGB color for --mask_mode red. Example: 128,0,0")
    p.add_argument("--tolerance", type=int, default=10, help="Color/grayscale threshold tolerance.")
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(args.processed_dir)

    # Validate multiclass args
    if args.mask_mode == "multiclass":
        if args.num_classes < 2:
            raise ValueError("--mask_mode multiclass requires --num_classes >= 2")
        if args.class_names is None:
            raise ValueError("--mask_mode multiclass requires --class_names <name0> <name1> ...")
        if len(args.class_names) != args.num_classes:
            raise ValueError(
                f"len(--class_names)={len(args.class_names)} != --num_classes={args.num_classes}"
            )

    rows = []
    for split in ["train", "val", "test"]:
        image_dir = base / "images" / split
        mask_dir = base / "masks" / split
        label_dir = ensure_dir(base / "labels" / split)
        images = list_files(image_dir, IMAGE_EXTS)
        for img_path in tqdm(images, desc=f"convert {split}"):
            img = read_image(img_path)
            h, w = img.shape[:2]
            mask_path = mask_dir / f"{img_path.stem}.png"
            label_path = label_dir / f"{img_path.stem}.txt"

            if args.mask_mode == "multiclass":
                # Read raw class-index mask, then run polygon extraction per class layer.
                if mask_path.exists():
                    mask = read_mask_multiclass(mask_path, image_shape=(h, w))
                else:
                    import numpy as np
                    mask = np.zeros((h, w), dtype="uint8")
                # Aggregate (class_id, polygons) for all classes >= 1; class 0 is background.
                all_polys: list = []
                num_polys_total = 0
                for cid in range(1, args.num_classes):
                    layer = (mask == cid).astype("uint8")
                    polys = binary_mask_to_polygons(layer, min_area=args.min_area, epsilon_ratio=args.epsilon_ratio)
                    for poly in polys:
                        all_polys.append((cid, poly))
                    num_polys_total += len(polys)
                # Write YOLO-seg with per-polygon class id.
                with open(label_path, "w") as fh:
                    for cid, poly in all_polys:
                        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                        fh.write(f"{cid} {coords}\n")
                rows.append(
                    {
                        "split": split,
                        "stem": img_path.stem,
                        "image_path": str(img_path),
                        "mask_path": str(mask_path),
                        "label_path": str(label_path),
                        "num_polygons": num_polys_total,
                        "has_foreground": bool((mask > 0).any()),
                    }
                )
            else:
                mask = read_mask_binary(
                    mask_path if mask_path.exists() else None,
                    image_shape=(h, w),
                    mask_mode=args.mask_mode,
                    foreground_rgb=args.foreground_rgb,
                    tolerance=args.tolerance,
                )
                polygons = binary_mask_to_polygons(mask, min_area=args.min_area, epsilon_ratio=args.epsilon_ratio)
                write_yolo_seg_txt(label_path, polygons, class_id=args.class_id)
                rows.append(
                    {
                        "split": split,
                        "stem": img_path.stem,
                        "image_path": str(img_path),
                        "mask_path": str(mask_path),
                        "label_path": str(label_path),
                        "num_polygons": len(polygons),
                        "has_foreground": bool((mask > 0).any()),
                    }
                )
    yaml_path = base / "waterlogging.yaml"
    if args.mask_mode == "multiclass":
        names = {i: name for i, name in enumerate(args.class_names)}
    else:
        names = {args.class_id: args.class_name}
    write_yaml(
        yaml_path,
        {
            "path": str(base.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": names,
        },
    )
    report_dir = ensure_dir(base.parent / "split_report")
    pd.DataFrame(rows).to_csv(report_dir / "mask_to_yolo_report.csv", index=False)
    print(f"YOLO labels written under: {base / 'labels'}")
    print(f"YOLO dataset yaml written to: {yaml_path}")


if __name__ == "__main__":
    main()

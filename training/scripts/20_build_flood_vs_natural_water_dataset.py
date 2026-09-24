#!/usr/bin/env python
"""Build a YOLO-seg dataset that separates flood from natural water.

The source FloodNet labels use 0=background, 1=flooded_road and 2=water.
For Ultralytics, background is implicit, so this script remaps them to
0=flood and 1=natural_water.  Keeping natural water as an explicit class is
critical: treating both classes as one foreground target makes a river a
high-confidence false flood prediction by construction.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_utils import IMAGE_EXTS, ensure_dir, list_files, read_image, read_mask_multiclass, write_yaml
from src.mask_utils import binary_mask_to_polygons


SOURCE_TO_TARGET = {1: 0, 2: 1}
TARGET_NAMES = {0: "flood", 1: "natural_water"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remap FloodNet masks to flood vs natural_water YOLO-seg labels."
    )
    parser.add_argument("--input_processed", default="data/floodnet/processed")
    parser.add_argument(
        "--output_processed", default="data/floodnet_flood_vs_water_1024/processed"
    )
    parser.add_argument("--output_size", type=int, default=1024)
    parser.add_argument("--min_area", type=int, default=20)
    parser.add_argument("--epsilon_ratio", type=float, default=0.001)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument(
        "--flood_augments",
        type=int,
        default=6,
        help="Extra geometry/photometry variants for each flood-containing training image.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output(base: Path, overwrite: bool) -> None:
    resolved = base.resolve()
    data_root = (ROOT / "data").resolve()
    if data_root not in resolved.parents:
        raise ValueError(f"Output must be below {data_root}; got {resolved}")
    if resolved.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {resolved}. Pass --overwrite to rebuild it.")
        shutil.rmtree(resolved)
    for split in ("train", "val", "test"):
        ensure_dir(resolved / "images" / split)
        ensure_dir(resolved / "masks" / split)
        ensure_dir(resolved / "labels" / split)


def write_label(path: Path, target_mask: np.ndarray, min_area: int, epsilon_ratio: float) -> dict[int, int]:
    counts: dict[int, int] = {target_id: 0 for target_id in TARGET_NAMES}
    lines: list[str] = []
    for target_id in TARGET_NAMES:
        polygons = binary_mask_to_polygons(
            (target_mask == target_id + 1).astype(np.uint8),
            min_area=min_area,
            epsilon_ratio=epsilon_ratio,
        )
        counts[target_id] = len(polygons)
        for polygon in polygons:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
            lines.append(f"{target_id} {coords}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return counts


def remap_mask(mask: np.ndarray) -> np.ndarray:
    # Store 0=background, 1=flood, 2=natural_water in the inspection mask.
    target = np.zeros(mask.shape[:2], dtype=np.uint8)
    for source_id, target_id in SOURCE_TO_TARGET.items():
        target[mask == source_id] = target_id + 1
    return target


def augment_pair(image: np.ndarray, mask: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Apply deterministic, class-preserving variants to scarce flood examples."""
    mode = index % 4
    if mode == 0:
        return cv2.flip(image, 1), cv2.flip(mask, 1), "hflip"
    h, w = mask.shape[:2]
    if mode == 1:
        angle = -12.0 if (index // 4) % 2 == 0 else 12.0
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return (
            cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101),
            cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0),
            f"rotate_{int(angle)}",
        )
    if mode == 2:
        alpha = 0.82 if (index // 4) % 2 == 0 else 1.18
        augmented = np.clip(image.astype(np.float32) * alpha + 8.0, 0, 255).astype(np.uint8)
        return augmented, mask.copy(), "brightness"
    crop_ratio = 0.82
    crop_w, crop_h = int(w * crop_ratio), int(h * crop_ratio)
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    return (
        cv2.resize(image[y0:y0 + crop_h, x0:x0 + crop_w], (w, h), interpolation=cv2.INTER_LINEAR),
        cv2.resize(mask[y0:y0 + crop_h, x0:x0 + crop_w], (w, h), interpolation=cv2.INTER_NEAREST),
        "context_crop",
    )


def process_split(args: argparse.Namespace, source: Path, output: Path, split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for image_path in list_files(source / "images" / split, IMAGE_EXTS):
        image = read_image(image_path)
        mask_path = source / "masks" / split / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")
        source_mask = read_mask_multiclass(mask_path, image_shape=image.shape[:2])
        target_mask = remap_mask(source_mask)

        if args.output_size > 0:
            image = cv2.resize(image, (args.output_size, args.output_size), interpolation=cv2.INTER_LINEAR)
            target_mask = cv2.resize(
                target_mask, (args.output_size, args.output_size), interpolation=cv2.INTER_NEAREST
            )

        image_out = output / "images" / split / f"{image_path.stem}.jpg"
        mask_out = output / "masks" / split / f"{image_path.stem}.png"
        label_out = output / "labels" / split / f"{image_path.stem}.txt"
        if not cv2.imwrite(str(image_out), image, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
            raise OSError(f"Failed to write {image_out}")
        if not cv2.imwrite(str(mask_out), target_mask):
            raise OSError(f"Failed to write {mask_out}")
        polygon_counts = write_label(label_out, target_mask, args.min_area, args.epsilon_ratio)
        rows.append(
            {
                "split": split,
                "stem": image_path.stem,
                "flood_pixels": int((target_mask == 1).sum()),
                "natural_water_pixels": int((target_mask == 2).sum()),
                "flood_polygons": polygon_counts[0],
                "natural_water_polygons": polygon_counts[1],
            }
        )
        if split != "train" or args.flood_augments <= 0 or not np.any(target_mask == 1):
            continue
        for index in range(args.flood_augments):
            augmented_image, augmented_mask, augmentation = augment_pair(image, target_mask, index)
            stem = f"{image_path.stem}__flood_aug{index + 1:02d}_{augmentation}"
            image_out = output / "images" / split / f"{stem}.jpg"
            mask_out = output / "masks" / split / f"{stem}.png"
            label_out = output / "labels" / split / f"{stem}.txt"
            if not cv2.imwrite(str(image_out), augmented_image, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                raise OSError(f"Failed to write {image_out}")
            if not cv2.imwrite(str(mask_out), augmented_mask):
                raise OSError(f"Failed to write {mask_out}")
            augmented_counts = write_label(label_out, augmented_mask, args.min_area, args.epsilon_ratio)
            rows.append(
                {
                    "split": split,
                    "stem": stem,
                    "flood_pixels": int((augmented_mask == 1).sum()),
                    "natural_water_pixels": int((augmented_mask == 2).sum()),
                    "flood_polygons": augmented_counts[0],
                    "natural_water_polygons": augmented_counts[1],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    source = Path(args.input_processed).resolve()
    output = Path(args.output_processed)
    if not source.exists():
        raise FileNotFoundError(source)
    prepare_output(output, args.overwrite)

    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        rows.extend(process_split(args, source, output, split))

    output = output.resolve()
    write_yaml(
        output / "flood_vs_water.yaml",
        {
            "path": str(output),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": TARGET_NAMES,
        },
    )
    report_dir = ensure_dir(output.parent / "split_report")
    report_path = report_dir / "flood_vs_water_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["split", "stem"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset: {output}")
    print(f"YAML: {output / 'flood_vs_water.yaml'}")
    print(f"Report: {report_path}")
    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        print(
            f"{split}: {len(subset)} images, "
            f"flood={sum(row['flood_pixels'] > 0 for row in subset)}, "
            f"natural_water={sum(row['natural_water_pixels'] > 0 for row in subset)}"
        )


if __name__ == "__main__":
    main()

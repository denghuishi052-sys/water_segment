#!/usr/bin/env python
"""Prepare FloodNet dataset from zip for YOLOv8-seg training.

FloodNet segmentation label mapping (from class_mapping.csv):
    0: Background
    1: Building-flooded
    2: Building-non-flooded
    3: Road-flooded
    4: Road-non-flooded
    5: Water
    6: Tree
    7: Vehicle
    8: Pool
    9: Grass

We convert to 3-class: background (0,1,2,4,6,7,9), flooded_road (3), water (5,8).

FloodNet zip structure:
    floodnet/Train/Labeled/Flooded/image/   + mask/
    floodnet/Train/Labeled/Non-Flooded/image/ + mask/
    floodnet/Train/Unlabeled/image/          (skipped)
    floodnet/Validation/image/               (no masks, used as val)
    floodnet/Test/image/                     (no masks, skipped)

Output (data/floodnet/raw/):
    images/  -->  all labeled images as .jpg
    masks/   -->  3-class flood masks as .png (0=background, 1=flooded_road, 2=water)
"""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# FloodNet pixel values in mask images correspond to class indices.
# We map these to 3 classes: background, flooded_road, water.
FLOODNET_CLASS_MAP = {
    0: 0, 1: 0, 2: 0,  # background
    3: 1,               # flooded_road
    4: 0,               # background
    5: 2, 8: 2,         # water (river + pool)
    6: 0, 7: 0, 9: 0,   # background
}


def read_mask_from_zip(zip_file: ZipFile, name: str) -> np.ndarray:
    """Read a mask image from zip, return as-is (single channel class index map)."""
    with zip_file.open(name) as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read mask: {name}")
    # If RGB, convert to single channel by taking first channel
    # (FloodNet masks are single-channel with class indices, but may be saved as RGB)
    if img.ndim == 3:
        # If all channels are the same (class index repeated), just take one
        if np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 1] == img[:, :, 2]):
            img = img[:, :, 0]
        else:
            # Multi-color mask — need to check pixel value encoding
            # Some FloodNet masks use unique colors per class
            # Take the red channel as class index hint
            img = img[:, :, 0]
    return img


def read_image_from_zip(zip_file: ZipFile, name: str) -> np.ndarray:
    """Read an RGB image from zip."""
    with zip_file.open(name) as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {name}")
    return img


def convert_to_multiclass_flood_mask(mask: np.ndarray) -> np.ndarray:
    """Convert 10-class FloodNet mask to 3-class mask.

    Class mapping (FLOODNET_CLASS_MAP):
        0 → background  (orig 0,1,2,4,6,7,9)
        1 → flooded_road (orig 3)
        2 → water        (orig 5,8)
    """
    out = np.zeros(mask.shape[:2], dtype=np.uint8)
    for orig_id, new_id in FLOODNET_CLASS_MAP.items():
        if new_id > 0:
            out[mask == orig_id] = new_id
    return out


def get_unique_mask_values(zip_file: ZipFile, mask_names: list[str], sample_count: int = 5) -> list[set]:
    """Sample a few masks to discover what pixel values they contain."""
    unique_sets = []
    for name in mask_names[:sample_count]:
        mask = read_mask_from_zip(zip_file, name)
        unique_vals = set(np.unique(mask).tolist())
        unique_sets.append(unique_vals)
        print(f"  {name}: unique values = {sorted(unique_vals)}")
    return unique_sets


def prepare_dataset(
    zip_path: Path,
    output_dir: Path,
    limit: int | None = None,
    overwrite: bool = False,
    inspect_only: bool = False,
) -> pd.DataFrame:
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    report_dir = output_dir / "reports"

    rows = []
    with ZipFile(zip_path) as z:
        all_names = z.namelist()

        # --- Collect labeled data ---
        # Flooded images and masks
        flooded_images = sorted(
            n for n in all_names
            if n.startswith("floodnet/Train/Labeled/Flooded/image/")
            and n.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        flooded_masks = sorted(
            n for n in all_names
            if n.startswith("floodnet/Train/Labeled/Flooded/mask/")
            and n.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        # Non-Flooded images and masks
        nonflooded_images = sorted(
            n for n in all_names
            if n.startswith("floodnet/Train/Labeled/Non-Flooded/image/")
            and n.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        nonflooded_masks = sorted(
            n for n in all_names
            if n.startswith("floodnet/Train/Labeled/Non-Flooded/mask/")
            and n.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        # Validation images (no masks — will create empty masks)
        val_images = sorted(
            n for n in all_names
            if n.startswith("floodnet/Validation/image/")
            and n.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        # Test images (no masks — skip for training, but note count)
        test_images = sorted(
            n for n in all_names
            if n.startswith("floodnet/Test/image/")
            and n.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        print(f"Flooded labeled:    {len(flooded_images)} images, {len(flooded_masks)} masks")
        print(f"Non-Flooded labeled: {len(nonflooded_images)} images, {len(nonflooded_masks)} masks")
        print(f"Validation (no mask): {len(val_images)} images")
        print(f"Test (no mask):       {len(test_images)} images")

        if inspect_only:
            print("\n=== Inspecting mask pixel values ===")
            print("Flooded masks:")
            get_unique_mask_values(z, flooded_masks, sample_count=3)
            print("Non-Flooded masks:")
            get_unique_mask_values(z, nonflooded_masks, sample_count=3)
            return pd.DataFrame()

        # Build image-mask pairs
        # Map image stem -> mask path
        pairs: list[tuple[str, str, str]] = []  # (image_zip_path, mask_zip_path, label_type)

        # Flooded
        mask_map_flooded = {Path(n).stem.replace("_lab", ""): n for n in flooded_masks}
        for img_name in flooded_images:
            stem = Path(img_name).stem
            mask_name = mask_map_flooded.get(stem)
            if mask_name is None:
                print(f"  WARNING: no mask for flooded image {img_name}")
                continue
            pairs.append((img_name, mask_name, "flooded"))

        # Non-Flooded
        mask_map_nonflooded = {Path(n).stem.replace("_lab", ""): n for n in nonflooded_masks}
        for img_name in nonflooded_images:
            stem = Path(img_name).stem
            mask_name = mask_map_nonflooded.get(stem)
            if mask_name is None:
                print(f"  WARNING: no mask for non-flooded image {img_name}")
                continue
            pairs.append((img_name, mask_name, "non_flooded"))

        if limit is not None:
            pairs = pairs[:limit]

        print(f"\nTotal labeled pairs to process: {len(pairs)}")

        # Create output directories
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        for img_zip_name, mask_zip_name, label_type in tqdm(pairs, desc="Converting FloodNet"):
            stem = Path(img_zip_name).stem
            # Prefix to avoid name collisions and identify source
            safe_stem = f"fn_{stem}"

            image_out = image_dir / f"{safe_stem}.jpg"
            mask_out = mask_dir / f"{safe_stem}.png"

            if not overwrite and image_out.exists() and mask_out.exists():
                # Skip already converted
                img_bgr = cv2.imread(str(image_out), cv2.IMREAD_COLOR)
                mask_bin = cv2.imread(str(mask_out), cv2.IMREAD_GRAYSCALE)
                if img_bgr is not None and mask_bin is not None:
                    h, w = img_bgr.shape[:2]
                    rows.append({
                        "stem": safe_stem,
                        "source_image": img_zip_name,
                        "source_mask": mask_zip_name,
                        "label_type": label_type,
                        "image_width": w,
                        "image_height": h,
                        "mask_area": int((mask_bin > 0).sum()),
                        "mask_area_ratio": float((mask_bin > 0).sum() / max(h * w, 1)),
                    })
                    continue

            # Read and convert
            img_bgr = read_image_from_zip(z, img_zip_name)
            mask_raw = read_mask_from_zip(z, mask_zip_name)

            # Convert multi-class mask to 3-class flood mask
            mask_binary = convert_to_multiclass_flood_mask(mask_raw)

            h, w = img_bgr.shape[:2]
            if mask_binary.shape[:2] != (h, w):
                mask_binary = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(str(image_out), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite(str(mask_out), mask_binary)

            rows.append({
                "stem": safe_stem,
                "source_image": img_zip_name,
                "source_mask": mask_zip_name,
                "label_type": label_type,
                "image_width": w,
                "image_height": h,
                "mask_area": int((mask_binary > 0).sum()),
                "mask_area_ratio": float((mask_binary > 0).sum() / max(h * w, 1)),
            })

    df = pd.DataFrame(rows)
    df.to_csv(report_dir / "meta.csv", index=False)

    # Print summary
    print(f"\n=== Conversion Summary ===")
    print(f"Total images converted: {len(df)}")
    print(f"Flooded: {len(df[df['label_type'] == 'flooded'])}")
    print(f"Non-Flooded: {len(df[df['label_type'] == 'non_flooded'])}")
    print(f"Images with flood pixels: {len(df[df['mask_area'] > 0])}")
    print(f"Images with no flood pixels: {len(df[df['mask_area'] == 0])}")
    print(f"\nOutput images: {image_dir}")
    print(f"Output masks:  {mask_dir}")
    print(f"Report: {report_dir / 'meta.csv'}")

    return df


def parse_args():
    p = argparse.ArgumentParser(description="Prepare FloodNet dataset for 3-class flood segmentation.")
    p.add_argument("--zip_path", default=r"C:\Users\17473\Downloads\floodnet.zip")
    p.add_argument("--output_dir", default="data/floodnet/raw")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples to convert.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing converted files.")
    p.add_argument("--inspect", action="store_true", help="Only inspect mask pixel values, don't convert.")
    return p.parse_args()


def main():
    args = parse_args()
    df = prepare_dataset(
        zip_path=Path(args.zip_path),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        overwrite=args.overwrite,
        inspect_only=args.inspect,
    )
    if not args.inspect and len(df) > 0:
        print("\nNext steps:")
        print("  1. python scripts/02_split_dataset.py --image_dir data/floodnet/raw/images --mask_dir data/floodnet/raw/masks --output_dir data/floodnet/processed --mask_mode multiclass")
        print("  2. python scripts/03_convert_mask_to_yolo_seg.py --processed_dir data/floodnet/processed --mask_mode multiclass --num_classes 3 --class_names background flooded_road water")
        print("  3. python scripts/04_train_yolov8.py --config configs/train_floodnet_yolov8m.yaml")


if __name__ == "__main__":
    main()

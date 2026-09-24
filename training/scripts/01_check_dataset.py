#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_utils import ensure_dir, infer_group_id, mask_area_stats, match_image_mask_pairs, read_image, read_mask_binary
from src.visualization import save_panel


def parse_args():
    p = argparse.ArgumentParser(description="Check image/mask dataset and produce stats.")
    p.add_argument("--image_dir", default="data/raw/images")
    p.add_argument("--mask_dir", default="data/raw/masks")
    p.add_argument("--output_dir", default="data/split_report")
    p.add_argument("--preview_num", type=int, default=50)
    p.add_argument("--allow_missing_masks", action="store_true", default=True)
    p.add_argument("--mask_mode", default="red", choices=["red", "non_black", "grayscale", "auto"],
                   help="How to convert annotation masks to binary. For the provided dataset, use red.")
    p.add_argument("--foreground_rgb", default="128,0,0",
                   help="Foreground RGB color for --mask_mode red. Example: 128,0,0")
    p.add_argument("--tolerance", type=int, default=10, help="Color/grayscale threshold tolerance.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    preview_dir = ensure_dir(out_dir / "sample_preview")
    pairs = match_image_mask_pairs(args.image_dir, args.mask_dir, allow_missing_masks=args.allow_missing_masks)
    if not pairs:
        raise RuntimeError("No image files found.")

    rows = []
    for i, pair in enumerate(tqdm(pairs, desc="checking dataset")):
        img = read_image(pair.image_path)
        h, w = img.shape[:2]
        mask = read_mask_binary(
            pair.mask_path,
            image_shape=(h, w),
            mask_mode=args.mask_mode,
            foreground_rgb=args.foreground_rgb,
            tolerance=args.tolerance,
        )
        area, ratio, bucket = mask_area_stats(mask)
        rows.append(
            {
                "stem": pair.stem,
                "image_path": str(pair.image_path),
                "mask_path": str(pair.mask_path) if pair.mask_path else "",
                "image_width": w,
                "image_height": h,
                "has_mask_file": pair.mask_path is not None,
                "has_foreground": area > 0,
                "mask_area": area,
                "mask_area_ratio": ratio,
                "bucket": bucket,
                "group_id": infer_group_id(pair.stem),
            }
        )
        if i < args.preview_num:
            save_panel(preview_dir / f"{pair.stem}.jpg", img, gt=mask)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "split_stats.csv", index=False)

    plt.figure(figsize=(8, 5))
    df["mask_area_ratio"].hist(bins=50)
    plt.xlabel("mask_area_ratio")
    plt.ylabel("count")
    plt.title("Mask area distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "mask_area_distribution.png", dpi=200)
    plt.close()

    bucket_summary = df.groupby("bucket").size().reset_index(name="count")
    bucket_summary.to_csv(out_dir / "bucket_summary.csv", index=False)
    print(f"Checked {len(df)} samples.")
    print(bucket_summary.to_string(index=False))
    print(f"Reports saved to: {out_dir}")


if __name__ == "__main__":
    main()

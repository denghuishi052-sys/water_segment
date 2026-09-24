#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_utils import ensure_dir, infer_group_id, mask_area_stats, match_image_mask_pairs, read_image, read_mask_binary, read_mask_multiclass, safe_copy, save_empty_mask
from src.split_utils import split_dataframe
from src.visualization import save_panel


def parse_args():
    p = argparse.ArgumentParser(description="Split raw image/mask pairs into train/val/test.")
    p.add_argument("--image_dir", default="data/raw/images")
    p.add_argument("--mask_dir", default="data/raw/masks")
    p.add_argument("--output_dir", default="data/processed")
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group_by_prefix", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    p.add_argument("--preview_num", type=int, default=30)
    p.add_argument("--mask_mode", default="red", choices=["red", "non_black", "grayscale", "auto", "multiclass"],
                   help="How to convert annotation masks. Use 'multiclass' for multi-class masks (0/1/2 values).")
    p.add_argument("--foreground_rgb", default="128,0,0",
                   help="Foreground RGB color for --mask_mode red. Example: 128,0,0")
    p.add_argument("--tolerance", type=int, default=10, help="Color/grayscale threshold tolerance.")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    report_dir = ensure_dir(out.parent / "split_report")

    pairs = match_image_mask_pairs(args.image_dir, args.mask_dir, allow_missing_masks=True)
    rows = []
    for pair in tqdm(pairs, desc="analyzing masks"):
        img = read_image(pair.image_path)
        h, w = img.shape[:2]
        if args.mask_mode == "multiclass":
            mask = read_mask_multiclass(pair.mask_path, image_shape=(h, w))
        else:
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
                "image_suffix": pair.image_path.suffix,
                "mask_suffix": pair.mask_path.suffix if pair.mask_path else ".png",
                "width": w,
                "height": h,
                "mask_area": area,
                "mask_area_ratio": ratio,
                "bucket": bucket,
                "group_id": infer_group_id(pair.stem),
            }
        )
    df = pd.DataFrame(rows)
    group_col = "group_id" if args.group_by_prefix else None
    split_df = split_dataframe(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        group_col=group_col,
        stratify_col="bucket",
    )

    # Create directories.
    for split in ["train", "val", "test"]:
        ensure_dir(out / "images" / split)
        ensure_dir(out / "masks" / split)
        ensure_dir(out / "labels" / split)

    preview_dir = ensure_dir(report_dir / "split_preview")
    for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc="copying split files"):
        split = row["split"]
        img_src = Path(row["image_path"])
        img_dst = out / "images" / split / f"{row['stem']}{img_src.suffix.lower()}"
        safe_copy(img_src, img_dst)

        mask_dst = out / "masks" / split / f"{row['stem']}.png"
        if row["mask_path"]:
            img = read_image(img_src)
            h, w = img.shape[:2]
            import cv2
            if args.mask_mode == "multiclass":
                # Preserve raw class indices (e.g. 0/1/2). Do not scale to 0/255.
                mask = read_mask_multiclass(Path(row["mask_path"]), image_shape=(h, w))
                cv2.imwrite(str(mask_dst), mask.astype("uint8"))
            else:
                # Normalize copied masks to png binary for downstream processing.
                mask = read_mask_binary(
                    Path(row["mask_path"]),
                    image_shape=(h, w),
                    mask_mode=args.mask_mode,
                    foreground_rgb=args.foreground_rgb,
                    tolerance=args.tolerance,
                )
                cv2.imwrite(str(mask_dst), (mask * 255).astype("uint8"))
        else:
            save_empty_mask(mask_dst, (int(row["height"]), int(row["width"])))

        if idx < args.preview_num:
            img = read_image(img_dst)
            if args.mask_mode == "multiclass":
                mask = read_mask_multiclass(mask_dst, image_shape=img.shape[:2])
                # Visualize as binary (any class > 0) for the panel preview.
                mask = (mask > 0).astype("uint8")
            else:
                mask = read_mask_binary(mask_dst, image_shape=img.shape[:2])
            save_panel(preview_dir / f"{split}_{row['stem']}.jpg", img, gt=mask)

    split_df.to_csv(report_dir / "split_assignments.csv", index=False)
    summary = split_df.groupby(["split", "bucket"]).size().reset_index(name="count")
    summary.to_csv(report_dir / "split_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Processed dataset saved to: {out}")
    print(f"Split report saved to: {report_dir}")


if __name__ == "__main__":
    main()

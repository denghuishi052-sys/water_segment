#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_utils import ensure_dir, write_yaml


def parse_args():
    p = argparse.ArgumentParser(description="Create hard-sample dataset from evaluation metrics.")
    p.add_argument("--metrics_csv", default="runs/eval/metrics_test.csv")
    p.add_argument("--data", default="data/processed/waterlogging.yaml")
    p.add_argument("--output_dir", default="data/hard")
    p.add_argument("--iou_threshold", type=float, default=0.4)
    p.add_argument("--normal_ratio", type=float, default=0.3, help="Normal samples mixed in relative to hard sample count.")
    p.add_argument("--source_split", default=None, choices=["train", "val", "test"],
                   help="Original split to copy from. Defaults to inferring from metrics file or image_path.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def infer_split(metrics_csv: Path, row: pd.Series, fallback: str = "test") -> str:
    image_path = Path(str(row.get("image_path", "")))
    parts = image_path.parts
    if "images" in parts:
        idx = parts.index("images")
        if idx + 1 < len(parts) and parts[idx + 1] in {"train", "val", "test"}:
            return parts[idx + 1]

    name = metrics_csv.name
    for split in ["train", "val", "test"]:
        if f"_{split}" in name:
            return split
    return fallback


def find_image(base: Path, split: str, stem: str, image_path: str | None = None) -> Path | None:
    if image_path:
        candidate = Path(image_path)
        if candidate.exists():
            return candidate
    image_dir = base / "images" / split
    src_img = None
    for p in image_dir.glob(stem + ".*"):
        if p.is_file():
            src_img = p
            break
    return src_img


def copy_sample(base: Path, split: str, stem: str, out: Path, out_split: str, image_path: str | None = None):
    src_img = find_image(base, split, stem, image_path=image_path)
    if src_img is None:
        return False
    src_label = base / "labels" / split / f"{stem}.txt"
    src_mask = base / "masks" / split / f"{stem}.png"
    ensure_dir(out / "images" / out_split)
    ensure_dir(out / "labels" / out_split)
    ensure_dir(out / "masks" / out_split)
    shutil.copy2(src_img, out / "images" / out_split / src_img.name)
    if src_label.exists():
        shutil.copy2(src_label, out / "labels" / out_split / src_label.name)
    else:
        (out / "labels" / out_split / f"{stem}.txt").write_text("", encoding="utf-8")
    if src_mask.exists():
        shutil.copy2(src_mask, out / "masks" / out_split / src_mask.name)
    return True


def main():
    args = parse_args()
    random.seed(args.seed)
    data_path = Path(args.data).resolve()
    metrics_csv = Path(args.metrics_csv).resolve()
    with data_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    base = Path(data_cfg["path"])
    out = Path(args.output_dir).resolve()
    ensure_dir(out)

    df = pd.read_csv(metrics_csv)
    has_case_type = "case_type" in df.columns
    if has_case_type:
        hard_mask = (
            df["case_type"].isin(["false_positive", "false_negative"])
            | ((df["case_type"] == "foreground") & (df["iou"] < args.iou_threshold))
        )
    else:
        hard_mask = (
            ((df["gt_area"] > 0) | (df["pred_area"] > 0)) & (df["iou"] < args.iou_threshold)
        ) | ((df["gt_area"] == 0) & (df["pred_area"] > 0)) | ((df["gt_area"] > 0) & (df["pred_area"] == 0))

    hard = df[hard_mask].copy()
    normal = df.drop(hard.index).copy()
    n_normal = min(len(normal), int(len(hard) * args.normal_ratio))
    normal_sample = normal.sample(n=n_normal, random_state=args.seed) if n_normal > 0 else normal.iloc[[]]

    selected = pd.concat([hard.assign(sample_type="hard"), normal_sample.assign(sample_type="normal")], ignore_index=True)
    selected = selected.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Split hard dataset into train/val. It is for fine-tuning; use test only from original processed test.
    train_cut = int(len(selected) * 0.85)
    selected["hard_split"] = "train"
    selected.loc[train_cut:, "hard_split"] = "val"

    copied = []
    for _, row in selected.iterrows():
        source_split = args.source_split or infer_split(metrics_csv, row)
        ok = copy_sample(
            base,
            source_split,
            str(row["stem"]),
            out,
            row["hard_split"],
            image_path=str(row.get("image_path", "")),
        )
        copied.append(ok)
    selected["copied"] = copied
    selected.to_csv(out / "hard_samples.csv", index=False)

    write_yaml(
        out / "waterlogging_hard.yaml",
        {
            "path": str(out.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/val",
            "names": data_cfg.get("names", {0: "waterlogging"}),
        },
    )
    print(f"Hard samples: {len(hard)}, normal mixed: {n_normal}, copied: {sum(copied)}")
    print(f"Hard dataset saved to: {out}")


if __name__ == "__main__":
    main()

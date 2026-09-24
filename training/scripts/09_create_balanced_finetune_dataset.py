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


DEFAULT_RATIOS = {
    "normal_positive": 0.45,
    "normal_negative": 0.10,
    "hard_false_positive_negative": 0.15,
    "hard_false_negative_or_low_iou_positive": 0.30,
}


def parse_args():
    p = argparse.ArgumentParser(description="Create a balanced fine-tuning dataset from eval metrics.")
    p.add_argument("--metrics_csv", default="runs/eval_yolov8m_b8_3_test/metrics_test.csv")
    p.add_argument("--data", default="data/processed/waterlogging.yaml")
    p.add_argument("--output_dir", default="data/hard_recall")
    p.add_argument("--yaml_name", default="waterlogging_hard_recall.yaml")
    p.add_argument("--total_samples", type=int, default=160)
    p.add_argument("--iou_threshold", type=float, default=0.4)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source_split", default="test", choices=["train", "val", "test"])
    p.add_argument("--normal_positive_ratio", type=float, default=DEFAULT_RATIOS["normal_positive"])
    p.add_argument("--normal_negative_ratio", type=float, default=DEFAULT_RATIOS["normal_negative"])
    p.add_argument(
        "--hard_false_positive_negative_ratio",
        type=float,
        default=DEFAULT_RATIOS["hard_false_positive_negative"],
    )
    p.add_argument(
        "--hard_false_negative_or_low_iou_positive_ratio",
        type=float,
        default=DEFAULT_RATIOS["hard_false_negative_or_low_iou_positive"],
    )
    return p.parse_args()


def sample_count(total: int, ratio: float) -> int:
    return int(round(total * ratio))


def ratios_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "normal_positive": args.normal_positive_ratio,
        "normal_negative": args.normal_negative_ratio,
        "hard_false_positive_negative": args.hard_false_positive_negative_ratio,
        "hard_false_negative_or_low_iou_positive": args.hard_false_negative_or_low_iou_positive_ratio,
    }


def counts_from_ratios(total: int, ratios: dict[str, float]) -> dict[str, int]:
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0:
        raise ValueError("At least one sampling ratio must be greater than zero.")

    normalized = {name: ratio / ratio_sum for name, ratio in ratios.items()}
    counts = {name: sample_count(total, ratio) for name, ratio in normalized.items()}
    diff = total - sum(counts.values())
    counts["normal_positive"] += diff
    return counts


def build_pools(df: pd.DataFrame, iou_threshold: float) -> dict[str, pd.DataFrame]:
    normal_positive = df[(df["case_type"] == "foreground") & (df["iou"] >= iou_threshold)].copy()
    normal_negative = df[df["case_type"] == "true_negative"].copy()
    hard_false_positive_negative = df[df["case_type"] == "false_positive"].copy()
    hard_false_negative_or_low_iou_positive = df[
        (df["case_type"] == "false_negative")
        | ((df["case_type"] == "foreground") & (df["iou"] < iou_threshold))
    ].copy()
    return {
        "normal_positive": normal_positive,
        "normal_negative": normal_negative,
        "hard_false_positive_negative": hard_false_positive_negative,
        "hard_false_negative_or_low_iou_positive": hard_false_negative_or_low_iou_positive,
    }


def choose(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if pool.empty:
        raise ValueError("Cannot sample from an empty pool.")
    if len(pool) >= n:
        return pool.sample(n=n, replace=False, random_state=seed).reset_index(drop=True)

    full_repeats, remainder = divmod(n, len(pool))
    parts = [pool.copy() for _ in range(full_repeats)]
    if remainder:
        parts.append(pool.sample(n=remainder, replace=False, random_state=seed))
    selected = pd.concat(parts, ignore_index=True)
    return selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def source_paths(base: Path, split: str, row: pd.Series) -> tuple[Path, Path, Path]:
    image_path = Path(str(row["image_path"]))
    if not image_path.exists():
        for candidate in (base / "images" / split).glob(f"{row['stem']}.*"):
            if candidate.is_file():
                image_path = candidate
                break
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for {row['stem']}: {row.get('image_path')}")

    stem = str(row["stem"])
    label_path = base / "labels" / split / f"{stem}.txt"
    mask_path = base / "masks" / split / f"{stem}.png"
    return image_path, label_path, mask_path


def copy_row(base: Path, split: str, out: Path, row: pd.Series, out_split: str, dst_stem: str) -> bool:
    image_path, label_path, mask_path = source_paths(base, split, row)
    image_dst = out / "images" / out_split / f"{dst_stem}{image_path.suffix.lower()}"
    label_dst = out / "labels" / out_split / f"{dst_stem}.txt"
    mask_dst = out / "masks" / out_split / f"{dst_stem}.png"

    ensure_dir(image_dst.parent)
    ensure_dir(label_dst.parent)
    ensure_dir(mask_dst.parent)

    shutil.copy2(image_path, image_dst)
    if label_path.exists():
        shutil.copy2(label_path, label_dst)
    else:
        label_dst.write_text("", encoding="utf-8")
    if mask_path.exists():
        shutil.copy2(mask_path, mask_dst)
    return True


def main():
    args = parse_args()
    random.seed(args.seed)

    metrics_csv = Path(args.metrics_csv).resolve()
    data_path = Path(args.data).resolve()
    out = Path(args.output_dir).resolve()

    with data_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    base = Path(data_cfg["path"])
    df = pd.read_csv(metrics_csv)
    if "case_type" not in df.columns:
        raise ValueError("metrics_csv must contain case_type. Re-run scripts/05_eval_test.py first.")

    ratios = ratios_from_args(args)
    counts = counts_from_ratios(args.total_samples, ratios)

    pools = build_pools(df, args.iou_threshold)
    selected_parts = []
    for idx, (category, count) in enumerate(counts.items()):
        part = choose(pools[category], count, args.seed + idx)
        part["sample_category"] = category
        selected_parts.append(part)

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    selected["sample_index"] = selected.index
    selected["dst_stem"] = selected.apply(
        lambda r: f"{r['stem']}__{r['sample_category']}__{int(r['sample_index']):03d}", axis=1
    )

    selected["hard_split"] = "train"
    val_n = int(round(len(selected) * args.val_ratio))
    if val_n > 0:
        val_idx = selected.sample(n=val_n, random_state=args.seed).index
        selected.loc[val_idx, "hard_split"] = "val"

    if out.exists():
        shutil.rmtree(out)
    ensure_dir(out)

    copied = []
    for _, row in selected.iterrows():
        copied.append(copy_row(base, args.source_split, out, row, row["hard_split"], row["dst_stem"]))
    selected["copied"] = copied
    selected.to_csv(out / "balanced_samples.csv", index=False)

    write_yaml(
        out / args.yaml_name,
        {
            "path": str(out.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/val",
            "names": data_cfg.get("names", {0: "waterlogging"}),
        },
    )

    print("Pool sizes:")
    for name, pool in pools.items():
        print(f"  {name}: {len(pool)}")
    print("Selected samples:")
    print(pd.Series(counts, name="target_count").to_string())
    print(selected.groupby(["sample_category", "hard_split"]).size().to_string())
    print(f"Copied: {sum(copied)}/{len(copied)}")
    print(f"Balanced dataset saved to: {out}")
    print(f"Dataset yaml saved to: {out / args.yaml_name}")


if __name__ == "__main__":
    main()

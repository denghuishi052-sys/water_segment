#!/usr/bin/env python
"""Evaluate a trained YOLOv8-seg model and save N annotated test images that
highlight TP, TN, FP, and FN pixel regions.

Each saved image is a 2-row panel:
  Row 1 (left -> right): RGB image | GT overlay | Pred overlay | Confusion map
  Row 2: legend (TP / FP / FN / TN swatches) + per-image metrics line

Usage example:
    python scripts/10_eval_tp_tn_fp.py \
        --model runs/segment/runs/segment/gf_floodnet_yolov8m_640_b16_from_combined_best-3/weights/best.pt \
        --data data/gf_floodnet/processed/waterlogging.yaml \
        --split test \
        --num 50 \
        --out_dir runs/eval/tp_tn_fp_50
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_utils import (  # noqa: E402
    IMAGE_EXTS,
    ensure_dir,
    list_files,
    read_image,
    read_mask_binary,
)
from src.mask_utils import postprocess_mask  # noqa: E402
from src.metrics import compute_binary_metrics  # noqa: E402
from src.metrics import aggregate_metrics  # noqa: E402


# BGR colors
COLOR_TP = (0, 220, 0)            # green  - GT & Pred
COLOR_FP = (0, 0, 255)            # red    - Pred only
COLOR_FN = (255, 128, 0)          # blue-ish (BGR) - GT only
COLOR_TN_BG = (180, 180, 180)     # desaturated background overlay
COLOR_TEXT = (255, 255, 255)
COLOR_BOX = (0, 0, 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="runs/segment/runs/segment/gf_floodnet_yolov8m_640_b16_from_combined_best-3/weights/best.pt",
        help="Path to best.pt",
    )
    p.add_argument("--data", default="data/gf_floodnet/processed/waterlogging.yaml")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--mask_thres", type=float, default=0.5)
    p.add_argument("--min_area_ratio", type=float, default=0.0005)
    p.add_argument("--morph_close", action="store_true", default=True)
    p.add_argument("--mask_mode", default="grayscale",
                   choices=["red", "non_black", "grayscale", "auto"])
    p.add_argument("--foreground_rgb", default="128,0,0")
    p.add_argument("--tolerance", type=int, default=0,
                   help="For grayscale masks, pixel value > tolerance counts as foreground.")
    p.add_argument("--num", type=int, default=50, help="How many test images to save.")
    p.add_argument("--sample_strategy", default="balanced_region",
                   choices=["first", "balanced_region"],
                   help="How to choose the saved visualization subset.")
    p.add_argument("--eval_all", action="store_true",
                   help="Run metrics on the full split while only saving --num visualizations.")
    p.add_argument("--exclude_full_gt", action=argparse.BooleanOptionalAction, default=True,
                   help="Exclude images whose GT mask covers the whole image.")
    p.add_argument("--out_dir", default="runs/eval/tp_tn_fp_50")
    return p.parse_args()


def infer_region(stem: str) -> str:
    """Infer region from GF FloodNet-style stems.

    Examples:
        gf_Australia_010_10_12 -> Australia
        gf_South_Africa_087_6_10 -> South_Africa
    """
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "gf":
        region_parts = []
        for token in parts[1:]:
            if token.isdigit():
                break
            region_parts.append(token)
        if region_parts:
            return "_".join(region_parts)
    return parts[0] if parts else stem


def select_images(images: list[Path], num: int | None, strategy: str) -> list[Path]:
    if num is None or num <= 0 or num >= len(images):
        return list(images)
    if strategy == "first":
        return images[:num]

    groups: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        groups[infer_region(image.stem)].append(image)

    selected: list[Path] = []
    region_names = sorted(groups)
    idx = 0
    while len(selected) < num:
        added = False
        for region in region_names:
            bucket = groups[region]
            if idx < len(bucket):
                selected.append(bucket[idx])
                added = True
                if len(selected) >= num:
                    break
        if not added:
            break
        idx += 1
    return selected


def filter_full_gt_images(
    images: list[Path],
    mask_dir: Path,
    mask_mode: str,
    foreground_rgb: str,
    tolerance: int,
) -> tuple[list[Path], list[dict]]:
    kept: list[Path] = []
    excluded: list[dict] = []
    for img_path in tqdm(images, desc="scan full-GT masks"):
        img = read_image(img_path)
        h, w = img.shape[:2]
        gt = read_mask_binary(
            mask_dir / f"{img_path.stem}.png",
            image_shape=(h, w),
            mask_mode=mask_mode,
            foreground_rgb=foreground_rgb,
            tolerance=tolerance,
        )
        gt_area = int(gt.sum())
        total_area = int(h * w)
        if gt_area >= total_area:
            excluded.append(
                {
                    "stem": img_path.stem,
                    "region": infer_region(img_path.stem),
                    "image_path": str(img_path),
                    "gt_area": gt_area,
                    "total_area": total_area,
                }
            )
        else:
            kept.append(img_path)
    return kept, excluded


def prediction_to_mask(result, height: int, width: int, mask_thres: float) -> np.ndarray:
    """Convert one YOLO result to a binary {0,1} mask at (height, width)."""
    pred = np.zeros((height, width), dtype=np.uint8)
    if result.masks is None or result.masks.data is None:
        return pred
    masks = result.masks.data.detach().cpu().numpy()
    for m in masks:
        resized = cv2.resize(m.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        pred |= (resized >= mask_thres).astype(np.uint8)
    return pred


def make_confusion_map(image_bgr: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Build a colored panel showing TP / FP / FN / TN pixel regions."""
    gt_b = (gt > 0).astype(np.uint8)
    pred_b = (pred > 0).astype(np.uint8)

    tp = (gt_b == 1) & (pred_b == 1)
    fp = (gt_b == 0) & (pred_b == 1)
    fn = (gt_b == 1) & (pred_b == 0)
    tn = (gt_b == 0) & (pred_b == 0)

    # Base: a desaturated copy of the image so TN shows up as a gray-ish field
    # while still being recognizable as the original photo.
    base = (image_bgr.astype(np.float32) * 0.45).astype(np.uint8)
    base[tn] = COLOR_TN_BG

    out = base.copy()
    colored = np.zeros_like(out)
    colored[tp] = COLOR_TP
    colored[fp] = COLOR_FP
    colored[fn] = COLOR_FN

    any_color = tp | fp | fn
    if any_color.any():
        blended = cv2.addWeighted(out, 0.30, colored, 0.70, 0)
        out[any_color] = blended[any_color]

    # Outlines: GT in white, Pred in cyan
    gt_contours, _ = cv2.findContours(gt_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, gt_contours, -1, (255, 255, 255), 2)
    cv2.drawContours(out, pred_contours, -1, (0, 255, 255), 1)

    return out


def add_label(canvas: np.ndarray, x: int, y: int, text: str, font_scale: float = 0.7) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 2)
    cv2.rectangle(canvas, (x, y - th - 8), (x + tw + 12, y + 6), COLOR_BOX, -1)
    cv2.putText(canvas, text, (x + 6, y - 4), font, font_scale, COLOR_TEXT, 2, cv2.LINE_AA)


def make_panel(
    image_bgr: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    metrics: dict,
    stem: str,
) -> np.ndarray:
    """Compose a 2-row panel with the image, GT, Pred, and confusion map."""
    h, w = image_bgr.shape[:2]

    # --- Row 1 sub-panels
    gt_overlay = image_bgr.copy()
    gt_bool = (gt > 0).astype(bool)
    if gt_bool.any():
        green = np.zeros_like(gt_overlay); green[:] = (0, 255, 0)
        blended = cv2.addWeighted(gt_overlay, 0.45, green, 0.55, 0)
        gt_overlay[gt_bool] = blended[gt_bool]
        gtc, _ = cv2.findContours(gt_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_overlay, gtc, -1, (255, 255, 255), 2)

    pred_overlay = image_bgr.copy()
    pred_bool = (pred > 0).astype(bool)
    if pred_bool.any():
        red = np.zeros_like(pred_overlay); red[:] = (0, 0, 255)
        blended = cv2.addWeighted(pred_overlay, 0.45, red, 0.55, 0)
        pred_overlay[pred_bool] = blended[pred_bool]
        pc, _ = cv2.findContours(pred_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(pred_overlay, pc, -1, (255, 255, 255), 2)

    confusion = make_confusion_map(image_bgr, gt, pred)

    row1 = np.concatenate([image_bgr, gt_overlay, pred_overlay, confusion], axis=1)

    # Top label strip
    label_strip_h = 36
    label_strip = np.zeros((label_strip_h, row1.shape[1], 3), dtype=np.uint8)
    add_label(label_strip, 10, 26, "Image")
    add_label(label_strip, w + 10, 26, "GT")
    add_label(label_strip, 2 * w + 10, 26, "Pred")
    add_label(label_strip, 3 * w + 10, 26, "TP/FP/FN/TN")
    row1 = np.concatenate([label_strip, row1], axis=0)

    # --- Row 2: legend + metrics
    legend_h = 90
    legend = np.zeros((legend_h, row1.shape[1], 3), dtype=np.uint8)

    swatches = [
        (10,                          "TP: GT & Pred (green)",  COLOR_TP),
        (int(row1.shape[1] * 0.28),   "FP: Pred only (red)",    COLOR_FP),
        (int(row1.shape[1] * 0.55),   "FN: GT only (blue)",     COLOR_FN),
        (int(row1.shape[1] * 0.80),   "TN: background (gray)",  COLOR_TN_BG),
    ]
    sw_size = 18
    for sx, txt, color in swatches:
        cv2.rectangle(legend, (sx, 14), (sx + sw_size, 14 + sw_size), color, -1)
        cv2.putText(legend, txt, (sx + sw_size + 8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

    m = metrics
    metric_line = (
        f"stem={stem}  "
        f"TP={m['tp']:,}  FP={m['fp']:,}  FN={m['fn']:,}  TN={m['tn']:,}  |  "
        f"IoU={m['iou']:.3f}  Dice={m['dice']:.3f}  "
        f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
    )
    cv2.putText(legend, metric_line, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

    return np.concatenate([row1, legend], axis=0)


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    model_path = Path(args.model).resolve()
    data_path = Path(args.data).resolve()
    out_dir = ensure_dir(Path(args.out_dir).resolve())

    with data_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    base = Path(data_cfg["path"])
    image_dir = base / data_cfg[args.split]
    mask_dir = base / "masks" / args.split

    images = list_files(image_dir, IMAGE_EXTS)
    excluded_full_gt: list[dict] = []
    if args.exclude_full_gt:
        images, excluded_full_gt = filter_full_gt_images(
            images,
            mask_dir,
            args.mask_mode,
            args.foreground_rgb,
            args.tolerance,
        )
        pd.DataFrame(excluded_full_gt).to_csv(
            out_dir / f"excluded_full_gt_{args.split}.csv",
            index=False,
        )
        print(f"Excluded {len(excluded_full_gt)} full-GT images from {args.split}.")

    selected_images = select_images(images, args.num, args.sample_strategy)
    selected_stems = {p.stem for p in selected_images}
    selected_index = {p.stem: i for i, p in enumerate(selected_images)}
    eval_images = images if args.eval_all else selected_images
    print(f"Running inference on {len(eval_images)} images from {image_dir}")
    print(f"Saving {len(selected_images)} visualizations with sample_strategy={args.sample_strategy}")

    model = YOLO(str(model_path))
    rows = []
    selected_rows = []

    for img_path in tqdm(eval_images, desc=f"eval {args.split}"):
        img = read_image(img_path)
        h, w = img.shape[:2]
        gt = read_mask_binary(
            mask_dir / f"{img_path.stem}.png",
            image_shape=(h, w),
            mask_mode=args.mask_mode,
            foreground_rgb=args.foreground_rgb,
            tolerance=args.tolerance,
        )
        results = model.predict(
            str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        pred = prediction_to_mask(results[0], h, w, args.mask_thres)
        pred = postprocess_mask(pred, min_area_ratio=args.min_area_ratio,
                                morph_close=args.morph_close)
        m = compute_binary_metrics(pred, gt)
        row = {
            "stem": img_path.stem,
            "region": infer_region(img_path.stem),
            "image_path": str(img_path),
            **asdict(m),
            "gt_area": int(gt.sum()),
            "pred_area": int(pred.sum()),
        }
        rows.append(row)

        if img_path.stem in selected_stems:
            selected_rows.append(row)
            idx = selected_index[img_path.stem]
            canvas = make_panel(img, gt, pred, row, img_path.stem)
            cv2.imwrite(str(out_dir / f"{idx:03d}_{img_path.stem}.jpg"), canvas)

    df = pd.DataFrame(rows)
    selected_df = pd.DataFrame(selected_rows)
    n = len(selected_df)
    selected_df.to_csv(out_dir / f"metrics_{n}.csv", index=False)
    pd.DataFrame([aggregate_metrics(selected_rows)]).to_csv(out_dir / f"summary_{n}.csv", index=False)

    selected_region_counts = (
        selected_df["region"].value_counts().rename_axis("region").reset_index(name="num_saved")
    )
    selected_region_counts.to_csv(out_dir / f"selected_region_counts_{n}.csv", index=False)

    if args.eval_all:
        df.to_csv(out_dir / f"metrics_all_{args.split}.csv", index=False)
        pd.DataFrame([aggregate_metrics(rows)]).to_csv(out_dir / f"summary_all_{args.split}.csv", index=False)
        region_summary = []
        for region, group in df.groupby("region", sort=True):
            row = {"region": region, **aggregate_metrics(group.to_dict("records"))}
            region_summary.append(row)
        pd.DataFrame(region_summary).to_csv(out_dir / f"region_metrics_{args.split}.csv", index=False)

    summary = aggregate_metrics(rows if args.eval_all else selected_rows)
    scope = f"full {args.split} split" if args.eval_all else "saved images"
    print(f"\n=== Summary over {scope} ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:>28s} = {v:.4f}")
        else:
            print(f"  {k:>28s} = {v}")
    print("\n=== Saved visualization region counts ===")
    for row in selected_region_counts.itertuples(index=False):
        print(f"  {row.region:>28s} = {row.num_saved}")
    print(f"\nAnnotated images saved to: {out_dir}")


if __name__ == "__main__":
    main()

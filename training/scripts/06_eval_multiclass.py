#!/usr/bin/env python
"""Evaluate a multi-class YOLOv8-seg model on a dataset split.

Outputs:
  - Per-class pixel metrics (precision, recall, F1, IoU, Dice)
  - Aggregate (micro / macro) metrics
  - Ultralytics built-in mAP metrics
  - Visual panels with per-class color coding
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_utils import IMAGE_EXTS, ensure_dir, list_files, read_image, read_mask_multiclass
from src.metrics import SegMetrics, compute_binary_metrics
from src.visualization import save_panel


# BGR colors for each class (class_id → BGR tuple)
CLASS_COLORS = {
    1: (0, 0, 255),    # flooded_road → red
    2: (255, 200, 0),  # water → cyan/yellow
}

CLASS_NAMES = {
    0: "background",
    1: "flooded_road",
    2: "water",
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate multi-class YOLOv8-seg model.")
    p.add_argument("--model", required=True, help="Path to best.pt")
    p.add_argument("--data", default="data/floodnet/processed/waterlogging.yaml")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--device", default=None, help="Inference device, e.g. 0, cuda:0, or cpu.")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--mask_thres", type=float, default=0.5)
    p.add_argument("--output_dir", default="runs/eval_floodnet_3class")
    p.add_argument("--visual_num", type=int, default=60, help="Number of visual panels to save.")
    return p.parse_args()


def prediction_to_multiclass_mask(result, height: int, width: int, num_classes: int, mask_thres: float) -> np.ndarray:
    """Convert YOLO prediction result to a single-channel class-index mask.

    Each mask segment is assigned its predicted class ID. If overlapping masks
    have different class IDs, the last one wins (rare in practice).
    """
    pred = np.zeros((height, width), dtype=np.uint8)
    if result.masks is None or result.masks.data is None:
        return pred
    masks = result.masks.data.detach().cpu().numpy()  # (N, H, W)
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)  # (N,)

    for m, cid in zip(masks, classes):
        if cid == 0:
            continue  # skip background predictions
        resized = cv2.resize(m.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        binary = (resized >= mask_thres).astype(np.uint8)
        pred[binary > 0] = cid
    return pred


def compute_per_class_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> Dict[str, SegMetrics]:
    """Compute binary metrics for each foreground class (one-vs-rest)."""
    metrics = {}
    for cid in range(1, num_classes):
        pred_bin = (pred == cid).astype(np.uint8)
        gt_bin = (gt == cid).astype(np.uint8)
        metrics[cid] = compute_binary_metrics(pred_bin, gt_bin)
    return metrics


def overlay_multiclass_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45, add_legend: bool = False) -> np.ndarray:
    """Overlay a multi-class mask on the image with per-class colors."""
    out = image_bgr.copy()
    for cid, color in CLASS_COLORS.items():
        region = mask == cid
        if not region.any():
            continue
        colored = np.zeros_like(out)
        colored[:, :] = color
        blended = cv2.addWeighted(out, 1 - alpha, colored, alpha, 0)
        out[region] = blended[region]
    if add_legend:
        h = image_bgr.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        lx, ly = 10, h - 10
        items = [(cid, CLASS_NAMES.get(cid, f"class{cid}"), color) for cid in CLASS_COLORS]
        max_tw = max(cv2.getTextSize(n, font, 0.45, 1)[0][0] for _, n, _ in items)
        box_w = max_tw + 40
        box_h = len(items) * 20 + 10
        cv2.rectangle(out, (lx - 5, ly - box_h + 5), (lx + box_w, ly + 10), (0, 0, 0), -1)
        for cid, name, color in items:
            cv2.rectangle(out, (lx, ly - 8), (lx + 15, ly + 5), color, -1)
            cv2.putText(out, name, (lx + 20, ly + 2), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            ly -= 20
    return out


def draw_multiclass_contours(image_bgr: np.ndarray, mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Draw contours for each class with their designated color."""
    out = image_bgr.copy()
    for cid, color in CLASS_COLORS.items():
        layer = (mask == cid).astype(np.uint8)
        contours, _ = cv2.findContours(layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, thickness)
    return out


def comparison_panel_multiclass(
    image_bgr: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """Create a comparison panel showing per-class TP/FP/FN with color coding."""
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()

    # Dim the base image
    dimmed = (image_bgr * (1 - alpha)).astype(np.uint8)
    overlay = np.zeros_like(image_bgr)

    for cid in range(1, max(gt.max(), pred.max()) + 1):
        gt_bin = gt == cid
        pred_bin = pred == cid
        tp = gt_bin & pred_bin
        fp = ~gt_bin & pred_bin
        fn = gt_bin & ~pred_bin

        # TP: class color, FP: red tint, FN: blue tint
        base_color = np.array(CLASS_COLORS.get(cid, (0, 255, 0)), dtype=np.uint8)
        if tp.any():
            overlay[tp] = base_color
        if fp.any():
            overlay[fp] = (0, 0, 255)  # red for FP
        if fn.any():
            overlay[fn] = (255, 100, 0)  # blue for FN

    mask_bool = np.any(overlay > 0, axis=-1)
    if mask_bool.any():
        blended = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)
        out[mask_bool] = blended[mask_bool]
    else:
        out = dimmed

    # Draw contours: GT = white, Pred = class color
    for cid, color in CLASS_COLORS.items():
        # GT contours
        gt_layer = (gt == cid).astype(np.uint8)
        gt_contours, _ = cv2.findContours(gt_layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, gt_contours, -1, (255, 255, 255), 2)
        # Pred contours
        pred_layer = (pred == cid).astype(np.uint8)
        pred_contours, _ = cv2.findContours(pred_layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, pred_contours, -1, color, 1)

    # Legend: GT vs Pred contour + TP/FP/FN
    font = cv2.FONT_HERSHEY_SIMPLEX
    lx, ly = 10, h - 10
    legend_items = [
        ((255, 255, 255), "GT contour", 2),
        ((0, 200, 0), "Pred contour", 1),
        ((0, 0, 255), "FP", -1),
        ((255, 100, 0), "FN", -1),
    ]
    # background box
    max_text_w = 0
    for _, text, _ in legend_items:
        ts, _ = cv2.getTextSize(text, font, 0.45, 1)
        max_text_w = max(max_text_w, ts[0])
    box_w = max_text_w + 40
    box_h = len(legend_items) * 20 + 10
    cv2.rectangle(out, (lx - 5, ly - box_h + 5), (lx + box_w, ly + 10), (0, 0, 0), -1)

    for color, text, thickness in legend_items:
        if thickness == -1:
            cv2.rectangle(out, (lx, ly - 8), (lx + 15, ly + 5), color, -1)
        else:
            cv2.line(out, (lx, ly - 2), (lx + 15, ly - 2), color, thickness)
        cv2.putText(out, text, (lx + 20, ly + 2), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        ly -= 20

    return out


def make_multiclass_panel(
    image_bgr: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
) -> np.ndarray:
    """Build a 4-panel image: original | GT overlay | Pred overlay | Comparison."""
    h, w = image_bgr.shape[:2]

    gt_overlay = overlay_multiclass_mask(image_bgr, gt, alpha=0.45, add_legend=True)
    gt_overlay = draw_multiclass_contours(gt_overlay, gt, thickness=2)
    pred_overlay = overlay_multiclass_mask(image_bgr, pred, alpha=0.45, add_legend=True)
    pred_overlay = draw_multiclass_contours(pred_overlay, pred, thickness=2)
    comp = comparison_panel_multiclass(image_bgr, gt, pred)

    panels = [image_bgr, gt_overlay, pred_overlay, comp]
    labels = ["Image", "GT", "Pred", "TP/FP/FN"]
    canvas = np.concatenate(panels, axis=1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, text in enumerate(labels):
        x = i * w + 10
        text_size, _ = cv2.getTextSize(text, font, 0.7, 2)
        cv2.rectangle(canvas, (x - 5, 5), (x + text_size[0] + 12, 35), (0, 0, 0), -1)
        cv2.putText(canvas, text, (x, 26), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Legend in bottom-right of last panel (class names + TP/FP/FN)
    legend_x = 3 * w + 10
    legend_y = h - 10
    legend_items = []
    for cid in sorted(CLASS_COLORS.keys()):
        name = CLASS_NAMES.get(cid, f"class{cid}")
        color = CLASS_COLORS[cid]
        legend_items.append((color, name, "rect"))
    # Add TP / FP / FN
    for cid in sorted(CLASS_COLORS.keys()):
        name = CLASS_NAMES.get(cid, f"class{cid}")
        color = CLASS_COLORS[cid]
        legend_items.append((color, f"TP({name})", "rect"))
    legend_items.append(((0, 0, 255), "FP", "rect"))
    legend_items.append(((255, 100, 0), "FN", "rect"))

    max_tw = max(cv2.getTextSize(t, font, 0.45, 1)[0][0] for _, t, _ in legend_items)
    box_w = max_tw + 40
    box_h = len(legend_items) * 20 + 10
    cv2.rectangle(canvas, (legend_x - 5, legend_y - box_h + 5), (legend_x + box_w, legend_y + 10), (0, 0, 0), -1)
    for color, text, _ in legend_items:
        cv2.rectangle(canvas, (legend_x, legend_y - 8), (legend_x + 15, legend_y + 5), color, -1)
        cv2.putText(canvas, text, (legend_x + 20, legend_y + 2), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        legend_y -= 20

    return canvas


def main():
    args = parse_args()
    from ultralytics import YOLO

    model_path = Path(args.model).resolve()
    data_path = Path(args.data).resolve()
    out_dir = ensure_dir(Path(args.output_dir).resolve())

    with data_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    base = Path(data_cfg["path"])
    image_dir = base / data_cfg[args.split]
    mask_dir = base / "masks" / args.split

    vis_dir = ensure_dir(out_dir / "visual_results")
    images = list_files(image_dir, IMAGE_EXTS)
    model = YOLO(model_path)

    # Per-image rows with per-class metrics
    rows: List[dict] = []

    for idx, img_path in enumerate(tqdm(images, desc=f"eval {args.split}")):
        img = read_image(img_path)
        h, w = img.shape[:2]
        gt = read_mask_multiclass(mask_dir / f"{img_path.stem}.png", num_classes=args.num_classes, image_shape=(h, w))

        results = model.predict(
            str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        pred = prediction_to_multiclass_mask(results[0], h, w, args.num_classes, args.mask_thres)

        # Per-class metrics
        per_class = compute_per_class_metrics(pred, gt, args.num_classes)
        row = {"stem": img_path.stem, "image_path": str(img_path)}
        for cid, m in per_class.items():
            name = CLASS_NAMES.get(cid, f"class{cid}")
            for k, v in m.__dict__.items():
                row[f"{name}_{k}"] = v
            row[f"{name}_gt_area"] = int((gt == cid).sum())
            row[f"{name}_pred_area"] = int((pred == cid).sum())
        rows.append(row)

        # Save visual panel
        if idx < args.visual_num:
            panel = make_multiclass_panel(img, gt, pred)
            cv2.imwrite(str(vis_dir / f"{img_path.stem}.jpg"), panel)

    # Save per-image metrics
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"metrics_{args.split}.csv", index=False)

    # Print per-class summary
    print("\n" + "=" * 70)
    print(f"  Per-Class Pixel Metrics — {args.split} split ({len(images)} images)")
    print("=" * 70)
    for cid in range(1, args.num_classes):
        name = CLASS_NAMES.get(cid, f"class{cid}")
        tp = int(df[f"{name}_tp"].sum())
        fp = int(df[f"{name}_fp"].sum())
        fn = int(df[f"{name}_fn"].sum())
        eps = 1e-8
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        macro_iou = float(df[f"{name}_iou"].mean())
        gt_pixels = int(df[f"{name}_gt_area"].sum())
        pred_pixels = int(df[f"{name}_pred_area"].sum())
        n_gt_pos = int((df[f"{name}_gt_area"] > 0).sum())
        n_pred_pos = int((df[f"{name}_pred_area"] > 0).sum())

        print(f"\n  [{name}] (class_id={cid})")
        print(f"    Micro  — Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  IoU: {iou:.4f}  Dice: {dice:.4f}")
        print(f"    Macro  — IoU: {macro_iou:.4f}")
        print(f"    Pixels — GT: {gt_pixels:,}  Pred: {pred_pixels:,}")
        print(f"    Images — GT positive: {n_gt_pos}  Pred positive: {n_pred_pos}")

    # Save summary
    summary = {}
    for cid in range(1, args.num_classes):
        name = CLASS_NAMES.get(cid, f"class{cid}")
        tp = int(df[f"{name}_tp"].sum())
        fp = int(df[f"{name}_fp"].sum())
        fn = int(df[f"{name}_fn"].sum())
        eps = 1e-8
        summary[f"{name}_precision"] = tp / (tp + fp + eps)
        summary[f"{name}_recall"] = tp / (tp + fn + eps)
        summary[f"{name}_f1"] = 2 * summary[f"{name}_precision"] * summary[f"{name}_recall"] / (summary[f"{name}_precision"] + summary[f"{name}_recall"] + eps)
        summary[f"{name}_iou"] = tp / (tp + fp + fn + eps)
        summary[f"{name}_dice"] = 2 * tp / (2 * tp + fp + fn + eps)
        summary[f"{name}_macro_iou"] = float(df[f"{name}_iou"].mean())

    pd.DataFrame([summary]).to_csv(out_dir / f"summary_{args.split}.csv", index=False)

    # Ultralytics built-in val for mAP
    print("\n" + "=" * 70)
    print("  Ultralytics mAP Validation")
    print("=" * 70)
    try:
        yolo_val = model.val(
            data=str(data_path),
            split=args.split,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
        )
        val_text = str(yolo_val)
        (out_dir / f"yolo_val_{args.split}.txt").write_text(val_text, encoding="utf-8")
    except Exception as e:
        print(f"Warning: Ultralytics built-in val failed: {e}")

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()

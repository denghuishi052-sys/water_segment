#!/usr/bin/env python
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

from src.dataset_utils import IMAGE_EXTS, ensure_dir, list_files, read_image, read_mask_binary
from src.mask_utils import postprocess_mask
from src.metrics import aggregate_metrics, compute_binary_metrics
from src.visualization import save_panel


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained YOLOv8-seg model on a split with pixel metrics.")
    p.add_argument("--model", required=True, help="Path to best.pt")
    p.add_argument("--data", default="data/processed/waterlogging.yaml")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="Inference device, e.g. 0, cuda:0, or cpu.")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--mask_thres", type=float, default=0.5)
    p.add_argument("--min_area_ratio", type=float, default=0.0005)
    p.add_argument("--morph_close", action="store_true", default=True)
    p.add_argument("--mask_mode", default="auto", choices=["red", "non_black", "grayscale", "auto"])
    p.add_argument("--foreground_rgb", default="128,0,0")
    p.add_argument("--tolerance", type=int, default=10)
    p.add_argument("--output_dir", default="runs/eval")
    p.add_argument("--visual_num", type=int, default=100)
    return p.parse_args()


def prediction_to_mask(result, height: int, width: int, mask_thres: float) -> np.ndarray:
    pred = np.zeros((height, width), dtype=np.uint8)
    if result.masks is None or result.masks.data is None:
        return pred
    masks = result.masks.data.detach().cpu().numpy()
    for m in masks:
        resized = cv2.resize(m.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        pred |= (resized >= mask_thres).astype(np.uint8)
    return pred


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
    bad_low_iou = ensure_dir(out_dir / "bad_cases" / "low_iou")
    bad_fp = ensure_dir(out_dir / "bad_cases" / "false_positive")
    bad_fn = ensure_dir(out_dir / "bad_cases" / "false_negative")

    images = list_files(image_dir, IMAGE_EXTS)
    model = YOLO(model_path)
    rows: List[dict] = []

    for idx, img_path in enumerate(tqdm(images, desc=f"eval {args.split}")):
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
        pred = postprocess_mask(pred, min_area_ratio=args.min_area_ratio, morph_close=args.morph_close)
        m = compute_binary_metrics(pred, gt)
        row = {
            "stem": img_path.stem,
            "image_path": str(img_path),
            **m.__dict__,
            "gt_area": int(gt.sum()),
            "pred_area": int(pred.sum()),
        }
        if row["gt_area"] == 0 and row["pred_area"] == 0:
            row["case_type"] = "true_negative"
        elif row["gt_area"] == 0 and row["pred_area"] > 0:
            row["case_type"] = "false_positive"
        elif row["gt_area"] > 0 and row["pred_area"] == 0:
            row["case_type"] = "false_negative"
        else:
            row["case_type"] = "foreground"
        rows.append(row)

        if idx < args.visual_num:
            save_panel(vis_dir / f"{img_path.stem}.jpg", img, gt=gt, pred=pred)
        # hard case categories
        if (row["gt_area"] > 0 or row["pred_area"] > 0) and m.iou < 0.4:
            save_panel(bad_low_iou / f"{img_path.stem}_iou{m.iou:.3f}.jpg", img, gt=gt, pred=pred)
        if row["gt_area"] == 0 and row["pred_area"] > 0:
            save_panel(bad_fp / f"{img_path.stem}.jpg", img, gt=gt, pred=pred)
        if row["gt_area"] > 0 and row["pred_area"] == 0:
            save_panel(bad_fn / f"{img_path.stem}.jpg", img, gt=gt, pred=pred)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"metrics_{args.split}.csv", index=False)
    summary = aggregate_metrics(rows)
    pd.DataFrame([summary]).to_csv(out_dir / f"summary_{args.split}.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Evaluation saved to: {out_dir}")

    # Also run built-in YOLO val to get mAP if labels are present.
    try:
        yolo_val = model.val(
            data=str(data_path),
            split=args.split,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        (out_dir / f"yolo_val_{args.split}.txt").write_text(str(yolo_val), encoding="utf-8")
    except Exception as e:
        print(f"Warning: Ultralytics built-in val failed: {e}")


if __name__ == "__main__":
    main()

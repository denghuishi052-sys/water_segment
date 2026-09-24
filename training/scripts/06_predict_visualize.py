#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_utils import IMAGE_EXTS, ensure_dir, list_files, read_image, read_mask_binary
from src.mask_utils import postprocess_mask
from src.visualization import save_panel


def parse_args():
    p = argparse.ArgumentParser(description="Predict and save visualization panels.")
    p.add_argument("--model", required=True)
    p.add_argument("--image_dir", default="data/processed/images/test")
    p.add_argument("--mask_dir", default="data/processed/masks/test")
    p.add_argument("--output_dir", default="runs/visual_test")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="Inference device, e.g. 0, cuda:0, or cpu.")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--mask_thres", type=float, default=0.5)
    p.add_argument("--mask_mode", default="auto", choices=["red", "non_black", "grayscale", "auto"])
    p.add_argument("--foreground_rgb", default="128,0,0")
    p.add_argument("--tolerance", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def prediction_to_mask(result, height: int, width: int, mask_thres: float) -> np.ndarray:
    pred = np.zeros((height, width), dtype=np.uint8)
    if result.masks is None or result.masks.data is None:
        return pred
    for m in result.masks.data.detach().cpu().numpy():
        resized = cv2.resize(m.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        pred |= (resized >= mask_thres).astype(np.uint8)
    return pred


def main():
    args = parse_args()
    from ultralytics import YOLO

    image_dir = Path(args.image_dir).resolve()
    mask_dir = Path(args.mask_dir).resolve() if args.mask_dir else None
    model_path = Path(args.model).resolve()
    out_dir = ensure_dir(Path(args.output_dir).resolve())
    images = list_files(image_dir, IMAGE_EXTS)
    random.seed(args.seed)
    if args.num_samples > 0 and len(images) > args.num_samples:
        images = random.sample(images, args.num_samples)
    model = YOLO(model_path)
    gt_loaded = 0
    gt_non_empty = 0

    for img_path in tqdm(images, desc="visualizing"):
        img = read_image(img_path)
        h, w = img.shape[:2]
        gt_path = mask_dir / f"{img_path.stem}.png" if mask_dir is not None else None
        gt = (
            read_mask_binary(
                gt_path,
                image_shape=(h, w),
                mask_mode=args.mask_mode,
                foreground_rgb=args.foreground_rgb,
                tolerance=args.tolerance,
            )
            if gt_path is not None and gt_path.exists()
            else None
        )
        if gt is not None:
            gt_loaded += 1
            gt_non_empty += int(gt.sum() > 0)
        elif mask_dir is not None:
            print(f"Warning: GT mask not found for {img_path.name}: {gt_path}")
        result = model.predict(
            str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]
        pred = prediction_to_mask(result, h, w, args.mask_thres)
        pred = postprocess_mask(pred)
        save_panel(out_dir / f"{img_path.stem}.jpg", img, gt=gt, pred=pred)
    print(f"GT masks loaded: {gt_loaded}/{len(images)}, non-empty: {gt_non_empty}")
    print(f"Visualization saved to: {out_dir}")


if __name__ == "__main__":
    main()

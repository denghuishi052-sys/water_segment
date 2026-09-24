#!/usr/bin/env python
from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.yolo_utils import train_yolo_from_config


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 segmentation model from YAML config.")
    p.add_argument("--config", default="configs/train_yolov8s.yaml")
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the epoch count in the YAML, useful for smoke runs.",
    )
    p.add_argument("--name", default=None, help="Override the output run name.")
    p.add_argument(
        "--resume_from",
        default=None,
        help="Resume training from a saved Ultralytics last.pt checkpoint.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.name is not None:
        overrides["name"] = args.name
    if args.resume_from is not None:
        checkpoint = Path(args.resume_from)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        overrides["model"] = str(checkpoint)
        overrides["resume"] = True
    result = train_yolo_from_config(args.config, overrides=overrides)
    print(result)


if __name__ == "__main__":
    main()

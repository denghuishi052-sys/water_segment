#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.yolo_utils import train_yolo_from_config


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 model on hard samples.")
    p.add_argument("--config", default="configs/train_hard_finetune.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    result = train_yolo_from_config(args.config)
    print(result)


if __name__ == "__main__":
    main()

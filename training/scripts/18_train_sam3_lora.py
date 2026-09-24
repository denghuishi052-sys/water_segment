"""Train the minimal SAM 3 visual Q/V LoRA adapter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.sam3_lora.trainer import train_lora


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="data/sam3_finetune/manifest_train_val.jsonl"
    )
    parser.add_argument(
        "--checkpoint",
        default="D:/BaiduNetdiskDownload/课程- 权重(1)/sam3.pt",
    )
    parser.add_argument("--output", default="runs/sam3_finetune/lora")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    args = parser.parse_args()
    report = train_lora(
        args.manifest,
        args.checkpoint,
        args.output,
        device=args.device,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        crop_size=args.crop_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    Path(args.output, "training_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

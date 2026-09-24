"""Build the train/val-only SAM 3 fine-tuning manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.sam3_lora.data import build_manifest_records, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default="data/floodnet_binary_multiscale_1024/split_report/"
        "multiscale_crop_report.csv",
    )
    parser.add_argument(
        "--processed",
        default="data/floodnet_binary_multiscale_1024/processed",
    )
    parser.add_argument(
        "--output",
        default="data/sam3_finetune/manifest_train_val.jsonl",
    )
    args = parser.parse_args()
    records, summary = build_manifest_records(args.report, args.processed)
    write_jsonl(records, args.output)
    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train and export the Base/LoRA-specific proposal selector."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.sam3_lora.selector_training import load_rows, train_selector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proposals", default="data/sam3_finetune/base_proposals.jsonl"
    )
    parser.add_argument(
        "--output", default="runs/sam3_finetune/base_selector.json"
    )
    parser.add_argument("--precision-tolerance", type=float, default=0.01)
    args = parser.parse_args()
    rows = load_rows(args.proposals)
    if any(row["split"] == "test" for row in rows):
        raise ValueError("Test proposals are forbidden during selector training")
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError("Both train and val proposal rows are required")
    payload, report = train_selector(
        train_rows,
        val_rows,
        precision_tolerance=args.precision_tolerance,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output.with_suffix(".metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["selected"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

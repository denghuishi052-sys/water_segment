"""Cache YOLO masks and Base/LoRA SAM 3 proposals for train and val."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.sam3_lora.data import read_jsonl, write_jsonl
from waterseg_platform.config import load_config
from waterseg_platform.image_io import read_image, read_mask_binary
from waterseg_platform.pipeline import SegmentationService
from waterseg_platform.sam3_refinement import (
    Sam3WorkerClient,
    extract_candidates,
    proposal_features,
)
from waterseg_platform.sam3_selector import selector_feature_vector
from waterseg_platform.sam3_selector import FEATURE_NAMES


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left) > 0
    right = np.asarray(right) > 0
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum()) / max(int(union), 1)


def apply_candidate_proposal(
    yolo: np.ndarray,
    proposal: np.ndarray,
    candidate_mask: np.ndarray | None,
) -> np.ndarray:
    output = (np.asarray(yolo) > 0).astype(np.uint8)
    if candidate_mask is not None:
        output[np.asarray(candidate_mask) > 0] = 0
    output |= (np.asarray(proposal) > 0).astype(np.uint8)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="data/sam3_finetune/manifest_train_val.jsonl"
    )
    parser.add_argument(
        "--config", default="configs/onnx_platform_sam3_trial.yaml"
    )
    parser.add_argument(
        "--output", default="data/sam3_finetune/base_proposals.jsonl"
    )
    parser.add_argument(
        "--cache-dir", default="data/sam3_finetune/base_cache"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing JSONL and skip already cached output_stem values.",
    )
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    if any(item["split"] == "test" for item in records):
        raise ValueError("Test records are forbidden in proposal caching")
    if args.limit:
        records = records[: args.limit]
    config = load_config(args.config)
    service = SegmentationService(config)
    worker = Sam3WorkerClient(config)
    cache_root = Path(args.cache_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_stems = set()
    rows_written = 0
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as existing:
            for line in existing:
                if not line.strip():
                    continue
                row = json.loads(line)
                processed_stems.add(row["output_stem"])
                rows_written += 1
    elif output_path.exists():
        output_path.unlink()
    mode = "a" if args.resume else "w"
    try:
        with output_path.open(mode, encoding="utf-8") as output_file:
            for sample_index, sample in enumerate(
                tqdm(records, desc="proposal cache", unit="image")
            ):
                if sample["output_stem"] in processed_stems:
                    continue
                image = read_image(Path(sample["image"]))
                gt = read_mask_binary(
                    Path(sample["mask"]),
                    image_shape=image.shape[:2],
                    mask_mode="grayscale",
                    tolerance=0,
                )
                yolo, _ = service.segment_array(image, sam3_enabled=False)
                candidates = extract_candidates(
                    yolo,
                    config.sam3_min_component_area_ratio,
                    config.sam3_max_candidates,
                    config.sam3_box_margin_ratio,
                    config.sam3_positive_points_per_component,
                    config.sam3_negative_points_per_component,
                    config.sam3_negative_ring_ratio,
                )
                response = worker.predict(
                    image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    candidates=candidates,
                    text_prompts=config.sam3_text_prompts,
                    mode="balanced",
                    local_enabled=True,
                    global_enabled=True,
                )
                sample_dir = cache_root / sample["split"] / sample["output_stem"]
                sample_dir.mkdir(parents=True, exist_ok=True)
                yolo_path = sample_dir / "yolo.png"
                cv2.imwrite(str(yolo_path), yolo * 255)
                gt_path = Path(sample["mask"]).resolve()
                base_iou = mask_iou(yolo, gt)
                sample_rows = []
                for proposal_index, item in enumerate(response["proposals"]):
                    mask = (np.asarray(item["mask"]) > 0).astype(np.uint8)
                    candidate_index = item.get("candidate_index")
                    candidate = (
                        candidates[int(candidate_index)]
                        if candidate_index is not None
                        else None
                    )
                    reference = (
                        candidate.component_mask if candidate is not None else yolo
                    )
                    features = proposal_features(mask, reference, item["score"])
                    applied = apply_candidate_proposal(
                        yolo,
                        mask,
                        candidate.component_mask if candidate is not None else None,
                    )
                    proposal_path = sample_dir / f"proposal_{proposal_index:03d}.png"
                    applied_path = sample_dir / f"applied_{proposal_index:03d}.png"
                    candidate_path = None
                    if candidate is not None:
                        candidate_path = (
                            sample_dir / f"candidate_{int(candidate_index):03d}.png"
                        )
                        if not candidate_path.exists():
                            cv2.imwrite(
                                str(candidate_path),
                                candidate.component_mask * 255,
                            )
                    cv2.imwrite(str(proposal_path), mask * 255)
                    cv2.imwrite(str(applied_path), applied * 255)
                    vector = selector_feature_vector(
                        features,
                        source=item["source"],
                        candidate_area_ratio=(
                            candidate.area / float(max(yolo.size, 1))
                            if candidate is not None
                            else 0.0
                        ),
                        yolo_area_ratio=float(yolo.mean()),
                        yolo_confidence=float(config.conf),
                    )
                    delta_iou = mask_iou(applied, gt) - base_iou
                    sample_rows.append(
                        {
                            "sample_id": sample_index,
                            "image": sample["image"],
                            "gt_mask": str(gt_path),
                            "source_stem": sample["source_stem"],
                            "output_stem": sample["output_stem"],
                            "split": sample["split"],
                            "crop_type": sample["crop_type"],
                            "prompt": item.get("prompt"),
                            "source": item["source"],
                            "candidate_index": candidate_index,
                            "yolo_mask": str(yolo_path.resolve()),
                            "candidate_mask": (
                                str(candidate_path.resolve())
                                if candidate_path is not None
                                else None
                            ),
                            "proposal_mask": str(proposal_path.resolve()),
                            "applied_mask": str(applied_path.resolve()),
                            "base_iou": base_iou,
                            "delta_iou": delta_iou,
                            "beneficial": bool(delta_iou > 0.0),
                            "features": {
                                name: float(value)
                                for name, value in zip(
                                    FEATURE_NAMES,
                                    vector,
                                )
                            },
                        }
                    )
                for row in sample_rows:
                    output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_file.flush()
                rows_written += len(sample_rows)
                processed_stems.add(sample["output_stem"])
                Path(args.output).with_suffix(".summary.json").write_text(
                    json.dumps(
                        {
                            "samples": len(records),
                            "completed_samples": len(processed_stems),
                            "proposals": rows_written,
                            "adapter_enabled": config.sam3_adapter_enabled,
                            "test_images_read": 0,
                            "complete": False,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
    finally:
        worker.close()
    Path(args.output).with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "samples": len(records),
                "completed_samples": len(processed_stems),
                "proposals": rows_written,
                "adapter_enabled": config.sam3_adapter_enabled,
                "test_images_read": 0,
                "complete": len(processed_stems) == len(records),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

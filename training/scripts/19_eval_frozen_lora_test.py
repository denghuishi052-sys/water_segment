"""Run the sealed final test once for YOLO vs frozen SAM3-LoRA selector.

This script intentionally evaluates only the frozen candidate configuration,
not multiple SAM3 modes, so the sealed test split is not used for mode/threshold
selection.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waterseg_platform.config import load_config  # noqa: E402
from waterseg_platform.image_io import read_image, read_mask_binary  # noqa: E402
from waterseg_platform.metrics import aggregate_metrics, compute_binary_metrics  # noqa: E402
from waterseg_platform.pipeline import SegmentationService  # noqa: E402


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    kernel = np.ones((3, 3), dtype=np.uint8)
    pred_edge = cv2.morphologyEx(
        (pred > 0).astype(np.uint8), cv2.MORPH_GRADIENT, kernel
    )
    gt_edge = cv2.morphologyEx(
        (gt > 0).astype(np.uint8), cv2.MORPH_GRADIENT, kernel
    )
    radius = tolerance * 2 + 1
    tol_kernel = np.ones((radius, radius), dtype=np.uint8)
    pred_hit = np.logical_and(pred_edge, cv2.dilate(gt_edge, tol_kernel) > 0).sum()
    gt_hit = np.logical_and(gt_edge, cv2.dilate(pred_edge, tol_kernel) > 0).sum()
    precision = pred_hit / max(int(pred_edge.sum()), 1)
    recall = gt_hit / max(int(gt_edge.sum()), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-8))


def small_object_recall(pred: np.ndarray, gt: np.ndarray, max_area_ratio: float = 0.01) -> float:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (gt > 0).astype(np.uint8), connectivity=8
    )
    max_area = gt.size * max_area_ratio
    total = recovered = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > max_area:
            continue
        total += 1
        component = labels == label
        if np.logical_and(component, pred > 0).sum() / max(area, 1) >= 0.5:
            recovered += 1
    return float(recovered / total) if total else float("nan")


def fp_image(row: dict) -> bool:
    return int(row["gt_area"]) == 0 and int(row["pred_area"]) > 0


def fn_image(row: dict) -> bool:
    return int(row["gt_area"]) > 0 and int(row["pred_area"]) == 0


def summarize(rows: list[dict], yolo_rows: list[dict] | None = None) -> dict:
    summary = aggregate_metrics(rows)
    df = pd.DataFrame(rows)
    summary["mean_elapsed_ms"] = float(df["elapsed_ms"].mean())
    summary["fp_images"] = int(sum(fp_image(row) for row in rows))
    summary["fn_images"] = int(sum(fn_image(row) for row in rows))
    for metric in (
        "boundary_f1",
        "false_positive_area_ratio",
        "false_negative_area_ratio",
        "small_object_recall",
        "peak_gpu_memory_mb",
    ):
        summary[f"mean_{metric}"] = float(df[metric].dropna().mean())
    if yolo_rows is not None:
        yolo_by_stem = {row["stem"]: row for row in yolo_rows}
        eligible = [
            row
            for row in rows
            if yolo_by_stem[row["stem"]]["pred_area"] == 0 and row["gt_area"] > 0
        ]
        summary["yolo_empty_recovery_rate"] = (
            sum(row["pred_area"] > 0 for row in eligible) / len(eligible)
            if eligible
            else float("nan")
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/onnx_platform_sam3_lora_frozen.yaml"
    )
    parser.add_argument(
        "--data-root",
        default="data/floodnet_binary_multiscale_1024/processed",
    )
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument(
        "--output-dir",
        default="runs/final_test/sam3_lora_selector_frozen",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    image_dir = data_root / "images" / args.split
    mask_dir = data_root / "masks" / args.split
    images = sorted(image_dir.glob("*"))
    if not images:
        raise FileNotFoundError(f"No test images found in {image_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    service = SegmentationService(config)
    yolo_rows: list[dict] = []
    frozen_rows: list[dict] = []

    try:
        for image_path in tqdm(images, desc="final frozen test", unit="img"):
            image = read_image(image_path)
            gt = read_mask_binary(
                mask_dir / f"{image_path.stem}.png",
                image_shape=image.shape[:2],
                mask_mode="grayscale",
                tolerance=0,
            )
            for name, enabled in (
                ("yolo_only", False),
                ("sam3_lora_selector", True),
            ):
                started = time.perf_counter()
                pred, info = service.segment_array(
                    image,
                    sam3_enabled=enabled,
                    sam3_mode=config.sam3_mode,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                row = asdict(compute_binary_metrics(pred, gt))
                row.update(
                    {
                        "stem": image_path.stem,
                        "gt_area": int(gt.sum()),
                        "pred_area": int(pred.sum()),
                        "elapsed_ms": elapsed_ms,
                        "boundary_f1": boundary_f1(pred, gt),
                        "false_positive_area_ratio": row["fp"] / max(pred.size, 1),
                        "false_negative_area_ratio": row["fn"] / max(pred.size, 1),
                        "small_object_recall": small_object_recall(pred, gt),
                        "peak_gpu_memory_mb": info.get("sam3", {}).get(
                            "peak_gpu_memory_mb"
                        ),
                        "fallback_stage": info.get("sam3", {}).get(
                            "fallback_stage"
                        ),
                        "sam3_fallback": bool(
                            info.get("sam3", {}).get("fallback", False)
                        ),
                        "selector_loaded": bool(
                            info.get("sam3", {}).get("selector_loaded", False)
                        ),
                        "selector_error": info.get("sam3", {}).get(
                            "selector_error"
                        ),
                        "adapter_loaded": bool(
                            info.get("sam3", {}).get("adapter_loaded", False)
                        ),
                        "adapter_error": info.get("sam3", {}).get(
                            "adapter_error"
                        ),
                        "candidate_count": info.get("sam3", {}).get(
                            "candidate_count"
                        ),
                        "accepted": info.get("sam3", {}).get("accepted"),
                        "rejected": info.get("sam3", {}).get("rejected"),
                        "sam3_error": info.get("sam3", {}).get("error"),
                    }
                )
                if name == "yolo_only":
                    yolo_rows.append(row)
                else:
                    frozen_rows.append(row)
    finally:
        refiner = getattr(service, "_sam3_refiner", None)
        backend = getattr(refiner, "_backend", None)
        if hasattr(backend, "close"):
            backend.close()

    pd.DataFrame(yolo_rows).to_csv(output_dir / "yolo_only_metrics.csv", index=False)
    pd.DataFrame(frozen_rows).to_csv(
        output_dir / "sam3_lora_selector_metrics.csv", index=False
    )
    summary = {
        "config": str(Path(args.config).resolve()),
        "split": args.split,
        "test_images": len(images),
        "yolo_only": summarize(yolo_rows),
        "sam3_lora_selector": summarize(frozen_rows, yolo_rows=yolo_rows),
        "test_images_read": len(images),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

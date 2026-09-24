"""Evaluate YOLO and both SAM 3 refinement modes on the 60-image test split."""
from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waterseg_platform.config import load_config  # noqa: E402
from waterseg_platform.image_io import read_image, read_mask_binary  # noqa: E402
from waterseg_platform.metrics import aggregate_metrics, compute_binary_metrics  # noqa: E402
from waterseg_platform.pipeline import SegmentationService  # noqa: E402


DATA_ROOT = (
    ROOT / "data" / "floodnet_binary_multiscale_1024" / "processed"
)
OUTPUT_DIR = ROOT / "runs" / "platform_compare" / "sam3_refinement_test60"


def boundary_f1(pred, gt, tolerance=2):
    kernel = np.ones((3, 3), dtype=np.uint8)
    pred_edge = cv2.morphologyEx(
        (pred > 0).astype(np.uint8), cv2.MORPH_GRADIENT, kernel
    )
    gt_edge = cv2.morphologyEx(
        (gt > 0).astype(np.uint8), cv2.MORPH_GRADIENT, kernel
    )
    radius = tolerance * 2 + 1
    tol_kernel = np.ones((radius, radius), dtype=np.uint8)
    pred_hit = np.logical_and(
        pred_edge, cv2.dilate(gt_edge, tol_kernel) > 0
    ).sum()
    gt_hit = np.logical_and(
        gt_edge, cv2.dilate(pred_edge, tol_kernel) > 0
    ).sum()
    precision = pred_hit / max(int(pred_edge.sum()), 1)
    recall = gt_hit / max(int(gt_edge.sum()), 1)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def small_object_recall(pred, gt, max_area_ratio=0.01):
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
    return recovered / total if total else float("nan")


def main() -> int:
    image_dir = DATA_ROOT / "images" / "test"
    mask_dir = DATA_ROOT / "masks" / "test"
    images = sorted(image_dir.glob("*"))
    if not images:
        raise FileNotFoundError(f"No test images found in {image_dir}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config(str(ROOT / "configs" / "onnx_platform_sam3_trial.yaml"))
    service = SegmentationService(config)
    modes = [
        ("yolo_only", False, "conservative"),
        ("sam3_conservative", True, "conservative"),
        ("sam3_balanced", True, "balanced"),
        ("sam3_open", True, "open"),
    ]
    rows = {name: [] for name, _, _ in modes}
    failure_rows = []

    for image_path in tqdm(images, desc="SAM3 eval", unit="img"):
        image = read_image(image_path)
        gt = read_mask_binary(
            mask_dir / f"{image_path.stem}.png",
            image_shape=image.shape[:2],
            mask_mode="grayscale",
            tolerance=0,
        )
        image_results = {}
        for name, enabled, mode in modes:
            started = time.perf_counter()
            pred, info = service.segment_array(
                image,
                sam3_enabled=enabled,
                sam3_mode=mode,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            row = asdict(compute_binary_metrics(pred, gt))
            row.update(
                {
                    "stem": image_path.stem,
                    "gt_area": int(gt.sum()),
                    "pred_area": int(pred.sum()),
                    "elapsed_ms": elapsed_ms,
                    "sam3_fallback": bool(
                        info.get("sam3", {}).get("fallback", False)
                    ),
                    "boundary_f1": boundary_f1(pred, gt),
                    "false_positive_area_ratio": row["fp"] / max(pred.size, 1),
                    "false_negative_area_ratio": row["fn"] / max(pred.size, 1),
                    "small_object_recall": small_object_recall(pred, gt),
                    "peak_gpu_memory_mb": info.get("sam3", {}).get(
                        "peak_gpu_memory_mb"
                    ),
                }
            )
            rows[name].append(row)
            image_results[name] = {"pred": pred, "row": row, "info": info}

        yolo_iou = image_results["yolo_only"]["row"]["iou"]
        for name in ("sam3_conservative", "sam3_balanced", "sam3_open"):
            result = image_results[name]
            sam_iou = result["row"]["iou"]
            pred = result["pred"]
            category = None
            if sam_iou + 0.02 < yolo_iou:
                category = "yolo_better_than_sam3"
            elif sam_iou > yolo_iou + 0.02:
                category = "sam3_better_than_yolo"
            elif not gt.any() and pred.any():
                category = "global_false_positive"
            elif (
                image_results["yolo_only"]["pred"].any()
                and pred.sum()
                > image_results["yolo_only"]["pred"].sum() * 4
            ):
                category = "local_over_expansion"
            elif (
                not image_results["yolo_only"]["pred"].any()
                and gt.any()
                and pred.any()
            ):
                category = "yolo_empty_recovery"
            if category is not None:
                case_dir = OUTPUT_DIR / "failure_cases" / category
                case_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, case_dir / image_path.name)
                result_mask = case_dir / f"{image_path.stem}_{name}.png"
                import cv2
                cv2.imwrite(str(result_mask), pred.astype("uint8") * 255)
                failure_rows.append(
                    {
                        "stem": image_path.stem,
                        "mode": name,
                        "category": category,
                        "yolo_iou": yolo_iou,
                        "sam3_iou": sam_iou,
                        "mask_path": str(result_mask),
                        "sam3": result["info"].get("sam3", {}),
                    }
                )

    summary = {}
    for name, mode_rows in rows.items():
        pd.DataFrame(mode_rows).to_csv(
            OUTPUT_DIR / f"{name}_metrics.csv", index=False
        )
        summary[name] = aggregate_metrics(mode_rows)
        summary[name]["mean_elapsed_ms"] = float(
            sum(row["elapsed_ms"] for row in mode_rows) / len(mode_rows)
        )
        summary[name]["fallback_count"] = int(
            sum(row["sam3_fallback"] for row in mode_rows)
        )
        mode_df = pd.DataFrame(mode_rows)
        for metric in (
            "boundary_f1",
            "false_positive_area_ratio",
            "false_negative_area_ratio",
            "small_object_recall",
            "peak_gpu_memory_mb",
        ):
            summary[name][f"mean_{metric}"] = float(
                mode_df[metric].dropna().mean()
            )
        if name != "yolo_only":
            yolo_by_stem = {
                row["stem"]: row for row in rows["yolo_only"]
            }
            eligible = [
                row
                for row in mode_rows
                if yolo_by_stem[row["stem"]]["pred_area"] == 0
                and row["gt_area"] > 0
            ]
            summary[name]["yolo_empty_recovery_rate"] = (
                sum(row["pred_area"] > 0 for row in eligible) / len(eligible)
                if eligible
                else float("nan")
            )

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    failure_path = OUTPUT_DIR / "failure_cases.jsonl"
    failure_path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, default=str)
            for row in failure_rows
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")
    print(f"Failure cases: {failure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

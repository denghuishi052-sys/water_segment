"""Leakage-safe manifest helpers for SAM 3 LoRA and selector training."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROMPTS = (
    "standing water",
    "flood water",
    "flooded road",
    "water-covered road",
    "puddle on road",
    "urban waterlogging",
    "inundated pavement",
)


def _grid_points(mask: np.ndarray, count: int) -> tuple[tuple[int, int], ...]:
    valid = (np.asarray(mask) > 0).astype(np.uint8)
    if not valid.any() or count <= 0:
        return ()
    priority = cv2.distanceTransform(valid, cv2.DIST_L2, 5)
    grid = max(1, int(np.ceil(np.sqrt(count))))
    height, width = valid.shape
    choices = []
    for row in range(grid):
        y1, y2 = row * height // grid, (row + 1) * height // grid
        for column in range(grid):
            x1, x2 = column * width // grid, (column + 1) * width // grid
            tile = priority[y1:y2, x1:x2]
            if tile.size == 0 or float(tile.max()) <= 0:
                continue
            offset = int(tile.argmax())
            local_y, local_x = np.unravel_index(offset, tile.shape)
            choices.append(
                (float(tile[local_y, local_x]), x1 + local_x, y1 + local_y)
            )
    choices.sort(reverse=True)
    return tuple((int(x), int(y)) for _, x, y in choices[:count])
SAMPLING_RATIOS = {
    "positive": 0.40,
    "boundary": 0.20,
    "hard_negative": 0.25,
    "small_recovery": 0.15,
}


def classify_row(row: dict) -> tuple[str, str, bool]:
    area = float(row["crop_foreground_area"])
    ratio = float(row["crop_foreground_ratio"])
    mode = str(row["crop_mode"])
    if area <= 0:
        return "hard_negative", "hard_negative", False
    if 0 < ratio <= 0.01:
        return "positive", "small_recovery", True
    if mode == "boundary":
        return "boundary", "boundary", False
    if mode in {"context", "global"}:
        return "context", "positive", False
    return "positive", "positive", False


def validate_source_splits(rows: list[dict]) -> dict[str, set[str]]:
    by_split: dict[str, set[str]] = {}
    for row in rows:
        by_split.setdefault(row["split"], set()).add(row["source_stem"])
    names = sorted(by_split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = by_split[left] & by_split[right]
            if overlap:
                raise ValueError(
                    f"source_stem leakage between {left} and {right}: "
                    f"{sorted(overlap)[:5]}"
                )
    return by_split


def prompt_geometry(mask: np.ndarray) -> tuple[list[int] | None, list, list]:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    height, width = binary.shape
    coordinates = np.argwhere(binary)
    if len(coordinates):
        y1, x1 = coordinates.min(axis=0)
        y2, x2 = coordinates.max(axis=0) + 1
        margin_x = max(1, int(round((x2 - x1) * 0.15)))
        margin_y = max(1, int(round((y2 - y1) * 0.15)))
        box = [
            max(0, int(x1) - margin_x),
            max(0, int(y1) - margin_y),
            min(width, int(x2) + margin_x),
            min(height, int(y2) + margin_y),
        ]
    else:
        box = [0, 0, width, height]
    positive = _grid_points(binary, 5)
    box_mask = np.zeros_like(binary)
    box_mask[box[1] : box[3], box[0] : box[2]] = 1
    ring = cv2.dilate(binary, np.ones((7, 7), np.uint8))
    ring = np.logical_and(ring > 0, binary == 0)
    ring = np.logical_and(ring, box_mask > 0).astype(np.uint8)
    if not ring.any():
        ring = np.logical_and(box_mask > 0, binary == 0).astype(np.uint8)
    negative = _grid_points(ring, 8)
    return (
        box if len(coordinates) else None,
        [list(point) for point in positive],
        [list(point) for point in negative],
    )


def build_manifest_records(
    report_csv: str | Path,
    processed_dir: str | Path,
) -> tuple[list[dict], dict]:
    report_path = Path(report_csv)
    processed = Path(processed_dir)
    with report_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    source_splits = validate_source_splits(rows)
    # Test metadata participates only in the leakage assertion. Test images
    # and masks are deliberately never opened or emitted.
    selected = [row for row in rows if row["split"] in {"train", "val"}]
    bucket_counts = Counter()
    classified = []
    for row in selected:
        crop_type, bucket, small = classify_row(row)
        classified.append((row, crop_type, bucket, small))
        if row["split"] == "train":
            bucket_counts[bucket] += 1

    records = []
    for row, crop_type, bucket, small in classified:
        split = row["split"]
        stem = row["output_stem"]
        image_path = processed / "images" / split / f"{stem}.jpg"
        mask_path = processed / "masks" / split / f"{stem}.png"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing image/mask pair for {stem}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        box, positive, negative = prompt_geometry(mask)
        weight = 1.0
        if split == "train":
            weight = SAMPLING_RATIOS[bucket] / max(bucket_counts[bucket], 1)
        records.append(
            {
                "image": str(image_path.resolve()),
                "mask": str(mask_path.resolve()),
                "source_stem": row["source_stem"],
                "output_stem": stem,
                "split": split,
                "crop_type": crop_type,
                "sampling_bucket": bucket,
                "sample_weight": weight,
                "small_object": small,
                "prompt": PROMPTS[
                    sum(stem.encode("utf-8")) % len(PROMPTS)
                ],
                "gt_box_xyxy": box,
                "positive_points": positive,
                "negative_points": negative,
                "yolo_mask_cache": None,
                "proposal_cache": [],
            }
        )
    summary = {
        "records": len(records),
        "split_counts": dict(Counter(item["split"] for item in records)),
        "train_sampling_buckets": dict(bucket_counts),
        "source_counts": {
            split: len(stems) for split, stems in source_splits.items()
        },
        "test_images_read": 0,
    }
    return records, summary


def write_jsonl(records: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_polygons(label_path: Path, width: int, height: int) -> tuple[list[np.ndarray], int]:
    polygons: list[np.ndarray] = []
    filtered = 0
    if not label_path.is_file():
        return polygons, filtered
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        class_id = int(float(parts[0]))
        if class_id not in {1, 2}:
            continue
        values = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
        box_w = float(values[:, 0].max() - values[:, 0].min())
        box_h = float(values[:, 1].max() - values[:, 1].min())
        if len(values) == 4 and box_w > 0.97 and box_h > 0.97:
            filtered += 1
            continue
        values[:, 0] *= width
        values[:, 1] *= height
        polygons.append(np.round(values).astype(np.int32))
    return polygons, filtered


def sample_center(mask: np.ndarray, mode: str, rng: random.Random) -> tuple[int, int]:
    if mode == "positive":
        candidates = np.argwhere(mask > 0)
    elif mode == "boundary":
        kernel = np.ones((9, 9), np.uint8)
        boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        candidates = np.argwhere(boundary > 0)
    else:
        candidates = np.argwhere(mask == 0)
    if len(candidates) == 0:
        candidates = np.argwhere(np.ones_like(mask, dtype=bool))
    y, x = candidates[rng.randrange(len(candidates))]
    return int(x), int(y)


def crop_bounds(cx: int, cy: int, width: int, height: int, crop_size: int) -> tuple[int, int, int, int]:
    size = min(crop_size, width, height)
    x0 = min(max(cx - size // 2, 0), width - size)
    y0 = min(max(cy - size // 2, 0), height - size)
    return x0, y0, x0 + size, y0 + size


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def build_split(
    source_root: Path,
    output_root: Path,
    split: str,
    crops_per_image: int,
    crop_size: int,
    local_size: int,
    global_size: int,
    seed: int,
) -> tuple[int, int]:
    rng = random.Random(seed)
    images = sorted(
        path for path in (source_root / "images" / split).iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    rows: list[dict[str, str | int]] = []
    filtered_total = 0

    local_image_dir = output_root / split / "local_images"
    local_mask_dir = output_root / split / "local_masks"
    global_image_dir = output_root / split / "global_images"
    global_mask_dir = output_root / split / "global_masks"
    for directory in (local_image_dir, local_mask_dir, global_image_dir, global_mask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    modes = ["positive", "positive", "boundary", "boundary", "negative"]
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        polygons, filtered = parse_polygons(
            source_root / "labels" / split / f"{image_path.stem}.txt",
            width,
            height,
        )
        filtered_total += filtered
        if filtered > 0 and not polygons:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        if polygons:
            cv2.fillPoly(mask, polygons, 255)

        global_image_path = global_image_dir / f"{image_path.stem}.jpg"
        global_mask_path = global_mask_dir / f"{image_path.stem}.png"
        global_image = cv2.resize(image, (global_size, global_size), interpolation=cv2.INTER_AREA)
        global_mask = cv2.resize(mask, (global_size, global_size), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(global_image_path), global_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(global_mask_path), global_mask)

        for crop_index in range(crops_per_image):
            mode = modes[crop_index % len(modes)]
            cx, cy = sample_center(mask, mode, rng)
            x0, y0, x1, y1 = crop_bounds(cx, cy, width, height, crop_size)
            crop_image = image[y0:y1, x0:x1]
            crop_mask = mask[y0:y1, x0:x1]
            crop_image = cv2.resize(crop_image, (local_size, local_size), interpolation=cv2.INTER_AREA)
            crop_mask = cv2.resize(crop_mask, (local_size, local_size), interpolation=cv2.INTER_NEAREST)
            stem = f"{image_path.stem}_{crop_index:02d}"
            local_image_path = local_image_dir / f"{stem}.jpg"
            local_mask_path = local_mask_dir / f"{stem}.png"
            cv2.imwrite(str(local_image_path), crop_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(local_mask_path), crop_mask)
            rows.append({
                "local_image": relative(local_image_path, output_root),
                "local_mask": relative(local_mask_path, output_root),
                "global_image": relative(global_image_path, output_root),
                "global_mask": relative(global_mask_path, output_root),
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "full_width": width,
                "full_height": height,
            })

    manifest = output_root / f"{split}.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), filtered_total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/floodnet/processed")
    parser.add_argument("--output", default="data/dual_context_water_v2")
    parser.add_argument("--train_crops", type=int, default=10)
    parser.add_argument("--val_crops", type=int, default=5)
    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--local_size", type=int, default=768)
    parser.add_argument("--global_size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    source_root = (ROOT / args.source).resolve()
    output_root = (ROOT / args.output).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_root}")
    output_root.mkdir(parents=True)

    train_count, train_filtered = build_split(
        source_root, output_root, "train", args.train_crops,
        args.crop_size, args.local_size, args.global_size, args.seed,
    )
    val_count, val_filtered = build_split(
        source_root, output_root, "val", args.val_crops,
        args.crop_size, args.local_size, args.global_size, args.seed + 1,
    )
    print(f"train={train_count}, val={val_count}")
    print(f"filtered suspicious full-frame polygons: train={train_filtered}, val={val_filtered}")


if __name__ == "__main__":
    main()

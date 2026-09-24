#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_utils import IMAGE_EXTS, ensure_dir, list_files, read_image, read_mask_multiclass, write_yaml
from src.mask_utils import polygons_to_mask, read_yolo_seg_txt, write_yolo_seg_txt


@dataclass(frozen=True)
class ScalePlanItem:
    name: str
    min_scale: float
    max_scale: float
    mode: str


def build_scale_plan() -> list[ScalePlanItem]:
    return [
        ScalePlanItem("global", 1.00, 1.00, "global"),
        ScalePlanItem("large_01", 0.75, 0.90, "context"),
        ScalePlanItem("medium_01", 0.50, 0.70, "foreground"),
        ScalePlanItem("medium_02", 0.50, 0.70, "boundary"),
        ScalePlanItem("local_01", 0.30, 0.50, "foreground"),
        ScalePlanItem("local_02", 0.30, 0.50, "boundary"),
    ]


def binarize_multiclass_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def crop_pair(
    image: np.ndarray,
    mask: np.ndarray,
    window: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Image/mask shape mismatch: {image.shape[:2]} vs {mask.shape[:2]}")
    x1, y1, x2, y2 = window
    h, w = image.shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"Invalid crop window {window} for image shape {(h, w)}")
    return image[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy()


def letterbox_pair(
    image: np.ndarray,
    mask: np.ndarray,
    output_size: int,
    image_pad_value: int = 114,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Image/mask shape mismatch: {image.shape[:2]} vs {mask.shape[:2]}")
    if output_size <= 0:
        raise ValueError("output_size must be positive")

    h, w = image.shape[:2]
    scale = min(output_size / w, output_size / h)
    resized_w = min(output_size, max(1, int(round(w * scale))))
    resized_h = min(output_size, max(1, int(round(h * scale))))
    resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(
        (mask > 0).astype(np.uint8),
        (resized_w, resized_h),
        interpolation=cv2.INTER_NEAREST,
    )

    pad_left = (output_size - resized_w) // 2
    pad_top = (output_size - resized_h) // 2
    out_image = np.full(
        (output_size, output_size, 3),
        image_pad_value,
        dtype=resized_image.dtype,
    )
    out_mask = np.zeros((output_size, output_size), dtype=np.uint8)
    out_image[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized_image
    out_mask[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized_mask

    return out_image, out_mask, {
        "scale": float(scale),
        "resized_width": resized_w,
        "resized_height": resized_h,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": output_size - resized_w - pad_left,
        "pad_bottom": output_size - resized_h - pad_top,
    }


def make_yolo_compatible_mask(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove tiny components and fill holes unsupported by YOLO polygons."""
    binary = (binary_mask > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    filtered = np.zeros_like(binary)
    for component_id in range(1, component_count):
        if int(stats[component_id, cv2.CC_STAT_AREA]) >= min_area:
            filtered[labels == component_id] = 1

    contours, _ = cv2.findContours(
        filtered,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    compatible = np.zeros_like(binary)
    if contours:
        cv2.drawContours(compatible, contours, -1, color=1, thickness=cv2.FILLED)
    return compatible


def accurate_mask_to_polygons(
    binary_mask: np.ndarray,
    epsilon_ratio: float,
) -> list[list[tuple[float, float]]]:
    binary = (binary_mask > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    polygons: list[list[tuple[float, float]]] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = max(0.05, epsilon_ratio * perimeter)
        points = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
        if len(points) < 3:
            points = cv2.convexHull(contour).reshape(-1, 2)
        if len(points) < 3:
            continue
        polygons.append(
            [
                (
                    min(max(float(x) / w, 0.0), 1.0),
                    min(max(float(y) / h, 0.0), 1.0),
                )
                for x, y in points
            ]
        )
    return polygons


def _sample_point(coords: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    index = int(rng.integers(0, len(coords)))
    y, x = coords[index]
    return int(x), int(y)


def _choose_crop_center(
    mask: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> tuple[int, int]:
    h, w = mask.shape[:2]
    foreground = np.argwhere(mask > 0)
    if len(foreground) == 0:
        return int(rng.integers(0, w)), int(rng.integers(0, h))

    if mode == "boundary":
        kernel = np.ones((3, 3), dtype=np.uint8)
        boundary_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        boundary = np.argwhere(boundary_mask > 0)
        if len(boundary):
            return _sample_point(boundary, rng)

    if mode == "context":
        ys, xs = np.where(mask > 0)
        center_x = (int(xs.min()) + int(xs.max())) // 2
        center_y = (int(ys.min()) + int(ys.max())) // 2
        jitter_x = int(rng.integers(-max(1, w // 10), max(2, w // 10 + 1)))
        jitter_y = int(rng.integers(-max(1, h // 10), max(2, h // 10 + 1)))
        return (
            int(np.clip(center_x + jitter_x, 0, w - 1)),
            int(np.clip(center_y + jitter_y, 0, h - 1)),
        )

    return _sample_point(foreground, rng)


def _window_around_center(
    center_x: int,
    center_y: int,
    crop_w: int,
    crop_h: int,
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int]:
    x1 = int(np.clip(center_x - crop_w // 2, 0, image_w - crop_w))
    y1 = int(np.clip(center_y - crop_h // 2, 0, image_h - crop_h))
    return x1, y1, x1 + crop_w, y1 + crop_h


def sample_crop_window(
    mask: np.ndarray,
    min_scale: float,
    max_scale: float,
    mode: str,
    rng: np.random.Generator,
    max_attempts: int = 50,
) -> tuple[int, int, int, int]:
    h, w = mask.shape[:2]
    if mode == "global":
        return 0, 0, w, h
    if not (0 < min_scale <= max_scale <= 1):
        raise ValueError(f"Invalid scale range: {(min_scale, max_scale)}")

    has_foreground = bool(np.any(mask > 0))
    for _ in range(max_attempts):
        scale_x = float(rng.uniform(min_scale, max_scale))
        scale_y = float(rng.uniform(min_scale, max_scale))
        crop_w = min(w, max(2, int(round(w * scale_x))))
        crop_h = min(h, max(2, int(round(h * scale_y))))
        center_x, center_y = _choose_crop_center(mask, mode, rng)
        window = _window_around_center(center_x, center_y, crop_w, crop_h, w, h)
        x1, y1, x2, y2 = window
        if not has_foreground or np.any(mask[y1:y2, x1:x2] > 0):
            return window

    ys, xs = np.where(mask > 0)
    center_x = int(round(float(xs.mean()))) if len(xs) else w // 2
    center_y = int(round(float(ys.mean()))) if len(ys) else h // 2
    crop_w = min(w, max(2, int(round(w * max_scale))))
    crop_h = min(h, max(2, int(round(h * max_scale))))
    return _window_around_center(center_x, center_y, crop_w, crop_h, w, h)


def write_sample(
    image: np.ndarray,
    binary_mask: np.ndarray,
    out_base: str | Path,
    split: str,
    stem: str,
    output_size: int,
    min_area: int,
    epsilon_ratio: float,
    jpeg_quality: int,
) -> tuple[int, dict[str, int | float]]:
    out_base = Path(out_base)
    image_path = ensure_dir(out_base / "images" / split) / f"{stem}.jpg"
    mask_path = ensure_dir(out_base / "masks" / split) / f"{stem}.png"
    label_path = ensure_dir(out_base / "labels" / split) / f"{stem}.txt"

    out_image, out_mask, letterbox_meta = letterbox_pair(image, binary_mask, output_size)
    image_ok = cv2.imwrite(
        str(image_path),
        out_image,
        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
    )
    if not image_ok:
        raise OSError(f"Failed to write sample: {stem}")

    compatible_mask = make_yolo_compatible_mask(out_mask, min_area=min_area)
    polygons = accurate_mask_to_polygons(
        compatible_mask,
        epsilon_ratio=epsilon_ratio,
    )
    write_yolo_seg_txt(label_path, polygons, class_id=0)
    saved_polygons = read_yolo_seg_txt(
        label_path,
        width=output_size,
        height=output_size,
        class_id=0,
    )
    saved_mask = polygons_to_mask(saved_polygons, (output_size, output_size))
    mask_ok = cv2.imwrite(str(mask_path), saved_mask)
    if not mask_ok:
        raise OSError(f"Failed to write mask: {stem}")
    return len(polygons), letterbox_meta


def _assert_safe_output_path(out_base: Path) -> Path:
    resolved = out_base.resolve()
    data_root = (ROOT / "data").resolve()
    if resolved == data_root or data_root not in resolved.parents:
        raise ValueError(f"Output must be a child of {data_root}, got: {resolved}")
    return resolved


def prepare_clean_output(out_base: Path, overwrite: bool) -> None:
    resolved = _assert_safe_output_path(out_base)
    if resolved.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {resolved}. Use --overwrite to replace it.")
        shutil.rmtree(resolved)
    for split in ("train", "val", "test"):
        ensure_dir(resolved / "images" / split)
        ensure_dir(resolved / "masks" / split)
        ensure_dir(resolved / "labels" / split)


def _read_binary_mask(mask_path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")
    return binarize_multiclass_mask(
        read_mask_multiclass(mask_path, image_shape=image_shape)
    )


def process_split(
    in_base: Path,
    out_base: Path,
    split: str,
    rng: np.random.Generator,
    output_size: int,
    min_area: int,
    epsilon_ratio: float,
    jpeg_quality: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    images = list_files(in_base / "images" / split, IMAGE_EXTS)
    plan = build_scale_plan() if split == "train" else build_scale_plan()[:1]

    for image_path in tqdm(images, desc=f"multiscale crop {split}"):
        image = read_image(image_path)
        mask = _read_binary_mask(
            in_base / "masks" / split / f"{image_path.stem}.png",
            image.shape[:2],
        )
        source_h, source_w = image.shape[:2]
        source_area = int((mask > 0).sum())

        for item in plan:
            window = sample_crop_window(
                mask,
                min_scale=item.min_scale,
                max_scale=item.max_scale,
                mode=item.mode,
                rng=rng,
            )
            crop_image, crop_mask = crop_pair(image, mask, window)
            output_stem = f"{image_path.stem}__ms_{item.name}"
            polygon_count, lb = write_sample(
                crop_image,
                crop_mask,
                out_base,
                split,
                output_stem,
                output_size,
                min_area,
                epsilon_ratio,
                jpeg_quality,
            )
            x1, y1, x2, y2 = window
            crop_area = int((crop_mask > 0).sum())
            rows.append(
                {
                    "split": split,
                    "source_stem": image_path.stem,
                    "output_stem": output_stem,
                    "scale_name": item.name,
                    "crop_mode": item.mode,
                    "source_width": source_w,
                    "source_height": source_h,
                    "crop_x1": x1,
                    "crop_y1": y1,
                    "crop_x2": x2,
                    "crop_y2": y2,
                    "crop_width_ratio": (x2 - x1) / source_w,
                    "crop_height_ratio": (y2 - y1) / source_h,
                    "source_foreground_area": source_area,
                    "crop_foreground_area": crop_area,
                    "crop_foreground_ratio": crop_area / max(crop_mask.size, 1),
                    "letterbox_scale": lb["scale"],
                    "pad_left": lb["pad_left"],
                    "pad_top": lb["pad_top"],
                    "num_polygons": polygon_count,
                }
            )
    return rows


def write_dataset_yaml(out_base: Path) -> Path:
    yaml_path = out_base / "waterlogging_binary_multiscale.yaml"
    write_yaml(
        yaml_path,
        {
            "path": str(out_base.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "waterlogging"},
        },
    )
    return yaml_path


def write_report(rows: list[dict[str, object]], out_base: Path) -> Path:
    report_path = ensure_dir(out_base.parent / "split_report") / "multiscale_crop_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed multiscale binary FloodNet YOLO segmentation dataset."
    )
    parser.add_argument("--input_processed", default="data/floodnet/processed")
    parser.add_argument(
        "--output_processed",
        default="data/floodnet_binary_multiscale_1024/processed",
    )
    parser.add_argument("--output_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_area", type=int, default=20)
    parser.add_argument("--epsilon_ratio", type=float, default=0.002)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_base = Path(args.input_processed)
    out_base = Path(args.output_processed)
    rng = np.random.default_rng(args.seed)

    prepare_clean_output(out_base, overwrite=args.overwrite)
    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        rows.extend(
            process_split(
                in_base=in_base,
                out_base=out_base,
                split=split,
                rng=rng,
                output_size=args.output_size,
                min_area=args.min_area,
                epsilon_ratio=args.epsilon_ratio,
                jpeg_quality=args.jpeg_quality,
            )
        )

    yaml_path = write_dataset_yaml(out_base)
    report_path = write_report(rows, out_base)
    print(f"Multiscale dataset written to: {out_base.resolve()}")
    print(f"Dataset YAML: {yaml_path.resolve()}")
    print(f"Crop report: {report_path.resolve()}")
    for split in ("train", "val", "test"):
        print(f"{split}: {sum(row['split'] == split for row in rows)} samples")


if __name__ == "__main__":
    main()

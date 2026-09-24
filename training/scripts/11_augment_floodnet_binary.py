#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_utils import IMAGE_EXTS, ensure_dir, list_files, read_image, read_mask_multiclass, write_yaml
from src.mask_utils import binary_mask_to_polygons, write_yolo_seg_txt


def binarize_multiclass_mask(mask: np.ndarray) -> np.ndarray:
    """Merge all non-background FloodNet classes into one foreground class."""
    return (mask > 0).astype(np.uint8)


def write_binary_sample(
    image: np.ndarray,
    binary_mask: np.ndarray,
    out_base: str | Path,
    split: str,
    stem: str,
    image_ext: str,
    min_area: int,
    epsilon_ratio: float,
    output_size: int | None = None,
    jpeg_quality: int = 90,
) -> int:
    out_base = Path(out_base)
    image_path = ensure_dir(out_base / "images" / split) / f"{stem}{image_ext}"
    mask_path = ensure_dir(out_base / "masks" / split) / f"{stem}.png"
    label_path = ensure_dir(out_base / "labels" / split) / f"{stem}.txt"

    if image.shape[:2] != binary_mask.shape[:2]:
        raise ValueError(f"Image/mask shape mismatch for {stem}: {image.shape[:2]} vs {binary_mask.shape[:2]}")

    clean_mask = (binary_mask > 0).astype(np.uint8)
    out_image = image
    if output_size is not None and output_size > 0 and image.shape[:2] != (output_size, output_size):
        out_image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
        clean_mask = cv2.resize(clean_mask, (output_size, output_size), interpolation=cv2.INTER_NEAREST)

    write_params: list[int] = []
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        write_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    cv2.imwrite(str(image_path), out_image, write_params)
    cv2.imwrite(str(mask_path), clean_mask)
    polygons = binary_mask_to_polygons(clean_mask, min_area=min_area, epsilon_ratio=epsilon_ratio)
    write_yolo_seg_txt(label_path, polygons, class_id=0)
    return len(polygons)


def foreground_area(mask: np.ndarray) -> int:
    return int((mask > 0).sum())


def horizontal_flip(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray, str]:
    return cv2.flip(image, 1), cv2.flip(mask, 1), "hflip"


def rotate_small(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray, str]:
    h, w = image.shape[:2]
    angle = rng.uniform(-15.0, 15.0)
    scale = rng.uniform(0.95, 1.05)
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    aug_img = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    aug_mask = cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aug_img, (aug_mask > 0).astype(np.uint8), "rotate"


def random_resized_crop(
    image: np.ndarray,
    mask: np.ndarray,
    rng: random.Random,
    scale_range: tuple[float, float],
    y_bias: str,
    min_keep_ratio: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    h, w = image.shape[:2]
    original_area = max(foreground_area(mask), 1)

    for _ in range(30):
        scale = rng.uniform(*scale_range)
        crop_w = max(8, min(w, int(w * scale)))
        crop_h = max(8, min(h, int(h * scale)))
        max_x = max(0, w - crop_w)
        max_y = max(0, h - crop_h)
        x0 = rng.randint(0, max_x) if max_x else 0
        if y_bias == "lower":
            y_min = int(max_y * 0.35)
            y0 = rng.randint(y_min, max_y) if max_y >= y_min else max_y
        elif y_bias == "upper":
            y_max = int(max_y * 0.65)
            y0 = rng.randint(0, y_max) if y_max else 0
        else:
            y0 = rng.randint(0, max_y) if max_y else 0

        crop_mask = mask[y0 : y0 + crop_h, x0 : x0 + crop_w]
        if foreground_area(mask) == 0 or foreground_area(crop_mask) >= original_area * min_keep_ratio:
            crop_img = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
            out_img = cv2.resize(crop_img, (w, h), interpolation=cv2.INTER_LINEAR)
            out_mask = cv2.resize(crop_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            return out_img, (out_mask > 0).astype(np.uint8), f"{y_bias}_crop"

    return image.copy(), mask.copy(), f"{y_bias}_crop_skipped"


def perspective_light(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray, str]:
    h, w = image.shape[:2]
    max_dx = w * 0.035
    max_dy = h * 0.035
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [rng.uniform(0, max_dx), rng.uniform(0, max_dy)],
            [w - 1 - rng.uniform(0, max_dx), rng.uniform(0, max_dy)],
            [w - 1 - rng.uniform(0, max_dx), h - 1 - rng.uniform(0, max_dy)],
            [rng.uniform(0, max_dx), h - 1 - rng.uniform(0, max_dy)],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    aug_img = cv2.warpPerspective(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    aug_mask = cv2.warpPerspective(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aug_img, (aug_mask > 0).astype(np.uint8), "perspective"


def photometric_jitter(image: np.ndarray, rng: random.Random) -> np.ndarray:
    img = image.astype(np.float32)
    alpha = rng.uniform(0.82, 1.18)
    beta = rng.uniform(-18.0, 18.0)
    img = img * alpha + beta
    gamma = rng.uniform(0.85, 1.15)
    img = np.clip(img, 0, 255) / 255.0
    img = np.power(img, gamma) * 255.0
    out = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if rng.random() < 0.2:
        noise = rng.uniform(2.0, 6.0)
        n = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, noise, out.shape)
        out = np.clip(out.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return out


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    rng: random.Random,
    min_keep_ratio: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    ops: list[Callable[[np.ndarray, np.ndarray, random.Random], tuple[np.ndarray, np.ndarray, str]]] = [
        horizontal_flip,
        rotate_small,
        perspective_light,
        lambda img, m, r: random_resized_crop(img, m, r, (0.55, 0.92), "lower", min_keep_ratio),
        lambda img, m, r: random_resized_crop(img, m, r, (0.70, 1.00), "upper", min_keep_ratio),
        lambda img, m, r: random_resized_crop(img, m, r, (0.60, 1.00), "random", min_keep_ratio),
    ]
    image_aug, mask_aug, op_name = rng.choice(ops)(image, mask, rng)
    if rng.random() < 0.75:
        image_aug = photometric_jitter(image_aug, rng)
        op_name = f"{op_name}+photo"
    return image_aug, (mask_aug > 0).astype(np.uint8), op_name


def read_binary_mask(mask_path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    mask = read_mask_multiclass(mask_path, image_shape=image_shape)
    return binarize_multiclass_mask(mask)


def prepare_clean_output(out_base: Path, overwrite: bool) -> None:
    if out_base.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {out_base}. Use --overwrite to replace it.")
        last_error: OSError | None = None
        for _ in range(5):
            try:
                shutil.rmtree(out_base)
                last_error = None
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.5)
        if last_error is not None:
            raise last_error
    for split in ["train", "val", "test"]:
        ensure_dir(out_base / "images" / split)
        ensure_dir(out_base / "masks" / split)
        ensure_dir(out_base / "labels" / split)


def process_split(
    in_base: Path,
    out_base: Path,
    split: str,
    rng: random.Random,
    foreground_augments: int,
    empty_augments: int,
    min_keep_ratio: float,
    min_area: int,
    epsilon_ratio: float,
    output_size: int | None,
    jpeg_quality: int,
) -> list[dict[str, object]]:
    image_dir = in_base / "images" / split
    mask_dir = in_base / "masks" / split
    rows: list[dict[str, object]] = []

    images = list_files(image_dir, IMAGE_EXTS)
    for image_path in tqdm(images, desc=f"binary augment {split}"):
        image = read_image(image_path)
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")
        mask = read_binary_mask(mask_path, image.shape[:2])
        image_ext = image_path.suffix.lower()
        base_area = foreground_area(mask)

        polygons = write_binary_sample(
            image,
            mask,
            out_base,
            split,
            image_path.stem,
            image_ext,
            min_area,
            epsilon_ratio,
            output_size=output_size,
            jpeg_quality=jpeg_quality,
        )
        rows.append(
            {
                "split": split,
                "source_stem": image_path.stem,
                "output_stem": image_path.stem,
                "augment": "original_binary",
                "foreground_area": base_area,
                "num_polygons": polygons,
            }
        )

        if split != "train":
            continue

        count = foreground_augments if base_area > 0 else empty_augments
        for i in range(count):
            aug_img, aug_mask, aug_name = augment_pair(image, mask, rng, min_keep_ratio)
            aug_stem = f"{image_path.stem}__aug{i + 1:02d}_{aug_name.replace('+', '_')}"
            aug_area = foreground_area(aug_mask)
            polygons = write_binary_sample(
                aug_img,
                aug_mask,
                out_base,
                split,
                aug_stem,
                image_ext,
                min_area,
                epsilon_ratio,
                output_size=output_size,
                jpeg_quality=jpeg_quality,
            )
            rows.append(
                {
                    "split": split,
                    "source_stem": image_path.stem,
                    "output_stem": aug_stem,
                    "augment": aug_name,
                    "foreground_area": aug_area,
                    "num_polygons": polygons,
                }
            )
    return rows


def write_dataset_yaml(out_base: Path) -> None:
    write_yaml(
        out_base / "waterlogging_binary.yaml",
        {
            "path": str(out_base.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "waterlogging"},
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a binary augmented FloodNet YOLO-seg dataset.")
    p.add_argument("--input_processed", default="data/floodnet/processed")
    p.add_argument("--output_processed", default="data/floodnet_binary_aug/processed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--foreground_augments", type=int, default=3)
    p.add_argument("--empty_augments", type=int, default=1)
    p.add_argument("--min_keep_ratio", type=float, default=0.20)
    p.add_argument("--min_area", type=int, default=20)
    p.add_argument("--epsilon_ratio", type=float, default=0.002)
    p.add_argument("--output_size", type=int, default=1024, help="Square output size. Use 0 to keep original size.")
    p.add_argument("--jpeg_quality", type=int, default=90)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_base = Path(args.input_processed)
    out_base = Path(args.output_processed)
    rng = random.Random(args.seed)

    prepare_clean_output(out_base, overwrite=args.overwrite)
    rows: list[dict[str, object]] = []
    for split in ["train", "val", "test"]:
        rows.extend(
            process_split(
                in_base=in_base,
                out_base=out_base,
                split=split,
                rng=rng,
                foreground_augments=args.foreground_augments,
                empty_augments=args.empty_augments,
                min_keep_ratio=args.min_keep_ratio,
                min_area=args.min_area,
                epsilon_ratio=args.epsilon_ratio,
                output_size=args.output_size if args.output_size > 0 else None,
                jpeg_quality=args.jpeg_quality,
            )
        )

    write_dataset_yaml(out_base)
    report_path = ensure_dir(out_base.parent / "split_report") / "augmentation_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "source_stem", "output_stem", "augment", "foreground_area", "num_polygons"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Binary augmented dataset written to: {out_base}")
    print(f"Dataset yaml: {out_base / 'waterlogging_binary.yaml'}")
    print(f"Augmentation report: {report_path}")


if __name__ == "__main__":
    main()

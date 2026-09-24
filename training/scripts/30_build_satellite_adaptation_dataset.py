from __future__ import annotations

import csv
import io
import zipfile
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "processed" / "flood_cases"
OUTPUT_ROOT = ROOT / "data" / "satellite_adaptation_v1"
BASE_ROOT = ROOT / "data" / "dual_context_water_v2"

SCENES = {
    "ian": {
        "directory": "ian_fort_myers_2022",
        "image": "noaa_20220930d-rgb_z19_rgb.tif",
        "zip": "ian_water_segmentation_results_conservative.zip",
        "mask": "ian_water_mask_conservative.tif",
        "split": "train",
        "sample_weight": 0.70,
    },
    "ida": {
        "directory": "ida_grand_isle_2021",
        "image": "noaa_20210831a-rgb_z20_rgb.tif",
        "zip": "ida_water_segmentation_results_open.zip",
        "mask": "ida_water_mask_open.tif",
        "split": "val",
        "sample_weight": 0.55,
    },
}


def starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    values = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if values[-1] != final:
        values.append(final)
    return values


def write_scene(
    name: str,
    config: dict,
    output_root: Path = OUTPUT_ROOT,
    tile_size: int = 1024,
    overlap: int = 256,
    local_size: int = 768,
    global_size: int = 384,
) -> list[dict[str, str | int | float]]:
    image_path = SOURCE_ROOT / config["directory"] / config["image"]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    with zipfile.ZipFile(SOURCE_ROOT / config["zip"]) as archive:
        Image.MAX_IMAGE_PIXELS = None
        mask = np.asarray(Image.open(io.BytesIO(archive.read(config["mask"]))))
    mask = (mask > 0).astype(np.uint8) * 255
    height, width = image.shape[:2]
    if mask.shape != (height, width):
        raise ValueError(f"{name}: source={image.shape}, mask={mask.shape}")

    split = str(config["split"])
    scene_root = output_root / split / name
    local_images = scene_root / "local_images"
    local_masks = scene_root / "local_masks"
    global_images = scene_root / "global_images"
    global_masks = scene_root / "global_masks"
    for directory in (local_images, local_masks, global_images, global_masks):
        directory.mkdir(parents=True, exist_ok=True)

    global_image_path = global_images / f"{name}.jpg"
    global_mask_path = global_masks / f"{name}.png"
    cv2.imwrite(
        str(global_image_path),
        cv2.resize(image, (global_size, global_size), interpolation=cv2.INTER_AREA),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    cv2.imwrite(
        str(global_mask_path),
        cv2.resize(mask, (global_size, global_size), interpolation=cv2.INTER_NEAREST),
    )

    rows: list[dict[str, str | int | float]] = []
    index = 0
    for y0 in starts(height, tile_size, overlap):
        for x0 in starts(width, tile_size, overlap):
            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            local_image = cv2.resize(
                image[y0:y1, x0:x1],
                (local_size, local_size),
                interpolation=cv2.INTER_AREA,
            )
            local_mask = cv2.resize(
                mask[y0:y1, x0:x1],
                (local_size, local_size),
                interpolation=cv2.INTER_NEAREST,
            )
            stem = f"{name}_{index:04d}"
            local_image_path = local_images / f"{stem}.jpg"
            local_mask_path = local_masks / f"{stem}.png"
            cv2.imwrite(str(local_image_path), local_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(local_mask_path), local_mask)
            rows.append({
                "local_image": str(local_image_path.resolve()),
                "local_mask": str(local_mask_path.resolve()),
                "global_image": str(global_image_path.resolve()),
                "global_mask": str(global_mask_path.resolve()),
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "full_width": width,
                "full_height": height,
                "sample_weight": float(config["sample_weight"]),
                "domain": "satellite",
            })
            index += 1
    return rows


def base_rows(base_root: Path = BASE_ROOT) -> list[dict[str, str | int | float]]:
    with (base_root / "train.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    path_fields = ("local_image", "local_mask", "global_image", "global_mask")
    for row in rows:
        for field in path_fields:
            row[field] = str((base_root / row[field]).resolve())
        row["sample_weight"] = 1.0
        row["domain"] = "base"
    return rows


def write_manifest(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--base-root", type=Path, default=BASE_ROOT)
    parser.add_argument("--local-size", type=int, default=768)
    parser.add_argument("--global-size", type=int, default=384)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    args = parser.parse_args()

    output_root = args.output.resolve()
    base_root = args.base_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_root}")
    output_root.mkdir(parents=True)
    scene_args = {
        "output_root": output_root,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "local_size": args.local_size,
        "global_size": args.global_size,
    }
    train_satellite = write_scene("ian", SCENES["ian"], **scene_args)
    val_satellite = write_scene("ida", SCENES["ida"], **scene_args)
    base_train = base_rows(base_root)
    write_manifest(output_root / "satellite_train.csv", train_satellite)
    write_manifest(output_root / "satellite_val.csv", val_satellite)
    write_manifest(output_root / "combined_train.csv", base_train + train_satellite)
    print(f"satellite_train={len(train_satellite)}")
    print(f"satellite_val={len(val_satellite)}")
    print(f"combined_train={len(base_train) + len(train_satellite)}")


if __name__ == "__main__":
    main()

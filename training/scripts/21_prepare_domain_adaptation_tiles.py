#!/usr/bin/env python
"""Create reviewable Google Earth tiles for flood-vs-natural-water annotation."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import tifffile
import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a large GeoTIFF into YOLO-seg annotation tiles."
    )
    parser.add_argument("--image", required=True, help="Source GeoTIFF or RGB image.")
    parser.add_argument("--output_dir", default="data/ge_adaptation_round1")
    parser.add_argument("--tile_size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            image = tifffile.imread(path)
        except ValueError as exc:
            # JPEG-compressed GeoTIFFs need imagecodecs in tifffile. Pillow
            # can decode the same common variant without adding a runtime
            # dependency to the training environment.
            if "requires the 'imagecodecs' package" not in str(exc):
                raise
            with Image.open(path) as tif:
                image = np.asarray(tif.convert("RGB"))
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    if image is None:
        raise ValueError(f"Could not read {path}")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected at least three channels; got shape {image.shape}")
    image = image[..., :3]
    if image.dtype != np.uint8:
        finite = image[np.isfinite(image)]
        max_value = float(finite.max()) if finite.size else 1.0
        image = np.clip(image.astype(np.float32) * (255.0 / max(max_value, 1.0)), 0, 255).astype(np.uint8)
    # tifffile returns RGB, while OpenCV writes/reads BGR.
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    values = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if values[-1] != final:
        values.append(final)
    return values


def main() -> None:
    args = parse_args()
    source = Path(args.image).resolve()
    output = Path(args.output_dir).resolve()
    if args.tile_size <= 0 or not 0 <= args.overlap < args.tile_size:
        raise ValueError("Require tile_size > 0 and 0 <= overlap < tile_size")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Pass --overwrite to rebuild it.")
        shutil.rmtree(output)
    image_dir = output / "images"
    label_dir = output / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image = read_image(source)
    h, w = image.shape[:2]
    rows: list[dict[str, object]] = []
    for row, y0 in enumerate(starts(h, args.tile_size, args.overlap)):
        for col, x0 in enumerate(starts(w, args.tile_size, args.overlap)):
            crop = image[y0 : y0 + args.tile_size, x0 : x0 + args.tile_size]
            # Keep all annotation canvases fixed-size; padding is background only.
            canvas = np.full((args.tile_size, args.tile_size, 3), 114, dtype=np.uint8)
            canvas[: crop.shape[0], : crop.shape[1]] = crop
            stem = f"{source.stem}__r{row:02d}_c{col:02d}"
            image_path = image_dir / f"{stem}.jpg"
            if not cv2.imwrite(str(image_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                raise OSError(f"Could not write {image_path}")
            # Empty files are deliberate: every tile must be reviewed and labelled.
            (label_dir / f"{stem}.txt").write_text("", encoding="utf-8")
            rows.append({"stem": stem, "image": str(image_path), "x0": x0, "y0": y0, "width": crop.shape[1], "height": crop.shape[0], "review_status": "pending"})

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "label_schema.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            {
                "names": {0: "flood", 1: "natural_water"},
                "instruction": "Annotate flood as class 0 and permanent river, lake, pool, or channel water as class 1. Leave dry land and image-border artefacts unlabelled.",
                "source_image": str(source),
                "tile_size": args.tile_size,
                "overlap": args.overlap,
            },
            file,
            allow_unicode=True,
            sort_keys=False,
        )
    print(f"Prepared {len(rows)} tiles: {output}")
    print(f"Manifest: {output / 'manifest.csv'}")
    print(f"Label schema: {output / 'label_schema.yaml'}")


if __name__ == "__main__":
    main()

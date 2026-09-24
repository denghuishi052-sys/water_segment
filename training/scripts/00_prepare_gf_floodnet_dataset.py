#!/usr/bin/env python
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm


def parse_channels(value: str | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, str):
        parts = [int(x.strip()) for x in value.split(",") if x.strip() != ""]
    else:
        parts = [int(x) for x in value]
    if len(parts) != 3:
        raise ValueError(f"rgb_channels must contain exactly 3 indexes, got: {value}")
    if min(parts) < 0:
        raise ValueError(f"rgb_channels must be zero-based non-negative indexes, got: {value}")
    return tuple(parts)  # type: ignore[return-value]


def _stretch_channel_percentile(channel: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(channel, [lower, upper])
    if hi <= lo:
        return np.zeros(channel.shape, dtype=np.uint8)
    out = (channel.astype(np.float32) - float(lo)) * 255.0 / float(hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def _to_uint8_no_stretch(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    lo = float(np.nanmin(image))
    hi = float(np.nanmax(image))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    out = (image.astype(np.float32) - lo) * 255.0 / float(hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def to_rgb_uint8(
    image: np.ndarray,
    rgb_channels: str | tuple[int, int, int] = (0, 1, 2),
    stretch: str = "percentile",
) -> np.ndarray:
    channels = parse_channels(rgb_channels)
    if image.ndim == 2:
        selected = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3:
        if max(channels) >= image.shape[2]:
            raise ValueError(f"Image has {image.shape[2]} channels, cannot select rgb_channels={channels}")
        selected = image[:, :, list(channels)]
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    stretch = stretch.lower().strip()
    if stretch == "none":
        return _to_uint8_no_stretch(selected)
    if stretch == "percentile":
        out = np.zeros(selected.shape, dtype=np.uint8)
        for channel_idx in range(3):
            out[:, :, channel_idx] = _stretch_channel_percentile(selected[:, :, channel_idx])
        return out
    raise ValueError(f"Unsupported stretch={stretch}. Use none or percentile.")


def gf_label_to_mask(label: np.ndarray, threshold: int = 128) -> np.ndarray:
    if label.ndim == 3:
        label = label[:, :, 0]
    return (label < int(threshold)).astype(np.uint8) * 255


def _read_tif_from_zip(zip_file: ZipFile, name: str) -> np.ndarray:
    with zip_file.open(name) as f:
        return tifffile.imread(BytesIO(f.read()))


def _image_names(zip_file: ZipFile) -> list[str]:
    return sorted(
        n for n in zip_file.namelist()
        if n.startswith("images/") and n.lower().endswith((".tif", ".tiff"))
    )


def prepare_dataset(
    zip_path: Path,
    output_dir: Path,
    rgb_channels: tuple[int, int, int] = (0, 1, 2),
    stretch: str = "percentile",
    threshold: int = 128,
    limit: int | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    report_dir = output_dir / "reports"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with ZipFile(zip_path) as z:
        names = _image_names(z)
        if limit is not None:
            names = names[:limit]
        for image_name in tqdm(names, desc="converting GF-FloodNet"):
            stem = Path(image_name).stem
            annotation_name = f"annotations/{stem}.tif"
            if annotation_name not in z.namelist():
                raise FileNotFoundError(f"Missing annotation for {image_name}: {annotation_name}")

            safe_stem = "gf_" + stem.replace(" ", "_")
            image_out = image_dir / f"{safe_stem}.jpg"
            mask_out = mask_dir / f"{safe_stem}.png"

            source_shape = ""
            source_dtype = ""
            source_channels = ""
            if not overwrite and image_out.exists() and mask_out.exists():
                rgb_bgr = cv2.imread(str(image_out), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(mask_out), cv2.IMREAD_GRAYSCALE)
                if rgb_bgr is None or mask is None:
                    raise ValueError(f"Failed to read existing outputs for {safe_stem}")
                rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            else:
                source = _read_tif_from_zip(z, image_name)
                label = _read_tif_from_zip(z, annotation_name)
                rgb = to_rgb_uint8(source, rgb_channels=rgb_channels, stretch=stretch)
                mask = gf_label_to_mask(label, threshold=threshold)
                if mask.shape[:2] != rgb.shape[:2]:
                    mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(image_out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                cv2.imwrite(str(mask_out), mask)
                source_shape = "x".join(str(x) for x in source.shape)
                source_dtype = str(source.dtype)
                source_channels = int(source.shape[2]) if source.ndim == 3 else 1

            h, w = rgb.shape[:2]
            rows.append(
                {
                    "stem": safe_stem,
                    "source_image_path": image_name,
                    "source_mask_path": annotation_name,
                    "source_shape": source_shape,
                    "source_dtype": source_dtype,
                    "source_channels": source_channels,
                    "rgb_channels": ",".join(str(x) for x in rgb_channels),
                    "stretch": stretch,
                    "mask_rule": f"label < {threshold}",
                    "image_width": w,
                    "image_height": h,
                    "mask_area": int((mask > 0).sum()),
                    "mask_area_ratio": float((mask > 0).sum() / max(h * w, 1)),
                    "output_image_path": str(image_out),
                    "output_mask_path": str(mask_out),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(report_dir / "meta.csv", index=False)
    return df


def parse_args():
    p = argparse.ArgumentParser(description="Prepare GF-FloodNet TIF files as RGB images and binary water masks.")
    p.add_argument("--zip_path", default=r"D:\BaiduNetdiskDownload\GF-FloodNet\GF-FloodNet-v1.zip")
    p.add_argument("--output_dir", default="data/gf_floodnet/raw")
    p.add_argument("--rgb_channels", default="0,1,2")
    p.add_argument("--stretch", default="percentile", choices=["none", "percentile"])
    p.add_argument("--threshold", type=int, default=128)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    df = prepare_dataset(
        zip_path=Path(args.zip_path),
        output_dir=Path(args.output_dir),
        rgb_channels=parse_channels(args.rgb_channels),
        stretch=args.stretch,
        threshold=args.threshold,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"Converted {len(df)} samples.")
    print(f"RGB images: {Path(args.output_dir) / 'images'}")
    print(f"Binary masks: {Path(args.output_dir) / 'masks'}")
    print(f"Report: {Path(args.output_dir) / 'reports' / 'meta.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_utils import ensure_dir


def extract_numeric_id(path: str | Path) -> str:
    matches = re.findall(r"\d+", Path(path).stem)
    if not matches:
        raise ValueError(f"Could not extract numeric id from file name: {path}")
    return matches[-1]


def parse_channels(value: str | Sequence[int]) -> tuple[int, int, int]:
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


def percentile_norm(x: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    x = x.astype(np.float32)
    low = np.percentile(x, p_low)
    high = np.percentile(x, p_high)
    x = np.clip(x, low, high)
    return (x - low) / (high - low + 1e-6)


def sar_to_db(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = np.maximum(x, 1e-6)
    return 10 * np.log10(x)


def build_flood_pseudo_rgb(
    b2: np.ndarray,
    b3: np.ndarray,
    b4: np.ndarray,
    b8: np.ndarray,
    hh: np.ndarray,
    hv: np.ndarray,
) -> np.ndarray:
    del b2, b4
    ndwi = (b3.astype(np.float32) - b8.astype(np.float32)) / (
        b3.astype(np.float32) + b8.astype(np.float32) + 1e-6
    )
    ndwi_norm = percentile_norm(ndwi)

    hh_norm = percentile_norm(sar_to_db(hh))
    hv_norm = percentile_norm(sar_to_db(hv))

    pseudo_rgb = np.stack([ndwi_norm, 1.0 - hh_norm, 1.0 - hv_norm], axis=-1)
    return (pseudo_rgb * 255).clip(0, 255).astype(np.uint8)


def sentinel_gf3_to_flood_pseudo_rgb(sentinel: np.ndarray, hh: np.ndarray, hv: np.ndarray) -> np.ndarray:
    if sentinel.ndim != 3 or sentinel.shape[2] < 4:
        raise ValueError(f"Flood pseudo RGB requires Sentinel-2 shape HxWx4+, got: {sentinel.shape}")
    if hh.shape[:2] != sentinel.shape[:2] or hv.shape[:2] != sentinel.shape[:2]:
        raise ValueError(f"GF3 HH/HV shapes must match Sentinel-2 shape: sentinel={sentinel.shape}, hh={hh.shape}, hv={hv.shape}")
    return build_flood_pseudo_rgb(
        sentinel[:, :, 0],
        sentinel[:, :, 1],
        sentinel[:, :, 2],
        sentinel[:, :, 3],
        hh,
        hv,
    )


def _to_uint8_no_stretch(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        if info.max <= 255 and info.min >= 0:
            return image.astype(np.uint8)
    lo = float(np.nanmin(image))
    hi = float(np.nanmax(image))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    out = (image.astype(np.float32) - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def to_rgb_uint8(
    image: np.ndarray,
    rgb_channels: str | Sequence[int] = (0, 1, 2),
    stretch: str = "none",
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


def label_to_mask(label: np.ndarray) -> np.ndarray:
    if label.ndim == 3:
        label = label[:, :, 0]
    return (label > 0).astype(np.uint8) * 255


def _index_files(directory: Path, prefix: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.glob(f"{prefix}*.[Tt][Ii][Ff]*")):
        sample_id = extract_numeric_id(path)
        if sample_id in files:
            raise ValueError(f"Duplicate sample id {sample_id}: {files[sample_id]} and {path}")
        files[sample_id] = path
    return files


def find_dataset_root(extract_dir: Path) -> Path:
    candidate = extract_dir / "Sen2GF3Floods"
    if candidate.exists():
        return candidate
    if (extract_dir / "sentinel2").exists() and (extract_dir / "label").exists():
        return extract_dir
    raise FileNotFoundError(f"Could not find Sen2GF3Floods dataset under: {extract_dir}")


def extract_archive_if_needed(archive: Path | None, extract_dir: Path, force_extract: bool = False) -> Path:
    if archive is None:
        return find_dataset_root(extract_dir)
    dataset_root = extract_dir / "Sen2GF3Floods"
    if dataset_root.exists() and not force_extract:
        return dataset_root
    ensure_dir(extract_dir)
    subprocess.run(["tar", "-xf", str(archive), "-C", str(extract_dir)], check=True)
    return find_dataset_root(extract_dir)


def iter_pairs(dataset_root: Path) -> list[dict[str, str | Path]]:
    sentinel_dir = dataset_root / "sentinel2"
    label_dir = dataset_root / "label"
    gf3_dir = dataset_root / "gaofen3"
    if not sentinel_dir.exists():
        raise FileNotFoundError(f"Missing sentinel2 directory: {sentinel_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    sentinel = _index_files(sentinel_dir, "before_s2_")
    labels = _index_files(label_dir, "label_")
    hh = _index_files(gf3_dir, "after_gf3_hh_") if gf3_dir.exists() else {}
    hv = _index_files(gf3_dir, "after_gf3_hv_") if gf3_dir.exists() else {}

    common_ids = sorted(set(sentinel) & set(labels), key=lambda x: int(x))
    missing_labels = sorted(set(sentinel) - set(labels), key=lambda x: int(x))
    if missing_labels:
        raise FileNotFoundError(f"Missing labels for {len(missing_labels)} sentinel images. Examples: {missing_labels[:10]}")

    rows: list[dict[str, str | Path]] = []
    for sample_id in common_ids:
        rows.append(
            {
                "id": sample_id,
                "sentinel_path": sentinel[sample_id],
                "label_path": labels[sample_id],
                "gf3_hh_path": hh.get(sample_id, ""),
                "gf3_hv_path": hv.get(sample_id, ""),
            }
        )
    return rows


def prepare_dataset(
    dataset_root: Path,
    output_dir: Path,
    rgb_mode: str = "sentinel",
    rgb_channels: tuple[int, int, int] = (0, 1, 2),
    stretch: str = "none",
    image_ext: str = ".jpg",
    limit: int | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    image_dir = ensure_dir(output_dir / "images")
    mask_dir = ensure_dir(output_dir / "masks")
    report_dir = ensure_dir(output_dir / "reports")

    rows = iter_pairs(dataset_root)
    if limit is not None:
        rows = rows[:limit]

    report_rows = []
    image_ext = image_ext if image_ext.startswith(".") else f".{image_ext}"
    for row in tqdm(rows, desc="converting Sen2GF3Floods"):
        sample_id = str(row["id"])
        stem = f"sen2gf3_{int(sample_id):05d}"
        image_out = image_dir / f"{stem}{image_ext.lower()}"
        mask_out = mask_dir / f"{stem}.png"

        source_shape = ""
        source_dtype = ""
        source_channels = ""
        if not overwrite and image_out.exists() and mask_out.exists():
            rgb_bgr = cv2.imread(str(image_out), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_out), cv2.IMREAD_GRAYSCALE)
            if rgb_bgr is None:
                raise ValueError(f"Failed to read existing image output: {image_out}")
            if mask is None:
                raise ValueError(f"Failed to read existing mask output: {mask_out}")
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        else:
            sentinel = tifffile.imread(row["sentinel_path"])
            label = tifffile.imread(row["label_path"])
            rgb_mode = rgb_mode.lower().strip()
            if rgb_mode == "sentinel":
                rgb = to_rgb_uint8(sentinel, rgb_channels=rgb_channels, stretch=stretch)
            elif rgb_mode == "flood_pseudo":
                if not row["gf3_hh_path"] or not row["gf3_hv_path"]:
                    raise FileNotFoundError(f"Flood pseudo RGB requires GF3 HH/HV for sample id {sample_id}")
                hh = tifffile.imread(row["gf3_hh_path"])
                hv = tifffile.imread(row["gf3_hv_path"])
                rgb = sentinel_gf3_to_flood_pseudo_rgb(sentinel, hh, hv)
            else:
                raise ValueError(f"Unsupported rgb_mode={rgb_mode}. Use sentinel or flood_pseudo.")
            mask = label_to_mask(label)
            if mask.shape[:2] != rgb.shape[:2]:
                mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            source_shape = "x".join(str(x) for x in sentinel.shape)
            source_dtype = str(sentinel.dtype)
            source_channels = int(sentinel.shape[2]) if sentinel.ndim == 3 else 1

            if image_ext.lower() in {".jpg", ".jpeg"}:
                cv2.imwrite(str(image_out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                cv2.imwrite(str(image_out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mask_out), mask)

        h, w = rgb.shape[:2]
        report_rows.append(
            {
                "id": sample_id,
                "stem": stem,
                "source_image_path": str(row["sentinel_path"]),
                "source_mask_path": str(row["label_path"]),
                "gf3_hh_path": str(row["gf3_hh_path"]),
                "gf3_hv_path": str(row["gf3_hv_path"]),
                "source_shape": source_shape,
                "source_dtype": source_dtype,
                "source_channels": source_channels,
                "rgb_mode": rgb_mode,
                "rgb_channels": ",".join(str(x) for x in rgb_channels),
                "stretch": stretch,
                "image_width": w,
                "image_height": h,
                "mask_area": int((mask > 0).sum()),
                "mask_area_ratio": float((mask > 0).sum() / max(h * w, 1)),
                "output_image_path": str(image_out),
                "output_mask_path": str(mask_out),
            }
        )

    df = pd.DataFrame(report_rows)
    df.to_csv(report_dir / "meta.csv", index=False)
    return df


def parse_args():
    p = argparse.ArgumentParser(description="Prepare Sen2GF3Floods TIF files as RGB images and binary masks.")
    p.add_argument("--archive", default=r"C:\Users\17473\Downloads\Sen2GF3Floods.rar", help="Optional .rar archive path.")
    p.add_argument("--extract_dir", default="data/external", help="Directory where Sen2GF3Floods is/will be extracted.")
    p.add_argument("--dataset_root", default="", help="Use an already extracted Sen2GF3Floods root instead of --archive.")
    p.add_argument("--output_dir", default="data/sen2gf3/raw")
    p.add_argument("--rgb_mode", default="sentinel", choices=["sentinel", "flood_pseudo"])
    p.add_argument("--rgb_channels", default="0,1,2", help="Zero-based channels selected from Sentinel2 TIF.")
    p.add_argument("--stretch", default="none", choices=["none", "percentile"])
    p.add_argument("--image_ext", default=".jpg", choices=[".jpg", ".png", "jpg", "png"])
    p.add_argument("--limit", type=int, default=None, help="Convert only the first N samples, useful for previews.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--force_extract", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    archive = Path(args.archive) if args.archive else None
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        dataset_root = extract_archive_if_needed(archive, Path(args.extract_dir), force_extract=args.force_extract)

    df = prepare_dataset(
        dataset_root=dataset_root,
        output_dir=Path(args.output_dir),
        rgb_mode=args.rgb_mode,
        rgb_channels=parse_channels(args.rgb_channels),
        stretch=args.stretch,
        image_ext=args.image_ext,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"Converted {len(df)} samples.")
    print(f"RGB images: {Path(args.output_dir) / 'images'}")
    print(f"Binary masks: {Path(args.output_dir) / 'masks'}")
    print(f"Report: {Path(args.output_dir) / 'reports' / 'meta.csv'}")


if __name__ == "__main__":
    main()

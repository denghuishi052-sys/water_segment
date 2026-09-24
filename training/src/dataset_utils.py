from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageMaskPair:
    stem: str
    image_path: Path
    mask_path: Optional[Path]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(directory: str | Path, exts: set[str]) -> List[Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    files: List[Path] = []
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def build_stem_map(files: Sequence[Path]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}
    for f in files:
        stem = f.stem
        if stem in mapping:
            duplicates.setdefault(stem, [mapping[stem]]).append(f)
        else:
            mapping[stem] = f
    if duplicates:
        msg = "Duplicate file stems detected. Please make file names unique. Examples: "
        msg += "; ".join(f"{k}: {[str(x) for x in v[:3]]}" for k, v in list(duplicates.items())[:5])
        raise ValueError(msg)
    return mapping


def match_image_mask_pairs(image_dir: str | Path, mask_dir: str | Path, allow_missing_masks: bool = True) -> List[ImageMaskPair]:
    image_files = list_files(image_dir, IMAGE_EXTS)
    mask_files = list_files(mask_dir, MASK_EXTS) if Path(mask_dir).exists() else []
    mask_map = build_stem_map(mask_files)
    pairs: List[ImageMaskPair] = []
    missing: List[str] = []
    for img in image_files:
        m = mask_map.get(img.stem)
        if m is None and not allow_missing_masks:
            missing.append(img.name)
        pairs.append(ImageMaskPair(stem=img.stem, image_path=img, mask_path=m))
    if missing:
        raise FileNotFoundError(f"Missing masks for {len(missing)} images. Examples: {missing[:10]}")
    return pairs


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def _parse_rgb(value: str | Sequence[int] | Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(value, str):
        parts = [int(x.strip()) for x in value.split(",") if x.strip() != ""]
    else:
        parts = [int(x) for x in value]
    if len(parts) != 3:
        raise ValueError(f"foreground_rgb must contain exactly 3 values, got: {value}")
    return tuple(parts)  # type: ignore[return-value]


def read_mask_binary(
    path: str | Path | None,
    image_shape: Optional[Tuple[int, int]] = None,
    mask_mode: str = "red",
    foreground_rgb: str | Sequence[int] | Tuple[int, int, int] = (128, 0, 0),
    tolerance: int = 10,
) -> np.ndarray:
    """Read a segmentation mask and convert it to uint8 binary {0,1}.

    Supported mask modes:
        - red: color mask where foreground is close to foreground_rgb, e.g. RGB=(128,0,0).
        - non_black: any non-black pixel is foreground.
        - grayscale: any gray value > tolerance is foreground.
        - auto: red for color masks, grayscale for single-channel masks.

    If path is None, return an empty mask with image_shape.
    """
    if path is None:
        if image_shape is None:
            raise ValueError("image_shape is required when mask path is None")
        return np.zeros(image_shape, dtype=np.uint8)

    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to read mask: {path}")

    mode = mask_mode.lower().strip()
    if mode not in {"red", "non_black", "grayscale", "auto"}:
        raise ValueError(f"Unsupported mask_mode={mask_mode}. Use red, non_black, grayscale or auto.")

    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]

    if mask.ndim == 3:
        # Drop alpha channel if present, then convert OpenCV BGR to RGB.
        rgb = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2RGB)
        if mode == "auto":
            mode = "red"

        if mode == "red":
            target = np.array(_parse_rgb(foreground_rgb), dtype=np.int16)
            diff = np.abs(rgb.astype(np.int16) - target.reshape(1, 1, 3))
            binary = np.all(diff <= int(tolerance), axis=-1).astype(np.uint8)
        elif mode == "non_black":
            binary = np.any(rgb > int(tolerance), axis=-1).astype(np.uint8)
        else:  # grayscale requested for a color mask
            gray = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
            binary = (gray > int(tolerance)).astype(np.uint8)
    else:
        if mask.dtype != np.uint8:
            mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        binary = (mask > int(tolerance)).astype(np.uint8)

    if image_shape is not None and binary.shape[:2] != image_shape:
        binary = cv2.resize(binary, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return binary.astype(np.uint8)


def read_mask_multiclass(
    path: str | Path,
    num_classes: int = 3,
    image_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Read a multi-class segmentation mask as uint8 with values in 0..num_classes-1.

    Handles:
        - Single-channel PNG with class indices as pixel values.
        - 3-channel PNG where R=G=B=class_index (takes first channel).
        - 3-channel PNG with a squeezed (H, W, 1) shape.

    No binarization or thresholding is applied — the raw class index is preserved.
    """
    if path is None:
        if image_shape is None:
            raise ValueError("image_shape is required when mask path is None")
        return np.zeros(image_shape, dtype=np.uint8)

    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to read mask: {path}")

    # Squeeze (H, W, 1) → (H, W)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]

    if mask.ndim == 3:
        # 3-channel: if all channels identical (class index repeated), take first.
        if np.all(mask[:, :, 0] == mask[:, :, 1]) and np.all(mask[:, :, 1] == mask[:, :, 2]):
            mask = mask[:, :, 0]
        else:
            # Take first channel as best guess for class index.
            mask = mask[:, :, 0]

    if mask.dtype != np.uint8:
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if image_shape is not None and mask.shape[:2] != image_shape:
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask.astype(np.uint8)


def mask_area_stats(mask: np.ndarray) -> Tuple[int, float, str]:
    h, w = mask.shape[:2]
    area = int((mask > 0).sum())
    ratio = float(area / max(h * w, 1))
    if area == 0:
        bucket = "empty"
    elif ratio <= 0.03:
        bucket = "small"
    elif ratio <= 0.15:
        bucket = "medium"
    else:
        bucket = "large"
    return area, ratio, bucket


def infer_group_id(stem: str) -> str:
    """Infer a scene/video/camera group from file stem.

    Examples:
    camera001_00023 -> camera001
    video-12-frame-0008 -> video-12
    abc001 -> abc001
    """
    parts = re.split(r"[_\-\s]+", stem)
    # Prefer the prefix before the last numeric frame token.
    if len(parts) >= 2 and re.fullmatch(r"\d+", parts[-1]):
        return "_".join(parts[:-1])
    return parts[0] if parts else stem


def safe_copy(src: str | Path, dst: str | Path) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def save_empty_mask(dst: str | Path, shape: Tuple[int, int]) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), np.zeros(shape, dtype=np.uint8))


def write_yaml(path: str | Path, payload: dict) -> None:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

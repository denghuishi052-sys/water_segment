from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


def remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    binary = (binary > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out


def binary_mask_to_polygons(
    binary: np.ndarray,
    min_area: int = 20,
    epsilon_ratio: float = 0.002,
    min_points: int = 3,
) -> List[List[Tuple[float, float]]]:
    """Convert binary mask to normalized YOLO segmentation polygons.

    Returns list of polygons. Each polygon is list of normalized (x, y).
    """
    binary = (binary > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    if h == 0 or w == 0:
        return []
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[Tuple[float, float]]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, closed=True)
        epsilon = max(1.0, epsilon_ratio * peri)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        pts = approx.reshape(-1, 2)
        if len(pts) < min_points:
            x, y, bw, bh = cv2.boundingRect(cnt)
            pts = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.float32)
        poly: List[Tuple[float, float]] = []
        for x, y in pts:
            xn = min(max(float(x) / w, 0.0), 1.0)
            yn = min(max(float(y) / h, 0.0), 1.0)
            poly.append((xn, yn))
        if len(poly) >= min_points:
            polygons.append(poly)
    return polygons


def write_yolo_seg_txt(path: str | Path, polygons: Sequence[Sequence[Tuple[float, float]]], class_id: int = 0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for poly in polygons:
        flat: List[str] = [str(class_id)]
        for x, y in poly:
            flat.extend([f"{x:.6f}", f"{y:.6f}"])
        if len(flat) >= 7:
            lines.append(" ".join(flat))
    path.write_text("\n".join(lines), encoding="utf-8")


def read_yolo_seg_txt(path: str | Path, width: int, height: int, class_id: int = 0) -> List[np.ndarray]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    polys: List[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        if int(float(parts[0])) != class_id:
            continue
        coords = [float(x) for x in parts[1:]]
        if len(coords) % 2 != 0:
            continue
        pts = []
        for x, y in zip(coords[0::2], coords[1::2]):
            pts.append([int(round(x * width)), int(round(y * height))])
        if len(pts) >= 3:
            polys.append(np.array(pts, dtype=np.int32))
    return polys


def polygons_to_mask(polygons: Sequence[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for p in polygons:
        if p is not None and len(p) >= 3:
            cv2.fillPoly(mask, [p.astype(np.int32)], 1)
    return mask


def postprocess_mask(
    mask: np.ndarray,
    min_area_ratio: float = 0.0005,
    morph_close: bool = True,
    kernel_size: int = 3,
) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape[:2]
    min_area = max(1, int(h * w * min_area_ratio))
    mask = remove_small_components(mask, min_area=min_area)
    if morph_close:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return (mask > 0).astype(np.uint8)

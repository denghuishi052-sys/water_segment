from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8)
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45, color=(0, 0, 255)) -> np.ndarray:
    out = image_bgr.copy()
    mask_bool = _binary_mask(mask, image_bgr.shape[:2])
    colored = np.zeros_like(out)
    colored[:, :] = color
    out[mask_bool] = cv2.addWeighted(out, 1 - alpha, colored, alpha, 0)[mask_bool]
    return out


def overlay_mask_with_contour(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.85,
    color=(0, 255, 0),
    contour_color=(255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    mask_bool = _binary_mask(mask, image_bgr.shape[:2])
    out = image_bgr.copy()
    if mask_bool.any():
        out = (image_bgr * 0.55).astype(np.uint8)
        colored = np.zeros_like(out)
        colored[:, :] = color
        highlighted = cv2.addWeighted(image_bgr, 1 - alpha, colored, alpha, 0)
        out[mask_bool] = highlighted[mask_bool]
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, contour_color, thickness)
    return out


def draw_contours(image_bgr: np.ndarray, mask: np.ndarray, color=(0, 255, 255), thickness: int = 2) -> np.ndarray:
    out = image_bgr.copy()
    mask_uint8 = _binary_mask(mask, image_bgr.shape[:2]).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def mask_panel(mask: np.ndarray, shape: tuple[int, int], color=(0, 255, 0)) -> np.ndarray:
    panel = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    mask_bool = _binary_mask(mask, shape)
    panel[mask_bool] = color
    return panel


def comparison_panel(image_bgr: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha: float = 0.75) -> np.ndarray:
    shape = image_bgr.shape[:2]
    gt_bool = _binary_mask(gt, shape)
    pred_bool = _binary_mask(pred, shape)
    out = image_bgr.copy()
    colors = np.zeros_like(image_bgr)

    # BGR colors: TP green, FP red, FN blue.
    colors[gt_bool & pred_bool] = (0, 220, 0)
    colors[~gt_bool & pred_bool] = (0, 0, 255)
    colors[gt_bool & ~pred_bool] = (255, 0, 0)

    mask_bool = np.any(colors > 0, axis=-1)
    if mask_bool.any():
        out = (image_bgr * 0.55).astype(np.uint8)
        blended = cv2.addWeighted(image_bgr, 1 - alpha, colors, alpha, 0)
        out[mask_bool] = blended[mask_bool]

    gt_contours, _ = cv2.findContours(gt_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(pred_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, gt_contours, -1, (255, 255, 255), 2)
    cv2.drawContours(out, pred_contours, -1, (0, 255, 255), 1)
    return out


def make_panel(image_bgr: np.ndarray, gt: Optional[np.ndarray] = None, pred: Optional[np.ndarray] = None) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    panels = [image_bgr]
    labels = ["Image"]
    if gt is not None:
        panels.append(overlay_mask_with_contour(image_bgr, gt, alpha=0.55, color=(0, 255, 0)))
        labels.append("GT")
    if pred is not None:
        panels.append(overlay_mask(image_bgr, pred, color=(0, 0, 255)))
        labels.append("Pred")
    if gt is not None and pred is not None:
        panels.append(comparison_panel(image_bgr, gt, pred))
        labels.append("TP/FP/FN")
    canvas = np.concatenate(panels, axis=1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, text in enumerate(labels):
        x = i * w + 10
        text_size, _ = cv2.getTextSize(text, font, 0.8, 2)
        cv2.rectangle(canvas, (x - 5, 5), (x + text_size[0] + 12, 38), (0, 0, 0), -1)
        cv2.putText(canvas, text, (x, 28), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def save_panel(path: str | Path, image_bgr: np.ndarray, gt: Optional[np.ndarray] = None, pred: Optional[np.ndarray] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), make_panel(image_bgr, gt, pred))

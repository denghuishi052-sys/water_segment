from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from dual_context_water.dataset import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int


def compute_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[Tile]:
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        values = list(range(0, length - tile_size + 1, stride))
        final = length - tile_size
        if values[-1] != final:
            values.append(final)
        return values

    return [
        Tile(x, y, min(x + tile_size, width), min(y + tile_size, height))
        for y in starts(height)
        for x in starts(width)
    ]


def _normalize(image_rgb: np.ndarray) -> np.ndarray:
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image.transpose(2, 0, 1)[None].astype(np.float32)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _hann_weight(size: int, edge_floor: float = 0.08) -> np.ndarray:
    axis = np.hanning(size).astype(np.float32)
    weight = np.outer(axis, axis)
    weight /= max(float(weight.max()), 1e-6)
    return edge_floor + (1.0 - edge_floor) * weight


def hysteresis_mask(probability: np.ndarray, low: float, high: float) -> np.ndarray:
    strong = (probability >= high).astype(np.uint8)
    weak = (probability >= low).astype(np.uint8)
    if not strong.any():
        return strong
    count, labels = cv2.connectedComponents(weak, connectivity=8)
    if count <= 1:
        return strong
    keep = np.unique(labels[strong > 0])
    keep = keep[keep > 0]
    return np.isin(labels, keep).astype(np.uint8)


class DualContextOnnxPredictor:
    def __init__(
        self,
        model_path: str | Path,
        tile_size: int = 1024,
        global_size: int = 512,
        overlap: int = 256,
        providers: list[str] | None = None,
    ) -> None:
        self.tile_size = int(tile_size)
        self.global_size = int(global_size)
        self.overlap = int(overlap)
        requested = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        active = [provider for provider in requested if provider in available]
        self.session = ort.InferenceSession(str(model_path), providers=active)
        self.providers = self.session.get_providers()

    def predict_probability(self, image_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        height, width = image_bgr.shape[:2]
        global_rgb = cv2.cvtColor(
            cv2.resize(image_bgr, (self.global_size, self.global_size), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        global_normalized = _normalize(global_rgb)[0]
        tiles = compute_tiles(width, height, self.tile_size, self.overlap)
        weight_template = _hann_weight(self.tile_size)
        probability_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)
        quality_scores: list[float] = []

        for tile in tiles:
            crop = image_bgr[tile.y0:tile.y1, tile.x0:tile.x1]
            valid_h, valid_w = crop.shape[:2]
            padded = cv2.copyMakeBorder(
                crop,
                0,
                self.tile_size - valid_h,
                0,
                self.tile_size - valid_w,
                cv2.BORDER_REFLECT_101,
            )
            local_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            local_tensor = _normalize(local_rgb)

            roi = np.zeros((self.global_size, self.global_size), dtype=np.float32)
            gx0 = int(round(tile.x0 * self.global_size / max(width, 1)))
            gy0 = int(round(tile.y0 * self.global_size / max(height, 1)))
            gx1 = int(round(tile.x1 * self.global_size / max(width, 1)))
            gy1 = int(round(tile.y1 * self.global_size / max(height, 1)))
            roi[gy0:gy1, gx0:gx1] = 1.0
            context = np.concatenate([global_normalized, roi[None]], axis=0)[None].astype(np.float32)

            local_logits, _, quality_logits = self.session.run(
                None,
                {"local_image": local_tensor, "global_context": context},
            )
            probability = _sigmoid(local_logits[0, 0, :valid_h, :valid_w])
            quality_scores.append(float(_sigmoid(quality_logits)[0, 0]))
            weight = weight_template[:valid_h, :valid_w]
            probability_sum[tile.y0:tile.y1, tile.x0:tile.x1] += probability * weight
            weight_sum[tile.y0:tile.y1, tile.x0:tile.x1] += weight

        probability = probability_sum / np.maximum(weight_sum, 1e-6)
        info = {
            "tile_count": len(tiles),
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "providers": self.providers,
            "mean_quality": float(np.mean(quality_scores)) if quality_scores else 0.0,
        }
        return probability, info

    def predict_mask(
        self,
        image_bgr: np.ndarray,
        low_threshold: float = 0.40,
        high_threshold: float = 0.65,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        probability, info = self.predict_probability(image_bgr)
        mask = hysteresis_mask(probability, low=low_threshold, high=high_threshold)
        info["pred_area_ratio"] = float(mask.mean()) if mask.size else 0.0
        return mask, probability, info

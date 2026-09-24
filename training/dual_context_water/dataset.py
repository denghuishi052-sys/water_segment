from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 127).astype(np.float32)


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return (image - IMAGENET_MEAN) / IMAGENET_STD


class DualContextDataset(Dataset):
    def __init__(self, manifest: str | Path, augment: bool = False) -> None:
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        with self.manifest.open("r", encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Manifest is empty: {self.manifest}")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, value: str) -> str:
        return str((self.root / value).resolve())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        local_rgb = _read_rgb(self._resolve(row["local_image"]))
        local_mask = _read_mask(self._resolve(row["local_mask"]))
        global_rgb = _read_rgb(self._resolve(row["global_image"]))
        global_mask = _read_mask(self._resolve(row["global_mask"]))

        global_h, global_w = global_rgb.shape[:2]
        full_w = max(int(row["full_width"]), 1)
        full_h = max(int(row["full_height"]), 1)
        x0 = int(round(int(row["x0"]) * global_w / full_w))
        y0 = int(round(int(row["y0"]) * global_h / full_h))
        x1 = int(round(int(row["x1"]) * global_w / full_w))
        y1 = int(round(int(row["y1"]) * global_h / full_h))
        roi = np.zeros((global_h, global_w), dtype=np.float32)
        roi[max(0, y0):min(global_h, y1), max(0, x0):min(global_w, x1)] = 1.0

        if self.augment:
            if random.random() < 0.5:
                local_rgb = np.ascontiguousarray(local_rgb[:, ::-1])
                local_mask = np.ascontiguousarray(local_mask[:, ::-1])
                global_rgb = np.ascontiguousarray(global_rgb[:, ::-1])
                global_mask = np.ascontiguousarray(global_mask[:, ::-1])
                roi = np.ascontiguousarray(roi[:, ::-1])
            if random.random() < 0.5:
                local_rgb = np.ascontiguousarray(local_rgb[::-1])
                local_mask = np.ascontiguousarray(local_mask[::-1])
                global_rgb = np.ascontiguousarray(global_rgb[::-1])
                global_mask = np.ascontiguousarray(global_mask[::-1])
                roi = np.ascontiguousarray(roi[::-1])
            contrast = random.uniform(0.85, 1.15)
            brightness = random.uniform(-12.0, 12.0)
            local_rgb = np.clip(local_rgb.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
            global_rgb = np.clip(global_rgb.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

        local_tensor = torch.from_numpy(_normalize_rgb(local_rgb).transpose(2, 0, 1)).float()
        global_rgb_tensor = torch.from_numpy(_normalize_rgb(global_rgb).transpose(2, 0, 1)).float()
        roi_tensor = torch.from_numpy(roi[None]).float()
        global_context = torch.cat([global_rgb_tensor, roi_tensor], dim=0)

        local_mask_tensor = torch.from_numpy(local_mask[None]).float()
        global_mask_tensor = torch.from_numpy(global_mask[None]).float()
        quality_target = torch.tensor([float(local_mask.mean() > 0.001)], dtype=torch.float32)
        sample_weight = torch.tensor(
            [float(row.get("sample_weight", 1.0) or 1.0)],
            dtype=torch.float32,
        )
        return {
            "local_image": local_tensor,
            "global_context": global_context,
            "local_mask": local_mask_tensor,
            "global_mask": global_mask_tensor,
            "quality_target": quality_target,
            "sample_weight": sample_weight,
        }

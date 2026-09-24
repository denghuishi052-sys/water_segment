from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "11_augment_floodnet_binary.py"


def load_module():
    spec = importlib.util.spec_from_file_location("augment_floodnet_binary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_binarize_multiclass_mask_merges_floodnet_foreground_classes():
    module = load_module()
    mask = np.array(
        [
            [0, 1, 2],
            [2, 0, 1],
        ],
        dtype=np.uint8,
    )

    binary = module.binarize_multiclass_mask(mask)

    assert binary.dtype == np.uint8
    assert binary.tolist() == [[0, 1, 1], [1, 0, 1]]


def test_write_binary_sample_creates_single_class_yolo_label(tmp_path):
    module = load_module()
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 10:22] = 1
    out = tmp_path / "processed"

    module.write_binary_sample(
        image=image,
        binary_mask=mask,
        out_base=out,
        split="train",
        stem="sample_aug",
        image_ext=".jpg",
        min_area=4,
        epsilon_ratio=0.002,
    )

    saved_mask = cv2.imread(str(out / "masks" / "train" / "sample_aug.png"), cv2.IMREAD_UNCHANGED)
    label_text = (out / "labels" / "train" / "sample_aug.txt").read_text(encoding="utf-8").strip()

    assert sorted(np.unique(saved_mask).tolist()) == [0, 1]
    assert label_text
    assert {line.split()[0] for line in label_text.splitlines()} == {"0"}


def test_write_binary_sample_can_resize_image_and_mask_together(tmp_path):
    module = load_module()
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[8:24, 12:36] = 1
    out = tmp_path / "processed"

    module.write_binary_sample(
        image=image,
        binary_mask=mask,
        out_base=out,
        split="train",
        stem="sample_resized",
        image_ext=".jpg",
        min_area=4,
        epsilon_ratio=0.002,
        output_size=16,
    )

    saved_image = cv2.imread(str(out / "images" / "train" / "sample_resized.jpg"), cv2.IMREAD_COLOR)
    saved_mask = cv2.imread(str(out / "masks" / "train" / "sample_resized.png"), cv2.IMREAD_UNCHANGED)
    label_text = (out / "labels" / "train" / "sample_resized.txt").read_text(encoding="utf-8").strip()

    assert saved_image.shape[:2] == (16, 16)
    assert saved_mask.shape[:2] == (16, 16)
    assert label_text

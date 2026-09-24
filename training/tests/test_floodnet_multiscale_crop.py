from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "12_build_floodnet_multiscale_crop.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_floodnet_multiscale_crop", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def test_letterbox_pair_preserves_aspect_ratio_and_binary_mask():
    module = load_module()
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    mask = np.zeros((40, 80), dtype=np.uint8)
    image[10:30, 20:60] = (255, 255, 255)
    mask[10:30, 20:60] = 1

    out_image, out_mask, meta = module.letterbox_pair(image, mask, output_size=64)

    assert out_image.shape == (64, 64, 3)
    assert out_mask.shape == (64, 64)
    assert meta["resized_width"] == 64
    assert meta["resized_height"] == 32
    assert meta["pad_top"] == 16
    assert meta["pad_left"] == 0
    assert set(np.unique(out_mask).tolist()) <= {0, 1}
    image_foreground = cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY) > 200
    assert foreground_bbox(image_foreground) == foreground_bbox(out_mask)


def test_crop_and_letterbox_keep_image_and_mask_aligned():
    module = load_module()
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    mask = np.zeros((100, 160), dtype=np.uint8)
    image[25:75, 50:130] = (255, 255, 255)
    mask[25:75, 50:130] = 1

    crop_image, crop_mask = module.crop_pair(image, mask, (30, 10, 150, 90))
    out_image, out_mask, _ = module.letterbox_pair(crop_image, crop_mask, output_size=96)

    image_foreground = cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY) > 200
    assert foreground_bbox(image_foreground) == foreground_bbox(out_mask)


def test_default_scale_plan_is_fixed_six_sample_pyramid():
    module = load_module()

    plan = module.build_scale_plan()

    assert [item.name for item in plan] == [
        "global",
        "large_01",
        "medium_01",
        "medium_02",
        "local_01",
        "local_02",
    ]
    assert [(item.min_scale, item.max_scale) for item in plan[1:]] == [
        (0.75, 0.90),
        (0.50, 0.70),
        (0.50, 0.70),
        (0.30, 0.50),
        (0.30, 0.50),
    ]


def test_sample_crop_window_stays_in_bounds_and_scale_range():
    module = load_module()
    rng = np.random.default_rng(42)
    mask = np.zeros((120, 200), dtype=np.uint8)
    mask[45:80, 80:135] = 1

    window = module.sample_crop_window(
        mask=mask,
        min_scale=0.50,
        max_scale=0.70,
        mode="foreground",
        rng=rng,
    )
    x1, y1, x2, y2 = window

    assert 0 <= x1 < x2 <= 200
    assert 0 <= y1 < y2 <= 120
    assert 0.50 <= (x2 - x1) / 200 <= 0.70
    assert 0.50 <= (y2 - y1) / 120 <= 0.70
    assert mask[y1:y2, x1:x2].any()


def test_write_sample_regenerates_normalized_binary_label_from_final_mask(tmp_path):
    module = load_module()
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    mask = np.zeros((50, 100), dtype=np.uint8)
    image[10:40, 25:75] = (255, 255, 255)
    mask[10:40, 25:75] = 1

    module.write_sample(
        image=image,
        binary_mask=mask,
        out_base=tmp_path,
        split="train",
        stem="rectangle",
        output_size=128,
        min_area=4,
        epsilon_ratio=0.001,
        jpeg_quality=95,
    )

    saved_mask = cv2.imread(
        str(tmp_path / "masks" / "train" / "rectangle.png"),
        cv2.IMREAD_UNCHANGED,
    )
    lines = (tmp_path / "labels" / "train" / "rectangle.txt").read_text(
        encoding="utf-8"
    ).strip().splitlines()

    assert lines
    for line in lines:
        values = line.split()
        assert values[0] == "0"
        coords = [float(value) for value in values[1:]]
        assert len(coords) >= 6
        assert len(coords) % 2 == 0
        assert all(0.0 <= value <= 1.0 for value in coords)

    polygons = module.read_yolo_seg_txt(
        tmp_path / "labels" / "train" / "rectangle.txt",
        width=128,
        height=128,
        class_id=0,
    )
    reconstructed = module.polygons_to_mask(polygons, saved_mask.shape)
    intersection = np.logical_and(reconstructed > 0, saved_mask > 0).sum()
    union = np.logical_or(reconstructed > 0, saved_mask > 0).sum()

    assert intersection / union >= 0.95


def test_write_sample_keeps_thin_region_without_bounding_box_explosion(tmp_path):
    module = load_module()
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.line(mask, (10, 110), (115, 15), color=1, thickness=3)
    image[mask > 0] = (255, 255, 255)

    module.write_sample(
        image=image,
        binary_mask=mask,
        out_base=tmp_path,
        split="train",
        stem="thin",
        output_size=128,
        min_area=4,
        epsilon_ratio=0.001,
        jpeg_quality=95,
    )

    saved_mask = cv2.imread(
        str(tmp_path / "masks" / "train" / "thin.png"),
        cv2.IMREAD_UNCHANGED,
    )
    polygons = module.read_yolo_seg_txt(
        tmp_path / "labels" / "train" / "thin.txt",
        width=128,
        height=128,
        class_id=0,
    )
    reconstructed = module.polygons_to_mask(polygons, saved_mask.shape)

    assert np.array_equal(reconstructed, saved_mask)
    assert int(saved_mask.sum()) <= int(mask.sum()) * 2

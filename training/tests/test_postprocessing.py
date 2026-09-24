"""Unit tests for ``waterseg_platform.postprocessing``.

Run with:
    cd D:/project/water_segment
    C:/Users/17473/miniforge3/envs/torch_env/python.exe -m pytest tests/test_postprocessing.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waterseg_platform.postprocessing import (
    decode_segmentation,
    nms_class_agnostic,
    sigmoid,
)
from waterseg_platform.preprocessing import LetterboxMeta


def test_sigmoid_basic() -> None:
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
    out = sigmoid(x)
    expected = np.array([4.5e-5, 0.269, 0.5, 0.731, 0.99995], dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-3)


def test_sigmoid_stable_for_large_negatives() -> None:
    x = np.array([-1000.0, -100.0, -50.0], dtype=np.float32)
    out = sigmoid(x)
    assert (out >= 0).all() and (out <= 1).all()
    # No NaN/Inf even at very negative inputs.
    assert np.isfinite(out).all()


def test_nms_class_agnostic_basic() -> None:
    """Two highly-overlapping boxes with the same score: only the first wins."""
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # heavy overlap with box 0
            [50.0, 50.0, 60.0, 60.0],  # disjoint
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms_class_agnostic(boxes, scores, iou_thres=0.5)
    # Box 0 keeps; box 1 is suppressed (IoU > 0.5); box 2 keeps.
    assert set(keep.tolist()) == {0, 2}


def test_nms_empty() -> None:
    boxes = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros((0,), dtype=np.float32)
    keep = nms_class_agnostic(boxes, scores, iou_thres=0.5)
    assert keep.shape == (0,)


def test_nms_lower_score_first() -> None:
    """If the lower-score box is encountered first, it still gets suppressed."""
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32
    )
    scores = np.array([0.1, 0.9], dtype=np.float32)  # higher score is index 1
    keep = nms_class_agnostic(boxes, scores, iou_thres=0.5)
    # Highest-score box (1) wins, lower-score (0) is suppressed.
    assert keep.tolist() == [1]


def test_decode_segmentation_no_detections_returns_empty_mask() -> None:
    """When no anchor passes the conf threshold, the mask is all-zero."""
    A = 100
    output0 = np.full((1, 37, A), -10.0, dtype=np.float32)  # all very low scores
    output1 = np.zeros((1, 32, 176, 176), dtype=np.float32)
    meta = LetterboxMeta(r=2.75, pad_w=0, pad_h=0, new_w=704, new_h=704, imgsz=704)
    mask = decode_segmentation(
        output0, output1, meta, original_shape=(256, 256),
        conf=0.25, iou=0.5, mask_thres=0.5,
    )
    assert mask.shape == (256, 256)
    assert mask.sum() == 0


def test_decode_segmentation_synthetic_one_detection() -> None:
    """Build a synthetic output pair with one obvious box + mask, decode it.

    Setup:
      - Proto map is all-zero EXCEPT a 1 in a 10x10 region at (50, 50)..(60, 60).
      - Coeffs are 0 except a single +10 entry, so sigmoid(10*1) ≈ 0.99995
        in that region and ≈ 0 elsewhere.
      - Box centered at (55*4=220, 55*4=220) in letterbox coords, big enough
        to contain the proto region.
    """
    imgsz = 704
    mh = mw = imgsz // 4
    nm, nc = 32, 1

    # Build a proto map with a 10x10 block of 1.0 in the top-left.
    proto = np.zeros((nm, mh, mw), dtype=np.float32)
    proto[0, 50:60, 50:60] = 1.0  # one channel has the foreground

    # Coefficients: channel 0 is +10, the rest are 0. So each detection's mask
    # is sigmoid(10 * proto[0]) which is ~0 in most places and ~1 in [50:60, 50:60].
    # Build a detection at (cx, cy) = (220, 220) with a 100x100 box.
    cx, cy, bw, bh = 220.0, 220.0, 200.0, 200.0
    x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    boxes = np.array([[cx, cy, bw, bh]], dtype=np.float32).T  # (4, 1)

    # Class score: a single anchor, score = 5 (sigmoid(5) ≈ 0.993).
    cls_scores = np.full((nc, 1), 5.0, dtype=np.float32)
    coeffs = np.zeros((nm, 1), dtype=np.float32)
    coeffs[0, 0] = 10.0

    output0 = np.concatenate([boxes, cls_scores, coeffs], axis=0)[None, ...]  # (1, 37, 1)
    output1 = proto[None, ...]  # (1, 32, 176, 176)

    meta = LetterboxMeta(r=imgsz / 256, pad_w=0, pad_h=0, new_w=imgsz, new_h=imgsz, imgsz=imgsz)
    # Original image is 256x256, so r = 704/256 = 2.75, no padding.
    mask = decode_segmentation(
        output0, output1, meta, original_shape=(256, 256),
        conf=0.25, iou=0.5, mask_thres=0.5, max_det=10,
    )
    assert mask.shape == (256, 256)
    # The proto region (10x10) is upsampled by INTER_LINEAR into the
    # box (200x200 letterbox pixels) which becomes (~73x73) in the 256x256
    # original image. With sigmoid(10) ~= 1.0 in the proto region and 0
    # outside, the bilinear upsampled region plus the 0.5-threshold means
    # the full ~73x73 box area ends up >= 0.5. Expected mask area: ~5329 px.
    assert mask.sum() > 0, "Decoded mask should be non-empty"
    ys, xs = np.where(mask > 0)
    assert 4000 < len(ys) < 6500, f"Unexpected number of mask pixels: {len(ys)}"
    # The mask should be in the upper-left quadrant of the 256x256 image.
    assert xs.mean() < 128
    assert ys.mean() < 128


def test_decode_segmentation_mask_box_expand_increases_crop_area() -> None:
    imgsz = 704
    mh = mw = imgsz // 4
    nm, nc = 32, 1

    proto = np.zeros((nm, mh, mw), dtype=np.float32)
    proto[0, :, :] = 1.0
    cx, cy, bw, bh = 352.0, 352.0, 176.0, 176.0
    boxes = np.array([[cx, cy, bw, bh]], dtype=np.float32).T
    cls_scores = np.full((nc, 1), 5.0, dtype=np.float32)
    coeffs = np.zeros((nm, 1), dtype=np.float32)
    coeffs[0, 0] = 10.0
    output0 = np.concatenate([boxes, cls_scores, coeffs], axis=0)[None, ...]
    output1 = proto[None, ...]
    meta = LetterboxMeta(
        r=imgsz / 256, pad_w=0, pad_h=0, new_w=imgsz, new_h=imgsz, imgsz=imgsz
    )

    base = decode_segmentation(
        output0, output1, meta, original_shape=(256, 256),
        conf=0.25, iou=0.5, mask_thres=0.9, max_det=10,
        mask_box_expand_ratio=0.0,
    )
    expanded = decode_segmentation(
        output0, output1, meta, original_shape=(256, 256),
        conf=0.25, iou=0.5, mask_thres=0.9, max_det=10,
        mask_box_expand_ratio=1.0,
    )

    assert expanded.sum() > base.sum()
    assert expanded.sum() >= base.sum() * 3


def test_decode_segmentation_rejects_wrong_output0_layout() -> None:
    bad = np.zeros((2, 37, 100), dtype=np.float32)  # batch=2 instead of 1
    good_proto = np.zeros((1, 32, 176, 176), dtype=np.float32)
    meta = LetterboxMeta(r=1.0, pad_w=0, pad_h=0, new_w=704, new_h=704, imgsz=704)
    with pytest.raises(ValueError, match="output0"):
        decode_segmentation(bad, good_proto, meta, (256, 256))


def test_decode_segmentation_rejects_nm_mismatch() -> None:
    """If output1's nm != configured nm, raise clearly."""
    output0 = np.zeros((1, 37, 1), dtype=np.float32)
    output0[0, 4, 0] = 10.0  # high class score
    bad_proto = np.zeros((1, 16, 176, 176), dtype=np.float32)  # nm=16, not 32
    meta = LetterboxMeta(r=1.0, pad_w=0, pad_h=0, new_w=704, new_h=704, imgsz=704)
    with pytest.raises(ValueError, match="nm"):
        decode_segmentation(output0, bad_proto, meta, (256, 256),
                            conf=0.0, iou=0.5, mask_thres=0.5, nm=32)

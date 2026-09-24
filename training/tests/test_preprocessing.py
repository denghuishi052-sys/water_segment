"""Unit tests for ``waterseg_platform.preprocessing``.

Run with:
    cd D:/project/water_segment
    C:/Users/17473/miniforge3/envs/torch_env/python.exe -m pytest tests/test_preprocessing.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waterseg_platform.preprocessing import LetterboxMeta, letterbox, to_nchw_float


def _make_image(h: int, w: int, color=(10, 20, 30)) -> np.ndarray:
    """Construct a constant-color BGR image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def test_letterbox_square_input_returns_same_size() -> None:
    """A 704×704 input should not be resized or padded."""
    img = _make_image(704, 704)
    canvas, meta = letterbox(img, imgsz=704)
    assert canvas.shape == (704, 704, 3)
    assert meta.r == pytest.approx(1.0)
    assert meta.pad_w == 0 and meta.pad_h == 0
    assert meta.new_w == 704 and meta.new_h == 704
    np.testing.assert_array_equal(canvas, img)


def test_letterbox_320x240_into_704() -> None:
    """Letterbox math on a 320×240 input.

    r = 704 / max(320, 240) = 704 / 320 = 2.2
    new_w = round(320 * 2.2) = 704
    new_h = round(240 * 2.2) = 528
    pad_w = (704 - 704) // 2 = 0
    pad_h = (704 - 528) // 2 = 88
    """
    img = _make_image(240, 320)
    canvas, meta = letterbox(img, imgsz=704)
    assert canvas.shape == (704, 704, 3)
    assert meta.r == pytest.approx(2.2)
    assert meta.new_w == 704
    assert meta.new_h == 528
    assert meta.pad_w == 0
    assert meta.pad_h == 88

    # The pad region (top 88 rows + bottom 88 rows) must be filled with 114.
    assert (canvas[:88, :, :] == 114).all()
    assert (canvas[-88:, :, :] == 114).all()
    # The image region (rows 88..616) should be the resized constant color.
    interior = canvas[88:616, :, :]
    # (10, 20, 30) is the BGR value we wrote. Bilinear resize may shift
    # edges by 1 unit; check >= 99% of pixels match exactly.
    same = (interior == np.array([10, 20, 30], dtype=np.uint8)).all(axis=-1)
    assert same.mean() > 0.99


def test_letterbox_pad_centering() -> None:
    """For a 480×640 image, padding should be symmetric and centered."""
    img = _make_image(480, 640)
    canvas, meta = letterbox(img, imgsz=704)
    # r = 704/640 = 1.1 ; new_w = 704, new_h = 528
    assert meta.r == pytest.approx(1.1)
    assert meta.new_w == 704
    assert meta.new_h == 528
    assert meta.pad_w == 0
    assert meta.pad_h == 88
    # Pad rows are top 88 and bottom 88 → total canvas height 704.
    assert canvas.shape == (704, 704, 3)


def test_letterbox_meta_roundtrip() -> None:
    """LetterboxMeta.to_original / to_letterbox should be exact inverses."""
    meta = LetterboxMeta(r=2.2, pad_w=0, pad_h=88, new_w=704, new_h=528, imgsz=704)
    for x, y in [(50, 100), (300, 400), (0, 0), (704, 704)]:
        x_orig, y_orig = meta.to_original(x, y)
        x_back, y_back = meta.to_letterbox(x_orig, y_orig)
        assert x_back == pytest.approx(x, abs=1e-6)
        assert y_back == pytest.approx(y, abs=1e-6)


def test_to_nchw_float_shape_and_range() -> None:
    """to_nchw_float must produce a (1, 3, H, W) float32 tensor in [0, 1]."""
    img = _make_image(64, 96, color=(255, 128, 64))  # BGR
    tensor = to_nchw_float(img, assume_rgb=False)
    assert tensor.shape == (1, 3, 64, 96)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0
    # BGR→RGB: the 255 channel (B) ends up in the LAST C channel (R slot after transpose).
    # Input is (B=255, G=128, R=64) → RGB = (64, 128, 255) → /255 ≈ (0.251, 0.502, 1.0)
    expected = np.array([64, 128, 255], dtype=np.float32) / 255.0
    actual = tensor[0, :, 0, 0]
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_to_nchw_float_assume_rgb() -> None:
    """When assume_rgb=True, the BGR→RGB swap is skipped."""
    img = _make_image(8, 8, color=(255, 0, 0))  # already RGB red
    tensor_rgb = to_nchw_float(img, assume_rgb=True)
    # First channel (R) should be ≈ 1.0, G and B ≈ 0.
    assert tensor_rgb[0, 0, 0, 0] == pytest.approx(1.0, abs=1e-6)
    assert tensor_rgb[0, 1, 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert tensor_rgb[0, 2, 0, 0] == pytest.approx(0.0, abs=1e-6)


def test_to_nchw_float_contiguous() -> None:
    """Tensor must be C-contiguous (ONNX Runtime requires this)."""
    img = _make_image(32, 32)
    tensor = to_nchw_float(img)
    assert tensor.flags["C_CONTIGUOUS"] is True


def test_letterbox_rejects_bad_input() -> None:
    """A 2D grayscale image must be rejected with a clear error."""
    gray = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="BGR"):
        letterbox(gray, imgsz=64)

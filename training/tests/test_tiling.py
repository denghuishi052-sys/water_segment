"""Unit tests for the tiled-inference pipeline.

Pure-function tests for ``waterseg_platform.tiling`` and the new
``decode_segmentation_probs`` are fast (no model load). The two
integration tests that compare the tiled and single-tile paths
end-to-end are gated on the ONNX model existing on disk, and skip
otherwise (the parity test follows the same pattern).

Run with:
    cd D:/project/water_segment
    C:/Users/17473/miniforge3/envs/torch_env/python.exe -m pytest tests/test_tiling.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waterseg_platform.tiling import (  # noqa: E402
    COMBINE_MODE,
    Tile,
    compute_tiles,
    make_tile_weight_map,
    stitch_probability_maps,
    stitch_weighted_probability_maps,
)
from waterseg_platform.pipeline import (  # noqa: E402
    fuse_global_local_probabilities,
    suppress_border_components,
)
from waterseg_platform.postprocessing import (  # noqa: E402
    decode_segmentation_probs,
    decode_segmentation,
)
from waterseg_platform.preprocessing import LetterboxMeta  # noqa: E402

MODEL_PATH = ROOT / "onnx" / "floodnet_binary_aug_yolov8m_1024.onnx"


# ---------------------------------------------------------------------- #
# Pure-function tests for the tiling grid
# ---------------------------------------------------------------------- #

def test_compute_tiles_basic() -> None:
    """1500x1000 image, 704x704 tiles, no overlap -> 3x2 = 6 tiles.

    Stride = 704. ys = [0, 704, ... up to <= 1500-704+1=797] = [0, 704];
    last anchor pinned to 1500-704=796 so 796+704=1500.
    xs = [0]; last anchor pinned to 1000-704=296. So x_anchors = [0, 296].
    """
    tiles = compute_tiles(1500, 1000, tile_size=704, overlap_px=0)
    assert len(tiles) == 6  # 3 rows * 2 cols
    # Row-major order: y ascending, then x within each row.
    assert tiles[0] == Tile(0, 704, 0, 704)
    assert tiles[1] == Tile(0, 704, 296, 1000)
    assert tiles[2] == Tile(704, 1408, 0, 704)
    assert tiles[3] == Tile(704, 1408, 296, 1000)
    assert tiles[4] == Tile(796, 1500, 0, 704)
    assert tiles[5] == Tile(796, 1500, 296, 1000)
    # The full image is covered: union of (clipped) tile boxes == image.
    covered = np.zeros((1500, 1000), dtype=bool)
    for t in tiles:
        y0c = max(0, t.y0)
        x0c = max(0, t.x0)
        y1c = min(1500, t.y1)
        x1c = min(1000, t.x1)
        covered[y0c:y1c, x0c:x1c] = True
    assert covered.all(), "some pixels are uncovered by the tile grid"


def test_compute_tiles_with_overlap() -> None:
    """1500x1000 image, 704x704 tiles, 100 px overlap -> more tiles, full coverage."""
    tiles = compute_tiles(1500, 1000, tile_size=704, overlap_px=100)
    # Stride = 704-100 = 604. y_anchors = [0, 604, 1208] then 1208+704=1912>1500,
    # so the last anchor is pinned to 1500-704=796. -> [0, 604, 796]
    # x_anchors = [0, 604, 1000-704=296] pinned -> [0, 296]
    # That's 3x2 = 6 tiles (same count as no overlap here, but with overlapping
    # coverage in the interior). For a more discriminating case try 1500x1500
    # with 100 overlap: y_anchors=[0,604,1208], pinned 1500-704=796 -> [0,604,796]
    # -> 3 anchors. x same -> 3x3 = 9.
    tiles_2 = compute_tiles(1500, 1500, tile_size=704, overlap_px=100)
    assert len(tiles_2) == 9
    # The last tile in each row/column must end at the image edge.
    for t in tiles_2:
        if t.x1 > 1500 or t.y1 > 1500:
            raise AssertionError(f"tile {t} extends past image edge")
    # Coverage check.
    covered = np.zeros((1500, 1500), dtype=bool)
    for t in tiles_2:
        y0c = max(0, t.y0)
        x0c = max(0, t.x0)
        y1c = min(1500, t.y1)
        x1c = min(1500, t.x1)
        covered[y0c:y1c, x0c:x1c] = True
    assert covered.all(), "overlap grid leaves gaps"


def test_compute_tiles_single() -> None:
    """A single tile covers the whole image when the image is small enough.

    For H,W <= tile_size, we expect exactly 1 tile that covers the whole
    image (y0=0, y1=tile_size, x0=0, x1=tile_size). The actual valid
    region of the tile is the image itself; the rest is letterbox pad.
    """
    for h, w in [(500, 500), (256, 256), (700, 200), (200, 700)]:
        tiles = compute_tiles(h, w, tile_size=704, overlap_px=0)
        assert len(tiles) == 1, f"expected 1 tile for {h}x{w}, got {len(tiles)}"
        t = tiles[0]
        assert t.y0 == 0 and t.x0 == 0
        assert t.y1 == 704 and t.x1 == 704


def test_compute_tiles_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="tile_size"):
        compute_tiles(100, 100, tile_size=0, overlap_px=0)
    with pytest.raises(ValueError, match="overlap_px"):
        compute_tiles(100, 100, tile_size=704, overlap_px=-1)
    with pytest.raises(ValueError, match="H and W"):
        compute_tiles(0, 100, tile_size=704, overlap_px=0)


# ---------------------------------------------------------------------- #
# stitch_probability_maps
# ---------------------------------------------------------------------- #

def test_stitch_probability_maps_uniform() -> None:
    """Two-by-two uniform prob maps over a 1408x1408 image (no overlap).

    Use 1408 = 2*704 so the grid is exactly 2x2 with 0 overlap. The count
    map should be 1 everywhere, the accumulator should be 0.7 everywhere.
    """
    h, w = 1408, 1408
    tiles = compute_tiles(h, w, tile_size=704, overlap_px=0)
    assert len(tiles) == 4  # 2 rows * 2 cols
    # All tiles are (704, 704).
    for t in tiles:
        th = min(t.y1, h) - t.y0
        tw = min(t.x1, w) - t.x0
        assert (th, tw) == (704, 704)

    prob_maps = [np.full((704, 704), 0.7, dtype=np.float32) for _ in tiles]
    prob_full, count_full = stitch_probability_maps(prob_maps, tiles, h, w)
    # No overlap -> count is 1 everywhere.
    assert count_full.min() == 1
    assert count_full.max() == 1
    # Sum is 0.7 everywhere (uniform per-tile value).
    np.testing.assert_allclose(prob_full, 0.7, atol=1e-6)


def test_stitch_probability_maps_with_overlap() -> None:
    """Two overlapping tiles: count at the overlap should be 2."""
    h, w = 1000, 1000
    # tile_size=600, overlap=100 -> stride=500, 2x2 grid with overlap
    tiles = compute_tiles(h, w, tile_size=600, overlap_px=100)
    prob_maps = [
        np.full((600, 600), 0.4, dtype=np.float32),
        np.full((600, 600), 0.6, dtype=np.float32),
        np.full((600, 500), 0.5, dtype=np.float32),
        np.full((500, 500), 0.8, dtype=np.float32),
    ]
    prob_full, count_full = stitch_probability_maps(prob_maps, tiles, h, w)
    # Interior overlap regions should see count > 1.
    assert count_full.max() >= 2
    # Coverage must be complete.
    assert (count_full > 0).all()


def test_stitch_probability_maps_max_preserves_overlap_peak() -> None:
    """Max stitching keeps a confident tile from being diluted at seams."""
    h, w = 1000, 1000
    tiles = compute_tiles(h, w, tile_size=600, overlap_px=100)
    prob_maps = [
        np.full((600, 600), 0.9, dtype=np.float32),
        np.full((600, 600), 0.1, dtype=np.float32),
        np.full((600, 500), 0.2, dtype=np.float32),
        np.full((500, 500), 0.3, dtype=np.float32),
    ]
    prob_full, count_full = stitch_probability_maps(
        prob_maps,
        tiles,
        h,
        w,
        mode="max",
    )

    assert (count_full > 0).all()
    # x=550 lies in the overlap between the first two tiles. Average would
    # dilute this to 0.5; max keeps the stronger tile's probability.
    assert prob_full[100, 550] == pytest.approx(0.9)


def test_stitch_probability_maps_rejects_length_mismatch() -> None:
    """Pass too many prob maps -> must raise ValueError (not silently truncate)."""
    tiles = compute_tiles(700, 700, tile_size=704, overlap_px=0)
    assert len(tiles) == 1
    # 2 prob maps for 1 tile -> mismatch must raise.
    with pytest.raises(ValueError, match="same length"):
        stitch_probability_maps(
            [np.zeros((700, 700), dtype=np.float32),
             np.zeros((700, 700), dtype=np.float32)],
            tiles, 700, 700,
        )


def test_make_tile_weight_map_downweights_edges() -> None:
    weights = make_tile_weight_map(9, 9, mode="hann", edge_weight=0.25)
    assert weights.shape == (9, 9)
    assert weights.dtype == np.float32
    assert weights[4, 4] == pytest.approx(1.0)
    assert weights[0, 0] == pytest.approx(0.25)
    assert weights[0, 4] < weights[4, 4]


def test_stitch_weighted_probability_maps_uses_center_weight() -> None:
    h, w = 10, 14
    tiles = [Tile(0, 10, 0, 10), Tile(0, 10, 4, 14)]
    prob_maps = [
        np.full((10, 10), 0.2, dtype=np.float32),
        np.full((10, 10), 0.8, dtype=np.float32),
    ]
    prob_full, weight_full = stitch_weighted_probability_maps(
        prob_maps,
        tiles,
        h,
        w,
        mode="hann",
        edge_weight=0.25,
    )
    avg = np.zeros_like(prob_full)
    nz = weight_full > 0
    avg[nz] = prob_full[nz] / weight_full[nz]
    assert (weight_full > 0).all()
    # In the overlap, the second tile is near its center while the first
    # tile is near its right edge, so the fused value should lean high.
    assert avg[5, 8] > 0.5


def test_global_local_fusion_downweights_unsupported_tile_detection() -> None:
    global_prob = np.zeros((4, 4), dtype=np.float32)
    tile_prob = np.ones((4, 4), dtype=np.float32)
    fused = fuse_global_local_probabilities(
        global_prob,
        tile_prob,
        tile_weight=1.0,
        global_context_strength=0.5,
    )
    np.testing.assert_allclose(fused, 0.5)

    global_prob[1, 1] = 0.9
    fused = fuse_global_local_probabilities(
        global_prob,
        tile_prob,
        tile_weight=1.0,
        global_context_strength=0.5,
    )
    assert fused[1, 1] == pytest.approx(0.95)


def test_suppress_border_components_removes_border_artifact() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:6, 30:45] = 1
    mask[40:50, 40:50] = 1
    cleaned, info = suppress_border_components(
        mask,
        margin_ratio=0.01,
        margin_min_px=10,
        touch_ratio_thresh=0.35,
        max_component_area_ratio=0.08,
    )
    assert info["removed_components"] == 1
    assert cleaned[:6, 30:45].sum() == 0
    assert cleaned[40:50, 40:50].sum() == 100


def test_suppress_border_components_keeps_large_border_region() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:30, :] = 1
    cleaned, info = suppress_border_components(
        mask,
        margin_ratio=0.01,
        margin_min_px=10,
        touch_ratio_thresh=0.35,
        max_component_area_ratio=0.08,
    )
    assert info["removed_components"] == 0
    assert np.array_equal(cleaned, mask)


# ---------------------------------------------------------------------- #
# decode_segmentation_probs (the new helper the tiled pipeline uses)
# ---------------------------------------------------------------------- #

def test_decode_segmentation_probs_no_detections_returns_zeros() -> None:
    """With no anchors above conf, the output is all-zero float32."""
    A = 100
    output0 = np.full((1, 37, A), -10.0, dtype=np.float32)
    output1 = np.zeros((1, 32, 176, 176), dtype=np.float32)
    meta = LetterboxMeta(r=2.75, pad_w=0, pad_h=0, new_w=704, new_h=704, imgsz=704)
    prob = decode_segmentation_probs(
        output0, output1, meta, original_shape=(256, 256),
        conf=0.25, iou=0.5,
    )
    assert prob.shape == (256, 256)
    assert prob.dtype == np.float32
    assert prob.min() == 0.0 and prob.max() == 0.0


def test_decode_segmentation_probs_returns_float_in_unit_range() -> None:
    """Build a synthetic output with one detection; verify the result is float
    in [0, 1] (not a binary mask)."""
    imgsz = 704
    mh = mw = imgsz // 4
    nm, nc = 32, 1

    proto = np.zeros((nm, mh, mw), dtype=np.float32)
    proto[0, 50:60, 50:60] = 1.0

    cx, cy, bw, bh = 220.0, 220.0, 200.0, 200.0
    boxes = np.array([[cx, cy, bw, bh]], dtype=np.float32).T
    cls_scores = np.full((nc, 1), 5.0, dtype=np.float32)
    coeffs = np.zeros((nm, 1), dtype=np.float32)
    coeffs[0, 0] = 10.0
    output0 = np.concatenate([boxes, cls_scores, coeffs], axis=0)[None, ...]
    output1 = proto[None, ...]
    meta = LetterboxMeta(r=imgsz / 256, pad_w=0, pad_h=0, new_w=imgsz, new_h=imgsz, imgsz=imgsz)

    prob = decode_segmentation_probs(
        output0, output1, meta, original_shape=(256, 256),
        conf=0.25, iou=0.5, max_det=10,
    )
    assert prob.shape == (256, 256)
    assert prob.dtype == np.float32
    # Values are in [0, 1]
    assert prob.min() >= 0.0
    assert prob.max() <= 1.0
    # It's NOT a binary mask: at least one value should be strictly between 0 and 1
    assert ((prob > 0.0) & (prob < 1.0)).any(), "expected non-binary probs"


# ---------------------------------------------------------------------- #
# Integration tests: tiled vs single-tile on the real model
# ---------------------------------------------------------------------- #

@pytest.mark.skipif(not MODEL_PATH.exists(), reason=f"ONNX model not present at {MODEL_PATH}")
def test_segment_tiled_matches_segment_array_for_fits_image() -> None:
    """For an image that fits in 1024x1024, tiled and single-tile paths agree.

    A 500x500 image: compute_tiles gives 1 tile at (0,0,1024,1024). The tiled
    pipeline does crop->letterbox->engine.run->decode->stitch (no-op for 1
    tile)->threshold->postprocess. The single-tile pipeline does the
    same operations in slightly different order. The two must produce
    near-identical binary masks.

    Use a tolerance of mean |dIoU| < 0.01 and max |dIoU| < 0.05 to allow
    for any sub-pixel floating-point differences.
    """
    from waterseg_platform import PlatformConfig, SegmentationService
    from waterseg_platform.metrics import compute_binary_metrics

    cfg = PlatformConfig(
        model_path=str(MODEL_PATH), imgsz=1024, conf=0.25, iou=0.5,
        mask_thres=0.5, min_area_ratio=0.0005, morph_close=True,
        tile_size=1024, tile_overlap_px=0,
    )
    svc = SegmentationService(cfg)
    # A constant 500x500 image: the model sees a single tile in both paths.
    img = np.full((500, 500, 3), 80, dtype=np.uint8)

    mask_single, _ = svc.segment_array(img)
    mask_tiled, info_tiled = svc.segment_array_tiled(img)
    assert mask_tiled.shape == mask_single.shape == (500, 500)
    # Empty coarse predictions skip tiling by design; otherwise this image
    # requires exactly one 1024x1024 tile.
    assert info_tiled["tiling"]["num_tiles"] == 1

    m = compute_binary_metrics(mask_tiled, mask_single)
    # Symmetric IoU between the two predictions.
    iou = m.iou
    assert iou > 0.99, (
        f"tiled path (1 tile) should match single-tile path bit-exactly; "
        f"got IoU={iou:.4f} (mean abs delta of binary masks > 0.01)"
    )


@pytest.mark.skipif(not MODEL_PATH.exists(), reason=f"ONNX model not present at {MODEL_PATH}")
def test_segment_tiled_handles_rectangular_image() -> None:
    """The tiled path returns a (1500, 1000) mask for a 1500x1000 image."""
    from waterseg_platform import PlatformConfig, SegmentationService

    cfg = PlatformConfig(
        model_path=str(MODEL_PATH), imgsz=1024, conf=0.25, iou=0.5,
        mask_thres=0.5, min_area_ratio=0.0005, morph_close=True,
        tile_size=1024, tile_overlap_px=0,
    )
    svc = SegmentationService(cfg)
    img = np.full((1500, 1000, 3), 80, dtype=np.uint8)

    mask, info = svc.segment_array_tiled(img)
    assert mask.shape == (1500, 1000)
    # Empty coarse predictions skip tiling; otherwise 1024px tiles cover this
    # image with two vertical tiles and one horizontal tile.
    assert info["tiling"]["num_tiles"] == 2
    assert info["tiling"]["combine_mode"] == cfg.tile_stitch_mode
    assert info["tiling"]["tile_size"] == 1024
    # mask_area_ratio must be a valid float (constant image => no detections)
    assert isinstance(info["pred_area_ratio"], float)
    assert 0.0 <= info["pred_area_ratio"] <= 1.0

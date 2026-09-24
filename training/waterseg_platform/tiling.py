"""Tiling utilities for the ONNX water-segmentation platform.

Splits an arbitrarily-sized image into a regular grid of overlapping
``tile_size x tile_size`` patches so each patch can be fed through the
model independently. Per-tile float probability maps are then stitched
back to the original image size with uniform averaging (count-weighted).

Public surface:
    Tile                          -- frozen dataclass with y0, y1, x0, x1
    compute_tiles(H, W, tile_size, overlap_px) -> list[Tile]
    make_tile_weight_map(h, w, mode, edge_weight) -> ndarray
    stitch_probability_maps(prob_maps, tiles, H, W)
        -> (prob_full, count_full)
    stitch_weighted_probability_maps(prob_maps, tiles, H, W, ...)
        -> (weighted_prob_full, weight_full)

The platform's combined pipeline (segment_array_tiled) wires these
together with the decoder and engine. See README.md "Tiled inference"
section for the full data flow.

.NET equivalent: ``Tiling.cs`` (pure C# math + OpenCvSharp4 for resize).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


# The only supported combine mode in v1. Hardcoded as a constant so the
# tiling path's behavior is explicit; if a second mode is ever added,
# promote this to a config field.
COMBINE_MODE: str = "average"


@dataclass(frozen=True)
class Tile:
    """A single tile in original-image coordinates.

    Attributes:
        y0, x0: Inclusive top-left corner of the full tile box.
        y1, x1: Exclusive bottom-right corner of the full tile box.
                Note: the *valid* region of the tile may be smaller than
                this box at the right/bottom edges of the image (use
                ``min(tile.y1, H) - tile.y0`` etc.). The crop is clipped
                to image bounds, then letterboxed to (tile_size,
                tile_size) for the model.
    """

    y0: int
    y1: int
    x0: int
    x1: int


def compute_tiles(
    H: int, W: int, tile_size: int = 704, overlap_px: int = 0,
) -> List[Tile]:
    """Return a row-major grid of ``Tile`` covering the ``H x W`` image.

    Args:
        H, W:        Image height and width in pixels.
        tile_size:   Side length of every tile in pixels. Must be > 0.
                     In v1 this equals the ONNX export's imgsz (704).
        overlap_px:  Overlap between adjacent tiles in pixels. Must be
                     in ``[0, tile_size - 1]``. 0 means no overlap
                     (tiles abut; last tile is clipped to image bounds).

    Returns:
        List of :class:`Tile` objects, one per tile, in row-major order
        (y0 ascending, then x0 ascending within each row).

    Notes:
        * For ``H <= tile_size`` and ``W <= tile_size`` the result is a
          single tile covering the whole image.
        * The last tile in each row/column is forced to end at the
          image's right/bottom edge so coverage is exact.
        * ``overlap_px >= tile_size`` would produce an infinite loop, so
          it is clamped: ``stride = max(1, tile_size - overlap_px)``.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be > 0, got {tile_size}")
    if overlap_px < 0:
        raise ValueError(f"overlap_px must be >= 0, got {overlap_px}")
    if H <= 0 or W <= 0:
        raise ValueError(f"H and W must be > 0, got H={H}, W={W}")

    stride = max(1, tile_size - overlap_px)

    # Anchor positions. We generate starts from 0 up to and including
    # the start that would produce a tile ending at the image's right
    # or bottom edge (or, when the image is smaller than tile_size, a
    # single anchor at 0).
    ys: List[int] = list(range(0, max(1, H - tile_size + 1), stride))
    xs: List[int] = list(range(0, max(1, W - tile_size + 1), stride))

    # If the last anchor doesn't end at the image edge, append a
    # final anchor that does. This is what guarantees full coverage
    # when the image dimensions aren't an exact multiple of stride.
    if not ys or ys[-1] + tile_size < H:
        ys.append(max(0, H - tile_size))
    if not xs or xs[-1] + tile_size < W:
        xs.append(max(0, W - tile_size))

    return [Tile(y, y + tile_size, x, x + tile_size) for y in ys for x in xs]


def stitch_probability_maps(
    prob_maps: Sequence[np.ndarray],
    tiles: Sequence[Tile],
    H: int,
    W: int,
    mode: str = COMBINE_MODE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stitch per-tile float probability maps into full-image accumulators.

    ``mode="average"`` keeps the original behavior: add probabilities and
    return a coverage count map so the caller can divide by count.

    ``mode="max"`` keeps the maximum probability over all tiles covering
    a pixel and returns a binary coverage map. This is useful for
    overlapping water regions where average stitching can dilute a
    high-confidence tile with a neighboring low-confidence tile and
    create artificial seams.

    Args:
        prob_maps: One float32 ``(valid_h, valid_w)`` array per tile.
                   ``valid_h = min(tile.y1, H) - tile.y0`` etc.
        tiles:     Matching list of :class:`Tile`.
        H, W:      Full image dimensions.

    Returns:
        ``(prob_full, count_full)`` -- two ``(H, W)`` arrays. ``prob_full``
        is float32 (sum of per-tile probabilities), ``count_full`` is
        int32 (number of tiles covering each pixel). Uniform averaging
        is the caller's job: ``avg = np.where(count > 0,
        prob_full / np.maximum(count, 1), 0)``.

    Notes:
        Uniform weight is correct: the count map already compensates
        for the irregular valid-pixel areas at the right/bottom edges.
        Interior pixels are seen by ``overlap_count`` tiles, edge pixels
        by fewer. This is the standard SAHI / YOLOv5-tile convention.
    """
    if len(prob_maps) != len(tiles):
        raise ValueError(
            f"prob_maps ({len(prob_maps)}) and tiles ({len(tiles)}) "
            "must be the same length"
        )

    mode = str(mode).lower()
    if mode not in {"average", "max"}:
        raise ValueError("mode must be one of: average, max")

    prob_full = np.zeros((H, W), dtype=np.float32)
    count_full = np.zeros((H, W), dtype=np.int32)

    for prob, tile in zip(prob_maps, tiles):
        if prob.dtype != np.float32:
            prob = prob.astype(np.float32)
        # Clip the tile bounds to the image. Last-column / last-row tiles
        # may extend past the right/bottom edge of the image.
        y0c = max(0, tile.y0)
        x0c = max(0, tile.x0)
        y1c = min(H, tile.y1)
        x1c = min(W, tile.x1)
        th = y1c - y0c
        tw = x1c - x0c
        if th <= 0 or tw <= 0:
            continue  # fully off-canvas (shouldn't happen with valid tiles)
        if prob.shape != (th, tw):
            # The decoder's prob map must match the valid region. If a
            # caller passes a prob map at the wrong shape we resize
            # defensively rather than crashing.
            import cv2
            prob = cv2.resize(prob, (tw, th), interpolation=cv2.INTER_LINEAR)
        if mode == "max":
            prob_full[y0c:y1c, x0c:x1c] = np.maximum(
                prob_full[y0c:y1c, x0c:x1c],
                prob,
            )
        else:
            prob_full[y0c:y1c, x0c:x1c] += prob
        count_full[y0c:y1c, x0c:x1c] += 1

    if mode == "max":
        count_full = (count_full > 0).astype(np.int32)

    return prob_full, count_full


def make_tile_weight_map(
    h: int,
    w: int,
    mode: str = "hann",
    edge_weight: float = 0.3,
) -> np.ndarray:
    """Build a center-weighted confidence map for a tile.

    ``mode="hann"`` gives the tile center full weight and softly reduces
    border pixels to ``edge_weight``. This reduces false positives caused
    by incomplete context at tile edges while overlap lets neighboring
    tiles vote with higher center confidence.
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"h and w must be > 0, got h={h}, w={w}")
    mode = str(mode).lower()
    if mode not in {"uniform", "hann"}:
        raise ValueError("mode must be one of: uniform, hann")
    edge = float(min(max(edge_weight, 0.0), 1.0))
    if mode == "uniform" or edge >= 1.0:
        return np.ones((h, w), dtype=np.float32)

    if h == 1:
        wy = np.ones((1,), dtype=np.float32)
    else:
        wy = np.hanning(h).astype(np.float32)
    if w == 1:
        wx = np.ones((1,), dtype=np.float32)
    else:
        wx = np.hanning(w).astype(np.float32)

    weight = np.outer(wy, wx).astype(np.float32)
    max_val = float(weight.max())
    if max_val > 0:
        weight /= max_val
    return edge + (1.0 - edge) * weight


def stitch_weighted_probability_maps(
    prob_maps: Sequence[np.ndarray],
    tiles: Sequence[Tile],
    H: int,
    W: int,
    mode: str = "hann",
    edge_weight: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stitch per-tile probability maps with center-weighted averaging.

    Returns weighted probability sums and floating-point weight sums. The
    caller should divide ``prob_full / weight_full`` where weights are
    non-zero. This is intentionally separate from
    :func:`stitch_probability_maps` so the original uniform path remains
    stable for parity tests and comparisons.
    """
    if len(prob_maps) != len(tiles):
        raise ValueError(
            f"prob_maps ({len(prob_maps)}) and tiles ({len(tiles)}) "
            "must be the same length"
        )

    prob_full = np.zeros((H, W), dtype=np.float32)
    weight_full = np.zeros((H, W), dtype=np.float32)

    for prob, tile in zip(prob_maps, tiles):
        if prob.dtype != np.float32:
            prob = prob.astype(np.float32)
        y0c = max(0, tile.y0)
        x0c = max(0, tile.x0)
        y1c = min(H, tile.y1)
        x1c = min(W, tile.x1)
        th = y1c - y0c
        tw = x1c - x0c
        if th <= 0 or tw <= 0:
            continue
        if prob.shape != (th, tw):
            import cv2
            prob = cv2.resize(prob, (tw, th), interpolation=cv2.INTER_LINEAR)

        weights = make_tile_weight_map(
            th,
            tw,
            mode=mode,
            edge_weight=edge_weight,
        )
        prob_full[y0c:y1c, x0c:x1c] += prob * weights
        weight_full[y0c:y1c, x0c:x1c] += weights

    return prob_full, weight_full

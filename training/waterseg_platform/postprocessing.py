"""YOLOv8-seg output decoding in pure NumPy.

This module contains the entire inference-side post-processing pipeline
that ``ultralytics`` normally hides inside ``model.predict()``. Everything
here is implemented with NumPy and OpenCV so the .NET port can replace it
1:1 with C# + OpenCvSharp4.

Pipeline:
    output0 (1, 4+nc+nm, A)        output1 (1, nm, mh, mw)
            |                                  |
            v                                  v
      split                    proto_flat (nm, mh*mw)
      ch[0:4]    = box (cx,cy,w,h) in LETTERBOX pixel coords
      ch[4:4+nc] = class score (ALREADY SIGMOID'D in the ONNX export)
      ch[4+nc:]  = 32 raw mask coefficients
            |                                  |
            v                                  v
      conf threshold (no extra sigmoid)        |
      xywh -> xyxy                              |
            |                                  |
            v                                  v
      class-agnostic NMS                  coeffs @ proto_flat
            |                                  |
            v                                  v
      for each kept detection:            reshape (mh, mw)
            crop to box in proto coords    sigmoid -> in [0, 1]
            cv2.resize to (W_orig, H_orig), INTER_LINEAR
            place at box origin (in original coords)
            threshold at mask_thres
            OR-merge into the final HxW mask
            |
            v
    postprocess_mask (remove_small_components + MORPH_CLOSE) -- from src/

.NET equivalent: ``Postprocessor.cs`` (OpenCvSharp4, pure C# math).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# Reuse the existing postprocess_mask (remove_small_components + MORPH_CLOSE).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.mask_utils import postprocess_mask  # noqa: E402

from waterseg_platform.preprocessing import LetterboxMeta  # noqa: E402


# ---------------------------------------------------------------------- #
# Small math helpers
# ---------------------------------------------------------------------- #

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise sigmoid."""
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def nms_class_agnostic(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_thres: float,
) -> np.ndarray:
    """Class-agnostic Non-Maximum Suppression.

    Args:
        boxes_xyxy: (N, 4) float32 array of [x1, y1, x2, y2] boxes.
        scores:     (N,)   float32 array of per-box scores in [0, 1].
        iou_thres:  IoU threshold; boxes with IoU > thres are suppressed.

    Returns:
        1-D int64 array of indices into ``boxes_xyxy`` that survive NMS,
        sorted by descending score.
    """
    if boxes_xyxy.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[rest] - inter + 1e-7
        iou = inter / union
        order = rest[iou <= iou_thres]
    return np.array(keep, dtype=np.int64)


# ---------------------------------------------------------------------- #
# Shared per-detection placement (used by both single-tile and tiled paths)
# ---------------------------------------------------------------------- #

def _place_masks_on_canvas(
    masks_low: np.ndarray,
    xyxy_letter: np.ndarray,
    letterbox_meta: LetterboxMeta,
    canvas_h: int,
    canvas_w: int,
    mask_box_expand_ratio: float = 0.0,
) -> np.ndarray:
    """Paste per-detection float probability maps onto a ``(canvas_h, canvas_w)`` canvas.

    This is the **only** piece of geometry shared between the single-tile
    path (:func:`decode_segmentation`) and the tiled path
    (:func:`decode_segmentation_probs`). It must therefore return a **float
    probability map** so the caller can either threshold it (single-tile)
    or stitch+threshold it (tiled).

    For each detection:
      1. Crop the proto at the box's 1/4-letterbox region.
      2. ``cv2.resize`` the crop to the box's original-image dims.
      3. Clip to the canvas and paste.

    Multiple detections are merged with ``np.maximum`` (max). No threshold,
    no MORPH_CLOSE, no remove_small_components — those happen in the
    caller, after the (optional) stitch step.

    Args:
        masks_low:      ``(M, mh, mw)`` float32 sigmoid'd per-detection
                        mask in proto coords.
        xyxy_letter:    ``(M, 4)`` float32 boxes in **letterbox** pixel
                        coords.
        letterbox_meta: Geometry of the letterbox that produced the
                        tensor the model ran on.
        canvas_h, canvas_w: Size of the output canvas in original-image
                        pixels. For the single-tile path this equals
                        ``letterbox_meta``'s source size; for the tiled
                        path this is the **tile's valid region** size
                        (``min(tile.y1, H) - tile.y0`` etc.).

    Returns:
        ``(canvas_h, canvas_w)`` float32 ndarray in [0, 1].
    """
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    if xyxy_letter.shape[0] == 0:
        return canvas
    if masks_low.shape[0] != xyxy_letter.shape[0]:
        raise ValueError(
            f"masks_low has {masks_low.shape[0]} rows but xyxy_letter has "
            f"{xyxy_letter.shape[0]} -- internal contract violated"
        )

    imgsz = letterbox_meta.imgsz
    pad_w, pad_h = letterbox_meta.pad_w, letterbox_meta.pad_h
    r = letterbox_meta.r
    if r <= 0:
        return canvas  # safety against a divide-by-zero on empty source

    M, mh, mw = masks_low.shape
    scale_proto = mh / float(imgsz)  # = 176/704 = 0.25
    boxes_for_mask = xyxy_letter.astype(np.float32, copy=True)
    expand_ratio = float(max(mask_box_expand_ratio, 0.0))
    if expand_ratio > 0:
        widths = boxes_for_mask[:, 2] - boxes_for_mask[:, 0]
        heights = boxes_for_mask[:, 3] - boxes_for_mask[:, 1]
        dx = widths * expand_ratio * 0.5
        dy = heights * expand_ratio * 0.5
        boxes_for_mask[:, 0] = np.clip(boxes_for_mask[:, 0] - dx, 0, imgsz)
        boxes_for_mask[:, 1] = np.clip(boxes_for_mask[:, 1] - dy, 0, imgsz)
        boxes_for_mask[:, 2] = np.clip(boxes_for_mask[:, 2] + dx, 0, imgsz)
        boxes_for_mask[:, 3] = np.clip(boxes_for_mask[:, 3] + dy, 0, imgsz)

    xyxy_proto = boxes_for_mask * scale_proto  # (M, 4) in proto coords

    for i in range(M):
        x1p, y1p, x2p, y2p = xyxy_proto[i]
        # Clip to the proto grid.
        x1c = int(max(0, np.floor(x1p)))
        y1c = int(max(0, np.floor(y1p)))
        x2c = int(min(mw, np.ceil(x2p)))
        y2c = int(min(mh, np.ceil(y2p)))
        if x2c <= x1c or y2c <= y1c:
            continue

        m_crop = masks_low[i, y1c:y2c, x1c:x2c]  # (h', w') at proto res

        # Map the letterbox-space box back to original-image coords, then
        # place the mask at that position on the canvas.
        x1_l, y1_l, x2_l, y2_l = boxes_for_mask[i]
        x1_orig = (x1_l - pad_w) / r
        y1_orig = (y1_l - pad_h) / r
        x2_orig = (x2_l - pad_w) / r
        y2_orig = (y2_l - pad_h) / r

        w_box = max(1, int(round(x2_orig - x1_orig)))
        h_box = max(1, int(round(y2_orig - y1_orig)))

        m_full_box = cv2.resize(m_crop, (w_box, h_box), interpolation=cv2.INTER_LINEAR)

        # Clip to canvas bounds and paste.
        x_start = max(0, int(round(x1_orig)))
        y_start = max(0, int(round(y1_orig)))
        x_end = min(canvas_w, x_start + w_box)
        y_end = min(canvas_h, y_start + h_box)
        if x_end <= x_start or y_end <= y_start:
            continue

        # Adjust the mask crop to match the actual pasted area.
        m_w = x_end - x_start
        m_h = y_end - y_start
        if m_w != w_box or m_h != h_box:
            m_full_box = cv2.resize(m_full_box, (m_w, m_h), interpolation=cv2.INTER_LINEAR)

        # np.maximum (max) across detections: within a single tile, two
        # detections are alternative segmentations of the same water,
        # not additive masks. For multi-class this would become per-class
        # max; for nc=1 (water) the current form is correct.
        canvas[y_start:y_end, x_start:x_end] = np.maximum(
            canvas[y_start:y_end, x_start:x_end], m_full_box
        )

    return canvas


# ---------------------------------------------------------------------- #
# Top-level decode
# ---------------------------------------------------------------------- #

def decode_segmentation(
    output0: np.ndarray,
    output1: np.ndarray,
    letterbox_meta: LetterboxMeta,
    original_shape: Tuple[int, int],
    conf: float = 0.25,
    iou: float = 0.5,
    mask_thres: float = 0.5,
    min_area_ratio: float = 0.0005,
    morph_close: bool = True,
    max_det: int = 300,
    nc: int = 1,
    nm: int = 32,
    class_scores_sigmoided: bool = True,
    mask_box_expand_ratio: float = 0.0,
) -> np.ndarray:
    """Decode a YOLOv8-seg raw output pair into a single HxW uint8 binary mask.

    Args:
        output0:        ``(1, 4+nc+nm, A)`` raw logits from the model.
        output1:        ``(1, nm, mh, mw)`` mask prototypes.
        letterbox_meta: Letterbox geometry from :func:`waterseg_platform.preprocessing.letterbox`.
        original_shape: ``(H, W)`` of the source image (pre-letterbox).
        conf:           Class score threshold.
        iou:            NMS IoU threshold.
        mask_thres:     Threshold applied to the per-pixel mask probability.
        min_area_ratio: Connected components with area < ``H*W*ratio`` are removed.
        morph_close:    Apply 3x3 ellipse MORPH_CLOSE.
        max_det:        Cap on the number of detections fed to mask decoding.
        nc, nm:         Class and mask-coefficient counts. The exported ONNX
                        uses nc=1, nm=32.
        class_scores_sigmoided: If True, the class score channels in
                        ``output0`` are already in [0, 1] and are used as-is.
                        Set to False (default) for the standard Ultralytics
                        ONNX export where class scores are raw logits.

    Returns:
        ``(H, W)`` uint8 binary mask with values in {0, 1}.
    """
    H_orig, W_orig = original_shape
    if output0.ndim != 3 or output0.shape[0] != 1:
        raise ValueError(f"output0 must be (1, C, A); got {output0.shape}")
    if output1.ndim != 4 or output1.shape[0] != 1:
        raise ValueError(f"output1 must be (1, nm, mh, mw); got {output1.shape}")

    # ---- 1. Split channels ------------------------------------------- #
    boxes_raw = output0[0, 0:4, :]            # (4, A) cx, cy, w, h (letterbox coords)
    cls_scores_raw = output0[0, 4:4 + nc, :]  # (nc, A)
    coeffs_raw = output0[0, 4 + nc:4 + nc + nm, :]  # (nm, A)

    # ---- 2. Sigmoid on class scores ---------------------------------- #
    if class_scores_sigmoided:
        scores = cls_scores_raw.astype(np.float32)  # already in [0, 1]
    else:
        scores = sigmoid(cls_scores_raw)

    # For single-class, scores is (1, A). For multi-class, take max over
    # foreground classes only (skip class 0 = background).
    if nc == 1:
        scores_flat = scores[0]  # (A,)
    else:
        scores_flat = scores[1:].max(axis=0)  # (A,) — foreground only

    # ---- 3. Conf threshold ------------------------------------------- #
    keep = scores_flat > conf
    if not keep.any():
        return np.zeros((H_orig, W_orig), dtype=np.uint8)

    boxes_raw = boxes_raw[:, keep]              # (4, K)
    scores_kept = scores_flat[keep]             # (K,)
    coeffs_kept = coeffs_raw[:, keep]            # (nm, K)

    # ---- 4. xywh -> xyxy (letterbox coords) -------------------------- #
    cx, cy, w, h = boxes_raw
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    xyxy_letter = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)  # (K, 4)

    # ---- 5. NMS ------------------------------------------------------- #
    keep_idx = nms_class_agnostic(xyxy_letter, scores_kept.astype(np.float32), iou)
    if keep_idx.size == 0:
        return np.zeros((H_orig, W_orig), dtype=np.uint8)
    if keep_idx.size > max_det:
        keep_idx = keep_idx[:max_det]

    xyxy_letter = xyxy_letter[keep_idx]    # (M, 4)
    coeffs_kept = coeffs_kept.T[keep_idx]  # (M, nm)

    # ---- 6. Decode masks via proto matmul ---------------------------- #
    proto = output1[0]  # (nm, mh, mw)
    nm_actual, mh, mw = proto.shape
    if nm_actual != nm:
        raise ValueError(
            f"output1 has nm={nm_actual}, but config says nm={nm}. "
            "Either re-export the ONNX or update nm in the config."
        )
    proto_flat = proto.reshape(nm_actual, mh * mw)  # (nm, mh*mw)
    masks_low = coeffs_kept @ proto_flat             # (M, mh*mw)
    masks_low = sigmoid(masks_low).reshape(-1, mh, mw)  # (M, mh, mw) in [0, 1]

    # ---- 7. Per-detection crop + resize + paste (shared helper) ----- #
    # `_place_masks_on_canvas` returns a float32 probability map; the
    # tiling path needs floats so it can stitch+threshold instead of
    # thresholding per-tile. The single-tile path thresholds here.
    # All foreground classes (flooded_road + water) are merged into a
    # single "waterlogging" mask via np.maximum in _place_masks_on_canvas.
    prob_canvas = _place_masks_on_canvas(
        masks_low,
        xyxy_letter,
        letterbox_meta,
        H_orig,
        W_orig,
        mask_box_expand_ratio=mask_box_expand_ratio,
    )
    binary = (prob_canvas >= mask_thres).astype(np.uint8)

    # ---- 8. Postprocess (remove small components + MORPH_CLOSE) ------ #
    pred = postprocess_mask(
        binary, min_area_ratio=min_area_ratio, morph_close=morph_close
    )
    return pred


def decode_segmentation_probs(
    output0: np.ndarray,
    output1: np.ndarray,
    letterbox_meta: LetterboxMeta,
    original_shape: Tuple[int, int],
    conf: float = 0.25,
    iou: float = 0.5,
    max_det: int = 300,
    nc: int = 1,
    nm: int = 32,
    class_scores_sigmoided: bool = True,
    mask_box_expand_ratio: float = 0.0,
    target_class_ids: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Decode a YOLOv8-seg raw output pair into a float probability map.

    Unlike :func:`decode_segmentation` (which thresholds + postprocesses
    and returns a uint8 binary mask), this function returns the raw
    per-pixel probability in ``[0, 1]`` as a float32 array. The tiling
    pipeline uses this so it can stitch per-tile probability maps with
    uniform averaging, then threshold the stitched result once.

    Args:
        output0, output1, letterbox_meta, original_shape, conf, iou,
            max_det, nc, nm, class_scores_sigmoided: Same as
            :func:`decode_segmentation`. Note that ``mask_thres``,
            ``min_area_ratio`` and ``morph_close`` are **not** accepted
            here -- they are applied by the caller after stitching.

    Returns:
        ``(H, W)`` float32 ndarray of per-pixel water probability in
        ``[0, 1]``. ``H, W`` are ``original_shape``. Empty if no
        detections survive the conf threshold / NMS.
    """
    H_orig, W_orig = original_shape
    if output0.ndim != 3 or output0.shape[0] != 1:
        raise ValueError(f"output0 must be (1, C, A); got {output0.shape}")
    if output1.ndim != 4 or output1.shape[0] != 1:
        raise ValueError(f"output1 must be (1, nm, mh, mw); got {output1.shape}")

    # ---- 1. Split channels ------------------------------------------- #
    boxes_raw = output0[0, 0:4, :]            # (4, A) cx, cy, w, h (letterbox coords)
    cls_scores_raw = output0[0, 4:4 + nc, :]  # (nc, A)
    coeffs_raw = output0[0, 4 + nc:4 + nc + nm, :]  # (nm, A)

    # ---- 2. Sigmoid on class scores ---------------------------------- #
    if class_scores_sigmoided:
        scores = cls_scores_raw.astype(np.float32)  # already in [0, 1]
    else:
        scores = sigmoid(cls_scores_raw)

    if target_class_ids:
        target_ids = np.asarray(list(target_class_ids), dtype=np.int64)
        if target_ids.min() < 0 or target_ids.max() >= nc:
            raise ValueError(f"target_class_ids={list(target_class_ids)} outside [0, {nc - 1}]")
        scores_flat = scores[target_ids].max(axis=0)
    elif nc == 1:
        scores_flat = scores[0]
    else:
        scores_flat = scores[1:].max(axis=0)  # foreground classes only

    # ---- 3. Conf threshold ------------------------------------------- #
    keep = scores_flat > conf
    if not keep.any():
        return np.zeros((H_orig, W_orig), dtype=np.float32)

    boxes_raw = boxes_raw[:, keep]
    scores_kept = scores_flat[keep]
    coeffs_kept = coeffs_raw[:, keep]

    # ---- 4. xywh -> xyxy (letterbox coords) -------------------------- #
    cx, cy, w, h = boxes_raw
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    xyxy_letter = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    # ---- 5. NMS ------------------------------------------------------- #
    keep_idx = nms_class_agnostic(xyxy_letter, scores_kept.astype(np.float32), iou)
    if keep_idx.size == 0:
        return np.zeros((H_orig, W_orig), dtype=np.float32)
    if keep_idx.size > max_det:
        keep_idx = keep_idx[:max_det]

    xyxy_letter = xyxy_letter[keep_idx]
    coeffs_kept = coeffs_kept.T[keep_idx]

    # ---- 6. Decode masks via proto matmul ---------------------------- #
    proto = output1[0]  # (nm, mh, mw)
    nm_actual, mh, mw = proto.shape
    if nm_actual != nm:
        raise ValueError(
            f"output1 has nm={nm_actual}, but config says nm={nm}. "
            "Either re-export the ONNX or update nm in the config."
        )
    proto_flat = proto.reshape(nm_actual, mh * mw)
    masks_low = coeffs_kept @ proto_flat
    masks_low = sigmoid(masks_low).reshape(-1, mh, mw)

    # ---- 7. Place on canvas (no threshold, no postprocess) ----------- #
    return _place_masks_on_canvas(
        masks_low,
        xyxy_letter,
        letterbox_meta,
        H_orig,
        W_orig,
        mask_box_expand_ratio=mask_box_expand_ratio,
    )

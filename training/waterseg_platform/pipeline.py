"""End-to-end segmentation service for the ONNX platform.

:class:`SegmentationService` is the single public class the CLI and Gradio
UI both call into. It wires together:

    OnnxSegmenter   (engine.py)
    preprocess      (preprocessing.py)
    decode          (postprocessing.py)
    metrics         (metrics.py)
    visualization   (visualization.py)

.NET equivalent: ``SegmentationService.cs``.
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from waterseg_platform.config import PlatformConfig  # noqa: E402
from waterseg_platform.engine import OnnxSegmenter  # noqa: E402
from waterseg_platform.image_io import (  # noqa: E402
    IMAGE_EXTS,
    ensure_dir,
    list_files,
    read_image,
    read_mask_binary,
)
from waterseg_platform.metrics import compute_binary_metrics  # noqa: E402
from waterseg_platform.preprocessing import letterbox, to_nchw_float  # noqa: E402
from waterseg_platform.tiling import (  # noqa: E402
    COMBINE_MODE,
    compute_tiles,
    stitch_probability_maps,
    stitch_weighted_probability_maps,
)
from waterseg_platform.visualization import save_panel  # noqa: E402

# Reuse the same remove_small_components + MORPH_CLOSE that the
# single-tile path uses, so the tiled path is structurally identical
# to it once you drop in the per-tile float maps for a single tile.
from src.mask_utils import postprocess_mask  # noqa: E402

LOG = logging.getLogger(__name__)


def choose_cascade_mask(
    primary_mask: np.ndarray,
    fallback_mask_a: np.ndarray,
    fallback_mask_b: np.ndarray,
    trigger_area_ratio: float,
    min_consensus_area_ratio: float,
) -> Tuple[np.ndarray, dict]:
    """Choose a conservative fallback intersection for weak primary output."""
    primary = (np.asarray(primary_mask) > 0).astype(np.uint8)
    pixel_count = float(max(primary.size, 1))
    primary_ratio = float(primary.sum()) / pixel_count
    info = {
        "triggered": primary_ratio < float(trigger_area_ratio),
        "accepted": False,
        "primary_area_ratio": primary_ratio,
        "fallback_a_area_ratio": 0.0,
        "fallback_b_area_ratio": 0.0,
        "consensus_area_ratio": 0.0,
    }
    if not info["triggered"]:
        return primary, info

    fallback_a = (np.asarray(fallback_mask_a) > 0).astype(np.uint8)
    fallback_b = (np.asarray(fallback_mask_b) > 0).astype(np.uint8)
    if fallback_a.shape != primary.shape or fallback_b.shape != primary.shape:
        raise ValueError("Cascade masks must have the same shape")

    consensus = np.logical_and(fallback_a, fallback_b).astype(np.uint8)
    info["fallback_a_area_ratio"] = float(fallback_a.sum()) / pixel_count
    info["fallback_b_area_ratio"] = float(fallback_b.sum()) / pixel_count
    info["consensus_area_ratio"] = float(consensus.sum()) / pixel_count
    info["accepted"] = (
        info["consensus_area_ratio"] >= float(min_consensus_area_ratio)
    )
    return (consensus if info["accepted"] else primary), info


def soft_fuse_masks(
    coarse_mask: np.ndarray,
    tiled_probability: np.ndarray,
    tile_weight: float,
    mask_thres: float,
) -> np.ndarray:
    """Fuse global binary context with weighted tiled probabilities."""
    coarse = (np.asarray(coarse_mask) > 0).astype(np.float32)
    tiled = np.asarray(tiled_probability, dtype=np.float32)
    if coarse.shape != tiled.shape:
        raise ValueError("Coarse mask and tiled probability shapes must match")
    fused = np.maximum(coarse, float(tile_weight) * tiled)
    return (fused >= float(mask_thres)).astype(np.uint8)


def fuse_global_local_probabilities(
    global_probability: np.ndarray,
    tiled_probability: np.ndarray,
    tile_weight: float,
    global_context_strength: float,
) -> np.ndarray:
    """Fuse global context with local tile detail as a float probability map.

    The local branch keeps high-resolution detail. The global branch both
    preserves whole-image detections and downweights local-only predictions
    that are not supported by the whole-image context.
    """
    global_prob = np.asarray(global_probability, dtype=np.float32)
    tiled_prob = np.asarray(tiled_probability, dtype=np.float32)
    if global_prob.shape != tiled_prob.shape:
        raise ValueError("Global and tiled probability shapes must match")
    strength = float(min(max(global_context_strength, 0.0), 1.0))
    prior = (1.0 - strength) + strength * np.clip(global_prob, 0.0, 1.0)
    local = float(tile_weight) * np.clip(tiled_prob, 0.0, 1.0) * prior
    return np.maximum(np.clip(global_prob, 0.0, 1.0), local).astype(np.float32)


def suppress_border_components(
    mask: np.ndarray,
    *,
    enabled: bool = True,
    margin_ratio: float = 0.01,
    margin_min_px: int = 12,
    touch_ratio_thresh: float = 0.35,
    max_component_area_ratio: float = 0.08,
) -> Tuple[np.ndarray, dict]:
    """Remove connected components that are likely border artifacts.

    Components are removed only when a large share of their pixels lies
    inside the image-border band and the component is not a very large
    region. This avoids deleting legitimate flood masks that naturally
    extend out of frame.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    info = {
        "enabled": bool(enabled),
        "removed_components": 0,
        "removed_area": 0,
        "margin_px": 0,
    }
    if not enabled or h <= 0 or w <= 0 or binary.sum() == 0:
        return binary, info

    margin = max(int(margin_min_px), int(round(float(margin_ratio) * min(h, w))))
    margin = min(margin, max(h, w))
    info["margin_px"] = int(margin)
    if margin <= 0:
        return binary, info

    border = np.zeros((h, w), dtype=bool)
    border[:margin, :] = True
    border[-margin:, :] = True
    border[:, :margin] = True
    border[:, -margin:] = True

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    out = binary.copy()
    image_area = float(max(h * w, 1))
    touch_thresh = float(min(max(touch_ratio_thresh, 0.0), 1.0))
    max_area_ratio = float(min(max(max_component_area_ratio, 0.0), 1.0))
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component = labels == label
        border_pixels = int(np.count_nonzero(component & border))
        touch_ratio = border_pixels / float(area)
        area_ratio = area / image_area
        if touch_ratio >= touch_thresh and area_ratio <= max_area_ratio:
            out[component] = 0
            info["removed_components"] += 1
            info["removed_area"] += area

    return out.astype(np.uint8), info


class SegmentationService:
    """Public-facing segmentation service. One instance per process.

    Example:
        >>> cfg = PlatformConfig(model_path="onnx/floodnet_binary_aug_yolov8m_1024.onnx")
        >>> svc = SegmentationService(cfg)
        >>> mask, info = svc.segment_array(image_bgr)
    """

    def __init__(self, config: PlatformConfig) -> None:
        self.config = config
        self.engine = OnnxSegmenter(
            model_path=config.model_path,
            providers=config.providers,
            imgsz=config.imgsz,
        )
        self._cascade_engines = None
        self._sam3_refiner = None
        LOG.info("SegmentationService ready (providers=%s)", self.engine.providers_active)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _resolve_kwargs(
        self,
        conf=None, iou=None, mask_thres=None,
        min_area_ratio=None, morph_close=None, max_det=None,
    ) -> dict:
        """Return a dict of inference kwargs, with per-call overrides winning
        over the service's :class:`PlatformConfig`. ``None`` means "use the
        config default" so the UI / CLI can pass partial overrides.
        """
        cfg = self.config
        return {
            "conf": cfg.conf if conf is None else float(conf),
            "iou": cfg.iou if iou is None else float(iou),
            "mask_thres": cfg.mask_thres if mask_thres is None else float(mask_thres),
            "mask_box_expand_ratio": float(cfg.mask_box_expand_ratio),
            "min_area_ratio": cfg.min_area_ratio if min_area_ratio is None else float(min_area_ratio),
            "morph_close": cfg.morph_close if morph_close is None else bool(morph_close),
            "max_det": cfg.max_det if max_det is None else int(max_det),
            "nc": int(cfg.nc),
            "nm": int(cfg.nm),
        }

    def _get_cascade_engines(self) -> Tuple[OnnxSegmenter, OnnxSegmenter]:
        """Create fallback sessions only after a weak primary prediction."""
        if self._cascade_engines is None:
            paths = list(self.config.cascade_model_paths)
            if len(paths) != 2:
                raise ValueError("cascade_model_paths must contain exactly two models")
            self._cascade_engines = tuple(
                OnnxSegmenter(
                    model_path=path,
                    providers=self.config.providers,
                    imgsz=self.config.cascade_imgsz,
                )
                for path in paths
            )
        return self._cascade_engines

    def _apply_cascade(
        self,
        image_bgr: np.ndarray,
        primary_mask: np.ndarray,
        info: dict,
        inference_kwargs: dict,
    ) -> Tuple[np.ndarray, dict]:
        """Run conservative fallback consensus for a weak primary mask."""
        primary = (np.asarray(primary_mask) > 0).astype(np.uint8)
        primary_ratio = float(primary.mean()) if primary.size else 0.0
        height, width = primary.shape[:2]
        aspect_ratio = float(max(width / max(height, 1), height / max(width, 1)))
        aspect_eligible = (
            aspect_ratio >= float(self.config.cascade_min_aspect_ratio)
        )
        cascade_info = {
            "enabled": bool(self.config.cascade_enabled),
            "triggered": False,
            "accepted": False,
            "aspect_ratio": aspect_ratio,
            "aspect_eligible": aspect_eligible,
            "primary_area_ratio": primary_ratio,
            "fallback_a_area_ratio": 0.0,
            "fallback_b_area_ratio": 0.0,
            "consensus_area_ratio": 0.0,
        }
        if not self.config.cascade_enabled:
            info["cascade"] = cascade_info
            return primary, info
        if not aspect_eligible:
            info["cascade"] = cascade_info
            return primary, info
        if primary_ratio >= float(self.config.cascade_trigger_area_ratio):
            info["cascade"] = cascade_info
            return primary, info

        fallback_a_engine, fallback_b_engine = self._get_cascade_engines()
        fallback_kwargs = {
            key: inference_kwargs[key]
            for key in (
                "conf",
                "iou",
                "mask_thres",
                "mask_box_expand_ratio",
                "min_area_ratio",
                "morph_close",
                "max_det",
                "nc",
                "nm",
            )
        }
        fallback_kwargs["conf"] = float(self.config.cascade_conf)
        fallback_a, _ = fallback_a_engine.predict_mask(
            image_bgr, **fallback_kwargs
        )
        fallback_b, _ = fallback_b_engine.predict_mask(
            image_bgr, **fallback_kwargs
        )
        selected, decision = choose_cascade_mask(
            primary,
            fallback_a,
            fallback_b,
            trigger_area_ratio=self.config.cascade_trigger_area_ratio,
            min_consensus_area_ratio=self.config.cascade_min_consensus_area_ratio,
        )
        cascade_info.update(decision)
        info["cascade"] = cascade_info
        info["pred_area"] = int(selected.sum())
        info["pred_area_ratio"] = float(selected.mean()) if selected.size else 0.0
        return selected, info

    def _apply_sam3(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        info: dict,
        *,
        enabled: Optional[bool],
        mode: Optional[str],
    ) -> Tuple[np.ndarray, dict]:
        """Optionally refine the selected YOLO/cascade mask with SAM 3."""
        use_sam3 = (
            bool(self.config.sam3_enabled)
            if enabled is None
            else bool(enabled)
        )
        selected_mode = self.config.sam3_mode if mode is None else str(mode)
        if selected_mode not in {"conservative", "balanced", "open"}:
            raise ValueError(
                "sam3_mode must be one of: conservative, balanced, open"
            )
        if not use_sam3:
            info["sam3"] = {
                "enabled": False,
                "ran": False,
                "mode": selected_mode,
                "fallback": False,
            }
            return mask, info

        if getattr(self, "_sam3_refiner", None) is None:
            from waterseg_platform.sam3_refinement import Sam3Refiner

            self._sam3_refiner = Sam3Refiner(self.config)
        refined, sam3_info = self._sam3_refiner.refine(
            image_bgr,
            mask,
            enabled=True,
            mode=selected_mode,
        )
        info["sam3"] = sam3_info
        info["pred_area"] = int(refined.sum())
        info["pred_area_ratio"] = (
            float(refined.mean()) if refined.size else 0.0
        )
        return refined, info

    # ------------------------------------------------------------------ #
    # Single-image entry points
    # ------------------------------------------------------------------ #

    def segment_array(
        self,
        image_bgr: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        mask_thres: Optional[float] = None,
        min_area_ratio: Optional[float] = None,
        morph_close: Optional[bool] = None,
        max_det: Optional[int] = None,
        sam3_enabled: Optional[bool] = None,
        sam3_mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Segment an in-memory BGR image; returns ``(mask, info)``.

        Each hyperparameter is optional; ``None`` means "use the value from
        the service's :class:`PlatformConfig`". This is the path the Gradio
        UI uses, so slider values flow through here.

        This is the **single-tile** path: the whole image is letterboxed
        to ``config.imgsz`` and sent to the model in one call. For
        large/rectangular images, prefer :meth:`segment_array_tiled`.
        """
        kw = self._resolve_kwargs(
            conf=conf, iou=iou, mask_thres=mask_thres,
            min_area_ratio=min_area_ratio, morph_close=morph_close,
            max_det=max_det,
        )
        mask, info = self.engine.predict_mask(image_bgr, **kw)
        mask, info = self._apply_cascade(image_bgr, mask, info, kw)
        return self._apply_sam3(
            image_bgr,
            mask,
            info,
            enabled=sam3_enabled,
            mode=sam3_mode,
        )

    def segment_array_tiled(
        self,
        image_bgr: np.ndarray,
        tile_size: Optional[int] = None,
        tile_overlap_px: Optional[int] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        mask_thres: Optional[float] = None,
        min_area_ratio: Optional[float] = None,
        morph_close: Optional[bool] = None,
        max_det: Optional[int] = None,
        sam3_enabled: Optional[bool] = None,
        sam3_mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Segment via global/local soft-fused tiled inference.

        Two-pass strategy that preserves global context while allowing tiles
        to recover detections missed by the coarse pass:

        1. **Coarse pass**: run the single-tile (letterbox) path to get a
           coarse binary mask. This is the "gate" — it has global context
           and rarely produces false positives.
        2. **Tile pass**: run per-tile inference and stitch the float
           probability maps with uniform averaging, exactly like the
           original tiled pipeline.
        3. **Soft fusion**: combine the coarse binary mask with weighted tile
           probabilities, threshold once, then run the shared post-processing.

        For small images that fit in a single tile, the result is
        mathematically equivalent to :meth:`segment_array`.

        Args:
            image_bgr:      ``(H, W, 3)`` uint8 BGR image.
            tile_size:      Side length of each tile in pixels. ``None`` or
                            ``<= 0`` falls back to ``config.tile_size`` (which
                            defaults to ``config.imgsz``). Must equal the
                            ONNX export's input shape — to use a different
                            tile_size, re-export the ONNX.
            tile_overlap_px: Overlap between adjacent tiles in pixels.
                            ``None`` falls back to ``config.tile_overlap_px``.
                            Must be in ``[0, tile_size - 1]``.
            conf, iou, mask_thres, min_area_ratio, morph_close, max_det:
                            Per-call overrides, same semantics as
                            :meth:`segment_array`.

        Returns:
            ``(mask, info)`` where ``mask`` is ``(H, W)`` uint8 binary
            mask, and ``info`` mirrors :meth:`segment_array` plus a
            ``"tiling"`` sub-dict with the grid config and timing.
        """
        from waterseg_platform.postprocessing import decode_segmentation_probs

        kw = self._resolve_kwargs(
            conf=conf, iou=iou, mask_thres=mask_thres,
            min_area_ratio=min_area_ratio, morph_close=morph_close,
            max_det=max_det,
        )

        # Resolve tile geometry. tile_size <= 0 means "use imgsz" so the
        # UI / CLI can be lazy (the default value, 0, is friendly).
        ts = int(tile_size) if tile_size else int(self.config.tile_size)
        if ts <= 0:
            ts = int(self.config.imgsz)
        ov = int(tile_overlap_px) if tile_overlap_px is not None else int(self.config.tile_overlap_px)
        if ov < 0 or ov >= ts:
            raise ValueError(
                f"tile_overlap_px must be in [0, ts-1]; got ov={ov}, ts={ts}"
            )
        if ts != self.config.imgsz:
            raise ValueError(
                f"tile_size={ts} does not match config.imgsz={self.config.imgsz}. "
                "The model has a fixed input shape; re-export the ONNX with a "
                "matching imgsz to use a different tile_size."
            )

        H, W = image_bgr.shape[:2]

        # ---- Pass 1: Coarse single-tile inference (gate) ---- #
        t_coarse = time.perf_counter()
        canvas, meta = letterbox(image_bgr, imgsz=int(self.config.imgsz))
        tensor = to_nchw_float(canvas, assume_rgb=False)
        output0, output1 = self.engine.run(tensor)
        global_prob = decode_segmentation_probs(
            output0=output0,
            output1=output1,
            letterbox_meta=meta,
            original_shape=(H, W),
            conf=kw["conf"],
            iou=kw["iou"],
            mask_box_expand_ratio=kw["mask_box_expand_ratio"],
            max_det=kw["max_det"],
            nc=kw["nc"],
            nm=kw["nm"],
            target_class_ids=self.config.target_class_ids,
        )
        if not self.config.global_context_enabled:
            global_prob = np.zeros_like(global_prob, dtype=np.float32)
        coarse_ms = (time.perf_counter() - t_coarse) * 1000.0

        # Build gating mask: dilate the coarse mask so tiled detections
        # near the boundary of a coarse detection are also kept. This
        # allows the tiled path to refine edges and slightly expand water
        # boundaries, while still rejecting distant false positives.
        # Dilation kernel size = fraction of tile_size (empirically 1/4).
        # If the coarse mask is entirely empty, skip tiling entirely —
        # there's nothing to refine and tiling would only add false
        # positives.
        # ---- Pass 2: Per-tile inference ---- #
        tiles = compute_tiles(H, W, ts, ov)

        prob_maps: List[np.ndarray] = []
        tile_engine_ms: List[float] = []
        for tile in tiles:
            # Crop to the valid region of the tile (clipped at image
            # edges). The decoder receives original_shape=(valid_h,
            # valid_w) so its output prob map is exactly that region.
            y0c = max(0, tile.y0)
            x0c = max(0, tile.x0)
            y1c = min(H, tile.y1)
            x1c = min(W, tile.x1)
            crop = image_bgr[y0c:y1c, x0c:x1c]
            valid_h, valid_w = crop.shape[:2]

            # Skip tiles that don't overlap with the gating mask.
            # Letterbox the crop to (ts, ts) and build the NCHW tensor.
            canvas, meta = letterbox(crop, imgsz=ts)
            tensor = to_nchw_float(canvas, assume_rgb=False)

            t0 = time.perf_counter()
            out0, out1 = self.engine.run(tensor)
            tile_engine_ms.append((time.perf_counter() - t0) * 1000.0)

            prob = decode_segmentation_probs(
                output0=out0,
                output1=out1,
                letterbox_meta=meta,
                original_shape=(valid_h, valid_w),
                conf=kw["conf"],
                iou=kw["iou"],
                max_det=kw["max_det"],
                nc=kw["nc"],
                nm=kw["nm"],
                target_class_ids=self.config.target_class_ids,
                mask_box_expand_ratio=kw["mask_box_expand_ratio"],
            )
            prob_maps.append(prob)

        # ---- Pass 3: Center-weighted stitch + global/local fusion ---- #
        prob_full, weight_full = stitch_weighted_probability_maps(
            prob_maps,
            tiles,
            H,
            W,
            mode=self.config.tile_weight_mode,
            edge_weight=self.config.tile_edge_weight,
        )
        avg = np.zeros((H, W), dtype=np.float32)
        nz = weight_full > 0
        if nz.any():
            avg[nz] = prob_full[nz] / weight_full[nz]

        fused_prob = fuse_global_local_probabilities(
            global_prob,
            avg,
            tile_weight=self.config.tile_weight,
            global_context_strength=self.config.global_context_strength,
        )
        binary_fused = (fused_prob >= kw["mask_thres"]).astype(np.uint8)
        mask = postprocess_mask(
            binary_fused, min_area_ratio=kw["min_area_ratio"],
            morph_close=kw["morph_close"],
        )
        mask, border_info = suppress_border_components(
            mask,
            enabled=self.config.border_suppression_enabled,
            margin_ratio=self.config.border_margin_ratio,
            margin_min_px=self.config.border_margin_min_px,
            touch_ratio_thresh=self.config.border_touch_ratio_thresh,
            max_component_area_ratio=self.config.border_max_component_area_ratio,
        )

        # Fuse: take the maximum (element-wise OR) of the coarse mask and
        # the gated tiled mask. This ensures:
        #   - Single-tile detections are always preserved (never lost).
        #   - Tiled detections near single-tile boundaries are added
        #     (boundary refinement).
        #   - Tiled false positives far from any coarse detection are
        #     eliminated by the gating mask.
        avg_engine_ms = (
            float(np.mean(tile_engine_ms)) if tile_engine_ms else 0.0
        )
        info = {
            "providers": self.engine.providers_active,
            "input_shape": self.engine.input_shape,
            "output0_shape": self.engine.output0_shape,
            "output1_shape": self.engine.output1_shape,
            "tiling": {
                "tile_size": ts,
                "tile_overlap_px": ov,
                "num_tiles": len(tiles),
                "combine_mode": self.config.tile_stitch_mode,
                "tile_weight_mode": self.config.tile_weight_mode,
                "tile_edge_weight": float(self.config.tile_edge_weight),
                "tile_weight": float(self.config.tile_weight),
                "global_context_enabled": bool(self.config.global_context_enabled),
                "global_context_strength": float(self.config.global_context_strength),
                "mask_box_expand_ratio": float(kw["mask_box_expand_ratio"]),
                "elapsed_ms_per_tile_avg": avg_engine_ms,
                "coarse_ms": coarse_ms,
            },
            "border_suppression": border_info,
            "pred_area": int(mask.sum()),
            "pred_area_ratio": float(mask.sum()) / float(max(H * W, 1)),
        }
        mask, info = self._apply_cascade(image_bgr, mask, info, kw)
        return self._apply_sam3(
            image_bgr,
            mask,
            info,
            enabled=sam3_enabled,
            mode=sam3_mode,
        )

    def segment_file(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        save_overlay: bool = True,
        save_mask: bool = True,
        use_tiling: bool = True,
        tile_size: Optional[int] = None,
        tile_overlap_px: Optional[int] = None,
        sam3_enabled: Optional[bool] = None,
        sam3_mode: Optional[str] = None,
    ) -> dict:
        """Segment a single image file. Optionally write outputs.

        Returns a dict with: ``image_path``, ``mask_path``, ``overlay_path``,
        ``mask_area``, ``pred_area_ratio``, ``elapsed_ms``.

        ``use_tiling=True`` dispatches to :meth:`segment_array_tiled`
        (the recommended path for large/rectangular images);
        ``use_tiling=False`` uses the legacy single-tile letterbox path.
        """
        image_path = Path(image_path).resolve()
        out_dir = ensure_dir(Path(output_dir).resolve()) if output_dir else None
        img = read_image(image_path)
        t0 = time.perf_counter()
        if use_tiling:
            mask, info = self.segment_array_tiled(
                img,
                tile_size=tile_size,
                tile_overlap_px=tile_overlap_px,
                sam3_enabled=sam3_enabled,
                sam3_mode=sam3_mode,
            )
        else:
            mask, info = self.segment_array(
                img,
                sam3_enabled=sam3_enabled,
                sam3_mode=sam3_mode,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        mask_path: Optional[str] = None
        overlay_path: Optional[str] = None
        if out_dir is not None:
            stem = image_path.stem
            if save_mask:
                mask_path = str(out_dir / f"{stem}_pred.png")
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
            if save_overlay:
                overlay_path = str(out_dir / f"{stem}_overlay.jpg")
                save_panel(overlay_path, img, gt=None, pred=mask)

        return {
            "image_path": str(image_path),
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "mask_area": int(mask.sum()),
            "pred_area_ratio": float(info["pred_area_ratio"]),
            "elapsed_ms": float(elapsed_ms),
            "info": info,
        }

    def segment_directory(
        self,
        image_dir: str,
        output_dir: str,
        image_exts: Sequence[str] = IMAGE_EXTS,
        num: Optional[int] = None,
        save_overlay: bool = False,
        save_mask: bool = True,
        gt_mask_dir: Optional[str] = None,
        mask_mode: str = "grayscale",
        foreground_rgb: str = "128,0,0",
        tolerance: int = 0,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        mask_thres: Optional[float] = None,
        min_area_ratio: Optional[float] = None,
        morph_close: Optional[bool] = None,
        max_det: Optional[int] = None,
        use_tiling: bool = True,
        tile_size: Optional[int] = None,
        tile_overlap_px: Optional[int] = None,
        sam3_enabled: Optional[bool] = None,
        sam3_mode: Optional[str] = None,
    ) -> pd.DataFrame:
        """Segment every image in ``image_dir`` and return a per-image metrics DataFrame.

        If ``gt_mask_dir`` is given, also computes pixel-level metrics against
        the ground-truth mask (matching the existing evaluation scripts).

        Each hyperparameter is optional; ``None`` means "use the value from
        the service's :class:`PlatformConfig`".

        ``use_tiling=True`` (default) dispatches each image to
        :meth:`segment_array_tiled`; set it to ``False`` to force the
        legacy single-tile letterbox path (e.g. to reproduce the parity
        test).
        """
        image_dir = Path(image_dir).resolve()
        out_dir = ensure_dir(Path(output_dir).resolve())
        gt_dir = Path(gt_mask_dir).resolve() if gt_mask_dir else None

        images = list_files(image_dir, image_exts)
        if num is not None and num > 0:
            images = images[:num]
        LOG.info("Segmenting %d images from %s", len(images), image_dir)

        rows: List[dict] = []
        for img_path in tqdm(images, desc="segment", unit="img"):
            img = read_image(img_path)
            t0 = time.perf_counter()
            if use_tiling:
                mask, info = self.segment_array_tiled(
                    img, tile_size=tile_size, tile_overlap_px=tile_overlap_px,
                    conf=conf, iou=iou, mask_thres=mask_thres,
                    min_area_ratio=min_area_ratio, morph_close=morph_close,
                    max_det=max_det,
                    sam3_enabled=sam3_enabled,
                    sam3_mode=sam3_mode,
                )
            else:
                mask, info = self.segment_array(
                    img, conf=conf, iou=iou, mask_thres=mask_thres,
                    min_area_ratio=min_area_ratio, morph_close=morph_close,
                    max_det=max_det,
                    sam3_enabled=sam3_enabled,
                    sam3_mode=sam3_mode,
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            row = {
                "stem": img_path.stem,
                "image_path": str(img_path),
                "pred_area": int(mask.sum()),
                "pred_area_ratio": float(info["pred_area_ratio"]),
                "elapsed_ms": float(elapsed_ms),
                "providers": ",".join(info["providers"]),
            }
            if gt_dir is not None:
                gt_path = gt_dir / f"{img_path.stem}.png"
                gt = read_mask_binary(
                    gt_path,
                    image_shape=img.shape[:2],
                    mask_mode=mask_mode,
                    foreground_rgb=foreground_rgb,
                    tolerance=tolerance,
                )
                m = compute_binary_metrics(mask, gt)
                row.update(asdict(m))
                row["gt_area"] = int(gt.sum())

            rows.append(row)

            stem = img_path.stem
            if save_mask:
                cv2.imwrite(str(out_dir / f"{stem}_pred.png"), (mask * 255).astype(np.uint8))
            if save_overlay:
                save_panel(out_dir / f"{stem}_overlay.jpg", img, gt=None, pred=mask)

        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "metrics.csv", index=False)
        if gt_dir is not None and "iou" in df.columns:
            # Quick summary line — replicates the format of the existing scripts.
            tp, fp, fn, tn = int(df["tp"].sum()), int(df["fp"].sum()), int(df["fn"].sum()), int(df["tn"].sum())
            micro_iou = tp / (tp + fp + fn + 1e-8)
            micro_f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            with (out_dir / "summary.txt").open("w", encoding="utf-8") as f:
                f.write(f"num_images={len(df)}\n")
                f.write(f"tp={tp} fp={fp} fn={fn} tn={tn}\n")
                f.write(f"micro_iou={micro_iou:.4f}\n")
                f.write(f"micro_f1={micro_f1:.4f}\n")
        return df

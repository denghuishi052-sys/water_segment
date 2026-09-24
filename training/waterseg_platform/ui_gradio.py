"""Gradio web UI for the ONNX water-segmentation platform."""
from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

# Module-level singleton so we can avoid passing the SegmentationService
# through gr.State (Gradio tries to deep-copy State values, and the
# underlying onnxruntime InferenceSession is not deepcopy-able).
_SERVICE_SINGLETON: Optional[object] = None


def _get_service(default_config=None):
    """Return the cached :class:`SegmentationService` (created lazily)."""
    global _SERVICE_SINGLETON
    if _SERVICE_SINGLETON is None:
        from waterseg_platform.config import PlatformConfig
        from waterseg_platform.pipeline import SegmentationService
        _SERVICE_SINGLETON = SegmentationService(default_config or PlatformConfig())
    return _SERVICE_SINGLETON


def _import_gradio():
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "Gradio is not installed. Run `pip install gradio` to use the UI."
        ) from exc
    return gr


def _make_overlay_rgb(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import cv2

    h, w = image_bgr.shape[:2]
    rgb = image_bgr[..., ::-1].copy()
    mask_panel = np.zeros((h, w, 3), dtype=np.uint8)
    mask_panel[mask > 0] = (255, 255, 255)
    overlay = rgb.copy()
    red = np.zeros_like(rgb)
    red[..., 0] = 255
    m_bool = mask > 0
    if m_bool.any():
        blended = (0.45 * rgb[m_bool].astype(np.float32)
                   + 0.55 * red[m_bool].astype(np.float32))
        overlay[m_bool] = np.clip(blended, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            m_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    return np.concatenate([rgb, mask_panel, overlay], axis=1)


def _segment_one(
    image_rgb, conf, iou, mask_thres, min_area_ratio, morph_close,
    use_tiling, tile_size, tile_overlap, sam3_enabled, sam3_mode,
) -> Tuple[Optional[np.ndarray], str]:
    if image_rgb is None:
        return None, "Upload an image to segment."
    image_bgr = image_rgb[..., ::-1].copy()
    svc = _get_service()
    try:
        # Pass the slider values through as overrides — do NOT mutate
        # svc.config (mutating shared state across requests is fragile).
        # `use_tiling=True` dispatches to the new tiled pipeline that
        # handles large/rectangular images by splitting into 704x704
        # patches and stitching per-tile probability maps.
        if use_tiling:
            mask, info = svc.segment_array_tiled(
                image_bgr,
                tile_size=tile_size if tile_size and tile_size > 0 else None,
                tile_overlap_px=tile_overlap,
                conf=conf, iou=iou, mask_thres=mask_thres,
                min_area_ratio=min_area_ratio, morph_close=morph_close,
                sam3_enabled=sam3_enabled,
                sam3_mode=sam3_mode,
            )
        else:
            mask, info = svc.segment_array(
                image_bgr,
                conf=conf, iou=iou, mask_thres=mask_thres,
                min_area_ratio=min_area_ratio, morph_close=morph_close,
                sam3_enabled=sam3_enabled,
                sam3_mode=sam3_mode,
            )
    except Exception as exc:
        return None, f"error: {exc}\n{traceback.format_exc(limit=2)}"
    panel = _make_overlay_rgb(image_bgr, mask)
    # Format the geometry line(s) based on which path produced the
    # prediction. The single-tile path exposes a single ``letterbox`` key
    # describing the whole-image letterbox; the tiled path doesn't (each
    # tile has its own letterbox), so we report the tile grid instead.
    geom_lines: List[str] = []
    if "letterbox" in info:
        lb = info["letterbox"]
        geom_lines.append(
            f"letterbox: r={lb['r']:.4f}  "
            f"pad=({lb['pad_w']}, {lb['pad_h']})  "
            f"new=({lb['new_w']}, {lb['new_h']})"
        )
    tiling_info = info.get("tiling")
    if tiling_info is not None:
        gate_skip = tiling_info.get('gate_skipped_tiling', False)
        gate_ms = tiling_info.get('gate_coarse_ms', 0)
        geom_lines.append(
            f"tiling: tile={tiling_info['tile_size']}  "
            f"overlap={tiling_info['tile_overlap_px']}  "
            f"n={tiling_info['num_tiles']}  "
            f"combine={tiling_info['combine_mode']}  "
            f"engine_ms/tile={tiling_info['elapsed_ms_per_tile_avg']:.1f}  "
            f"gate_ms={gate_ms:.1f}  "
            f"skip={gate_skip}"
        )
    sam3_info = info.get("sam3", {})
    sam3_line = (
        f"sam3: enabled={sam3_info.get('enabled', False)}  "
        f"mode={sam3_info.get('mode', 'conservative')}  "
        f"ran={sam3_info.get('ran', False)}  "
        f"candidates={sam3_info.get('candidate_count', 0)}  "
        f"accepted={sam3_info.get('accepted', 0)}  "
        f"fallback={sam3_info.get('fallback', False)}"
    )
    if sam3_info.get("error"):
        sam3_line += f"  error={sam3_info['error']}"
    geom_lines.append(sam3_line)
    text = (
        f"providers: {','.join(info['providers'])}\n"
        f"mask_area: {info['pred_area']:,} px\n"
        f"pred_area_ratio: {info['pred_area_ratio']:.4f}\n"
        + "\n".join(geom_lines) + "\n"
        f"conf={conf:.2f}  iou={iou:.2f}  mask_thres={mask_thres:.2f}  "
        f"min_area_ratio={min_area_ratio:.5f}  morph_close={morph_close}"
    )
    return panel, text


def _segment_directory(
    image_dir, output_dir, num, conf, iou, mask_thres,
    min_area_ratio, morph_close, use_tiling, tile_size, tile_overlap,
    sam3_enabled, sam3_mode,
) -> str:
    if not image_dir or not Path(image_dir).is_dir():
        return f"error: image_dir does not exist: {image_dir}"
    if not output_dir:
        return "error: output_dir is required"
    svc = _get_service()
    # Pass the slider values through as overrides — same as the
    # single-image path. The previous version mutated svc.config, which
    # is fragile (state leaks across calls, partial overrides).
    df = svc.segment_directory(
        image_dir=image_dir,
        output_dir=output_dir,
        num=int(num) if num and num > 0 else None,
        conf=conf, iou=iou, mask_thres=mask_thres,
        min_area_ratio=min_area_ratio, morph_close=morph_close,
        use_tiling=use_tiling,
        tile_size=tile_size if tile_size and tile_size > 0 else None,
        tile_overlap_px=tile_overlap,
        sam3_enabled=sam3_enabled,
        sam3_mode=sam3_mode,
    )
    csv_path = Path(output_dir) / "metrics.csv"
    return f"Processed {len(df)} images. Metrics saved to {csv_path}."


def build_app(default_config=None):
    gr = _import_gradio()
    from waterseg_platform.config import PlatformConfig

    cfg = default_config or PlatformConfig()
    # Initialize the singleton service eagerly so we can read its
    # providers for the header. The service is reused across requests
    # via the _get_service() lookup in the click handlers.
    svc = _get_service(cfg)
    providers_str = ",".join(svc.engine.providers_active)

    with gr.Blocks(title="Water Segmentation (ONNX)") as demo:
        gr.Markdown(
            f"# Water Segmentation (ONNX)\n"
            f"Model: `{cfg.model_path}`  imgsz: {cfg.imgsz}  "
            f"providers: {providers_str}"
        )

        with gr.Tab("Single image"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(label="Input image", type="numpy")
                    with gr.Row():
                        conf = gr.Slider(0.05, 0.95, value=cfg.conf, step=0.05,
                                         label="conf")
                        iou = gr.Slider(0.05, 0.95, value=cfg.iou, step=0.05,
                                        label="IoU (NMS)")
                    with gr.Row():
                        mask_thres = gr.Slider(0.10, 0.90, value=cfg.mask_thres,
                                               step=0.05, label="mask threshold")
                        min_area = gr.Slider(0.0, 0.01, value=cfg.min_area_ratio,
                                             step=0.0001,
                                             label="min area ratio")
                    morph_close = gr.Checkbox(value=cfg.morph_close,
                                              label="MORPH_CLOSE")
                    with gr.Accordion("Tiling", open=False):
                        use_tiling = gr.Checkbox(
                            value=True,
                            label="Use tiling (split large/rectangular images "
                                  "into 1024x1024 patches)",
                        )
                        tile_size = gr.Number(
                            value=cfg.tile_size,
                            label="Tile size (0 = use imgsz)",
                            precision=0,
                        )
                        tile_overlap = gr.Slider(
                            0, max(1, cfg.tile_size - 1),
                            value=cfg.tile_overlap_px, step=1,
                            label="Tile overlap (px)",
                        )
                    with gr.Accordion("SAM 3", open=False):
                        sam3_enabled = gr.Checkbox(
                            value=cfg.sam3_enabled,
                            label="SAM 3 refinement",
                        )
                        sam3_mode = gr.Dropdown(
                            choices=["conservative", "balanced", "open"],
                            value=cfg.sam3_mode,
                            label="Fusion mode",
                        )
                    btn = gr.Button("Segment", variant="primary")
                with gr.Column():
                    out_panel = gr.Image(label="image | mask | overlay",
                                         type="numpy")
                    out_text = gr.Textbox(label="info", lines=7)

            btn.click(
                _segment_one,
                inputs=[inp, conf, iou, mask_thres, min_area, morph_close,
                        use_tiling, tile_size, tile_overlap, sam3_enabled,
                        sam3_mode],
                outputs=[out_panel, out_text],
            )

        with gr.Tab("Directory"):
            with gr.Row():
                with gr.Column():
                    image_dir = gr.Textbox(label="image_dir (absolute path)")
                    output_dir = gr.Textbox(label="output_dir (absolute path)")
                    num = gr.Number(value=0, label="num (0 = all)",
                                    precision=0)
                    with gr.Row():
                        d_conf = gr.Slider(0.05, 0.95, value=cfg.conf,
                                           step=0.05, label="conf")
                        d_iou = gr.Slider(0.05, 0.95, value=cfg.iou,
                                          step=0.05, label="IoU (NMS)")
                    with gr.Row():
                        d_mt = gr.Slider(0.10, 0.90, value=cfg.mask_thres,
                                         step=0.05, label="mask threshold")
                        d_min_area = gr.Slider(
                            0.0, 0.01, value=cfg.min_area_ratio,
                            step=0.0001, label="min area ratio",
                        )
                    d_morph = gr.Checkbox(value=cfg.morph_close,
                                          label="MORPH_CLOSE")
                    with gr.Accordion("Tiling", open=False):
                        d_use_tiling = gr.Checkbox(
                            value=True,
                            label="Use tiling",
                        )
                        d_tile_size = gr.Number(
                            value=cfg.tile_size,
                            label="Tile size (0 = use imgsz)",
                            precision=0,
                        )
                        d_tile_overlap = gr.Slider(
                            0, max(1, cfg.tile_size - 1),
                            value=cfg.tile_overlap_px, step=1,
                            label="Tile overlap (px)",
                        )
                    with gr.Accordion("SAM 3", open=False):
                        d_sam3_enabled = gr.Checkbox(
                            value=cfg.sam3_enabled,
                            label="SAM 3 refinement",
                        )
                        d_sam3_mode = gr.Dropdown(
                            choices=["conservative", "balanced", "open"],
                            value=cfg.sam3_mode,
                            label="Fusion mode",
                        )
                    d_btn = gr.Button("Run", variant="primary")
                with gr.Column():
                    d_out = gr.Textbox(label="result", lines=4)
            d_btn.click(
                _segment_directory,
                inputs=[image_dir, output_dir, num, d_conf, d_iou, d_mt,
                        d_min_area, d_morph, d_use_tiling, d_tile_size,
                        d_tile_overlap, d_sam3_enabled,
                        d_sam3_mode],
                outputs=d_out,
            )

    return demo


def launch(
    default_config=None,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    app = build_app(default_config=default_config)
    app.launch(server_name=server_name, server_port=server_port, share=share)

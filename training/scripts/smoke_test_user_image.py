"""Smoke test: run the waterseg_platform on the user's test image.

Exercises both the legacy single-tile path and the new tiled path on a
high-resolution aerial flood image, prints a side-by-side summary, and
saves masks + overlays to runs/smoke_test/.

Run:
    C:/Users/17473/miniforge3/envs/torch_env/python.exe scripts/smoke_test_user_image.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waterseg_platform import PlatformConfig, SegmentationService
from waterseg_platform.visualization import save_panel

IMAGE_PATH = Path(r"C:\Users\17473\Desktop\test_image.png")
OUT_DIR = ROOT / "runs" / "smoke_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if not IMAGE_PATH.exists():
        print(f"error: image not found: {IMAGE_PATH}", file=sys.stderr)
        return 2

    print(f"loading image: {IMAGE_PATH}")
    img_bgr = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"error: could not read {IMAGE_PATH}", file=sys.stderr)
        return 2
    H, W = img_bgr.shape[:2]
    print(f"  image shape: H={H}, W={W}, channels={img_bgr.shape[2]}")

    cfg = PlatformConfig(
        model_path=str(ROOT / "onnx" / "best.onnx"),
        imgsz=704,
        conf=0.25,
        iou=0.5,
        mask_thres=0.5,
        min_area_ratio=0.0005,
        morph_close=True,
        tile_size=704,
        tile_overlap_px=0,
        providers=["CPUExecutionProvider"],
    )
    svc = SegmentationService(cfg)
    print(f"  engine providers: {svc.engine.providers_active}")
    print()

    # ---- Path 1: legacy single-tile letterbox --------------------------- #
    print("[1/2] legacy single-tile letterbox path (--no_tiling equivalent) ...")
    t0 = time.perf_counter()
    mask_single, info_single = svc.segment_array(img_bgr)
    t_single = (time.perf_counter() - t0) * 1000.0
    print(f"  elapsed: {t_single:.1f} ms")
    print(f"  mask shape: {mask_single.shape}")
    print(f"  pred_area: {info_single['pred_area']:,} px")
    print(f"  pred_area_ratio: {info_single['pred_area_ratio']:.4f}")
    print(f"  letterbox: r={info_single['letterbox']['r']:.4f}  "
          f"new=({info_single['letterbox']['new_w']}, {info_single['letterbox']['new_h']})  "
          f"pad=({info_single['letterbox']['pad_w']}, {info_single['letterbox']['pad_h']})")
    cv2.imwrite(str(OUT_DIR / "single_pred.png"), (mask_single * 255).astype(np.uint8))
    save_panel(OUT_DIR / "single_overlay.jpg", img_bgr, gt=None, pred=mask_single)
    print(f"  wrote {OUT_DIR / 'single_pred.png'}")
    print(f"  wrote {OUT_DIR / 'single_overlay.jpg'}")
    print()

    # ---- Path 2: tiled (the new default) -------------------------------- #
    print("[2/2] tiled path (always tile, default) ...")
    t0 = time.perf_counter()
    mask_tiled, info_tiled = svc.segment_array_tiled(img_bgr)
    t_tiled = (time.perf_counter() - t0) * 1000.0
    t = info_tiled["tiling"]
    print(f"  elapsed: {t_tiled:.1f} ms (avg per tile: {t['elapsed_ms_per_tile_avg']:.1f} ms)")
    print(f"  mask shape: {mask_tiled.shape}")
    print(f"  pred_area: {info_tiled['pred_area']:,} px")
    print(f"  pred_area_ratio: {info_tiled['pred_area_ratio']:.4f}")
    print(f"  tiling: tile={t['tile_size']}  overlap={t['tile_overlap_px']}  "
          f"n={t['num_tiles']}  combine={t['combine_mode']}")
    cv2.imwrite(str(OUT_DIR / "tiled_pred.png"), (mask_tiled * 255).astype(np.uint8))
    save_panel(OUT_DIR / "tiled_overlay.jpg", img_bgr, gt=None, pred=mask_tiled)
    print(f"  wrote {OUT_DIR / 'tiled_pred.png'}")
    print(f"  wrote {OUT_DIR / 'tiled_overlay.jpg'}")
    print()

    # ---- Headline numbers ---------------------------------------------- #
    # How much detail did we lose in the single-tile path's downscale?
    # An approximate IoU between the two predictions gives a sense.
    inter = int(np.logical_and(mask_single, mask_tiled).sum())
    union = int(np.logical_or(mask_single, mask_tiled).sum())
    iou = inter / max(union, 1)
    only_single = int(np.logical_and(mask_single, np.logical_not(mask_tiled)).sum())
    only_tiled = int(np.logical_and(mask_tiled, np.logical_not(mask_single)).sum())
    print("--- comparison ---")
    print(f"  tiled pred_area: {info_tiled['pred_area']:,}")
    print(f"  single pred_area: {info_single['pred_area']:,}")
    print(f"  extra water found by tiled path: {only_tiled:,} px "
          f"({only_tiled / max(info_tiled['pred_area'], 1) * 100:.1f}% of tiled total)")
    print(f"  water found by single only: {only_single:,} px")
    print(f"  IoU(single, tiled): {iou:.4f}")
    print()
    print(f"all outputs in: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

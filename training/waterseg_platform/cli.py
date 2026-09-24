"""Command-line interface for the ONNX water-segmentation platform.

Two execution modes:

* Single image:
      python -m waterseg_platform.cli --image path/to/image.jpg --output_dir out/

* Directory of images:
      python -m waterseg_platform.cli --image_dir path/to/dir/ --output_dir out/

* Launch the Gradio web UI:
      python -m waterseg_platform.cli ui

A YAML config file (``--config configs/onnx_platform.yaml``) supplies the
default hyperparameters. CLI flags override the config.

.NET equivalent: ``Program.cs`` (``System.CommandLine``).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

# Ensure project root is importable so `waterseg_platform` resolves.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from waterseg_platform.config import (  # noqa: E402
    PlatformConfig,
    dump_config,
    load_config,
)
from waterseg_platform.pipeline import SegmentationService  # noqa: E402


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_config(args: argparse.Namespace) -> PlatformConfig:
    """Compose a ``PlatformConfig`` from YAML + CLI overrides."""
    cfg = load_config(args.config)
    if args.model:
        cfg.model_path = args.model
    if args.imgsz is not None:
        cfg.imgsz = args.imgsz
    if args.conf is not None:
        cfg.conf = args.conf
    if args.iou is not None:
        cfg.iou = args.iou
    if args.mask_thres is not None:
        cfg.mask_thres = args.mask_thres
    if args.min_area_ratio is not None:
        cfg.min_area_ratio = args.min_area_ratio
    if args.no_morph_close:
        cfg.morph_close = False
    if args.providers:
        cfg.providers = list(args.providers)
    # Tiling config: tile_size / tile_overlap_px must equal imgsz unless
    # the user re-exports the ONNX. The service enforces this at call time.
    if args.tile_size is not None and args.tile_size > 0:
        cfg.tile_size = args.tile_size
    if args.tile_overlap is not None:
        cfg.tile_overlap_px = args.tile_overlap
    if args.sam3_enabled is not None:
        cfg.sam3_enabled = bool(args.sam3_enabled)
    if args.sam3_mode is not None:
        cfg.sam3_mode = args.sam3_mode
    if args.sam3_open:
        cfg.sam3_mode = "open"
    return cfg


def _cmd_single(args: argparse.Namespace, cfg: PlatformConfig) -> int:
    if not args.image:
        print("error: --image is required for single-image mode", file=sys.stderr)
        return 2
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 2
    out_dir = Path(args.output_dir).resolve() if args.output_dir else None

    LOG = logging.getLogger("cli")
    LOG.info("Single-image inference: %s", image_path)
    svc = SegmentationService(cfg)
    t0 = time.perf_counter()
    use_tiling = not args.no_tiling
    result = svc.segment_file(
        str(image_path),
        output_dir=str(out_dir) if out_dir else None,
        save_overlay=args.save_overlay,
        save_mask=args.save_mask,
        use_tiling=use_tiling,
    )
    elapsed = (time.perf_counter() - t0) * 1000.0
    LOG.info(
        "Done in %.1f ms (engine-only: %.1f ms). mask_area=%d, ratio=%.4f",
        elapsed, result["elapsed_ms"], result["mask_area"], result["pred_area_ratio"],
    )
    for k in ("mask_path", "overlay_path"):
        v = result.get(k)
        if v:
            print(f"  {k}: {v}")
    return 0


def _cmd_directory(args: argparse.Namespace, cfg: PlatformConfig) -> int:
    if not args.image_dir:
        print("error: --image_dir is required for directory mode", file=sys.stderr)
        return 2
    image_dir = Path(args.image_dir).resolve()
    if not image_dir.is_dir():
        print(f"error: not a directory: {image_dir}", file=sys.stderr)
        return 2
    if not args.output_dir:
        print("error: --output_dir is required for directory mode", file=sys.stderr)
        return 2
    out_dir = Path(args.output_dir).resolve()

    LOG = logging.getLogger("cli")
    LOG.info(
        "Directory inference: %s -> %s (num=%s, gt_mask_dir=%s)",
        image_dir, out_dir, args.num, args.gt_mask_dir,
    )
    svc = SegmentationService(cfg)
    df = svc.segment_directory(
        image_dir=str(image_dir),
        output_dir=str(out_dir),
        num=args.num,
        save_overlay=args.save_overlay,
        save_mask=args.save_mask,
        gt_mask_dir=args.gt_mask_dir,
        mask_mode=args.mask_mode,
        foreground_rgb=args.foreground_rgb,
        tolerance=args.tolerance,
        use_tiling=not args.no_tiling,
    )
    LOG.info("Wrote %d rows to %s/metrics.csv", len(df), out_dir)
    summary_path = out_dir / "summary.txt"
    if summary_path.exists():
        print("\n--- Summary ---")
        print(summary_path.read_text(encoding="utf-8"))
    return 0


def _cmd_ui(args: argparse.Namespace, cfg: PlatformConfig) -> int:
    """Launch the Gradio web UI. Lazy import to keep `cli --help` fast."""
    LOG = logging.getLogger("cli")
    try:
        from waterseg_platform.ui_gradio import launch
    except ImportError as exc:
        print(
            f"error: Gradio UI not available ({exc}). "
            "Install with `pip install gradio`.",
            file=sys.stderr,
        )
        return 1
    server_name = args.server_name or "127.0.0.1"
    LOG.info(
        "Launching Gradio UI on %s:%d (share=%s)  "
        "imgsz=%d  conf=%.2f  iou=%.2f  mask_thres=%.2f  "
        "min_area_ratio=%.5f  morph_close=%s  providers=%s  "
        "tile_size=%d  tile_overlap_px=%d",
        server_name, args.server_port, args.share,
        cfg.imgsz, cfg.conf, cfg.iou, cfg.mask_thres,
        cfg.min_area_ratio, cfg.morph_close, cfg.providers,
        cfg.tile_size, cfg.tile_overlap_px,
    )
    launch(
        default_config=cfg,
        server_name=server_name,
        server_port=args.server_port,
        share=args.share,
    )
    return 0


def _cmd_dump_config(args: argparse.Namespace) -> int:
    cfg = PlatformConfig()
    dump_config(cfg, args.dump_config)
    print(f"Wrote default config to {args.dump_config}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="waterseg_platform",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/onnx_platform.yaml",
                   help="YAML config file. CLI flags override it.")
    p.add_argument("--model", default=None,
                   help="Path to the ONNX model. Overrides the config.")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--mask_thres", type=float, default=None)
    p.add_argument("--min_area_ratio", type=float, default=None)
    p.add_argument("--no_morph_close", action="store_true",
                   help="Disable MORPH_CLOSE on the predicted mask.")
    p.add_argument(
        "--providers", nargs="+", default=None,
        help="Override ONNX Runtime provider list, e.g. "
             "--providers CPUExecutionProvider",
    )
    # Tiling flags. The model has a fixed input shape, so tile_size
    # should normally equal imgsz (704). tile_overlap must be < tile_size.
    # --no_tiling forces the legacy single-tile letterbox path.
    p.add_argument("--tile_size", type=int, default=None,
                   help="Side length of each tile in pixels. Must equal "
                        "--imgsz unless the ONNX was re-exported. "
                        "Default: 0 (use config.tile_size).")
    p.add_argument("--tile_overlap", type=int, default=None,
                   help="Overlap between adjacent tiles in pixels, "
                        "in [0, tile_size-1]. Default: 0.")
    p.add_argument("--no_tiling", action="store_true",
                   help="Disable tiled inference; use the legacy single-tile "
                        "letterbox path (matches the parity test).")
    sam3_group = p.add_mutually_exclusive_group()
    sam3_group.add_argument(
        "--sam3",
        dest="sam3_enabled",
        action="store_true",
        default=None,
        help="Enable optional SAM 3 refinement.",
    )
    sam3_group.add_argument(
        "--no_sam3",
        dest="sam3_enabled",
        action="store_false",
        help="Disable SAM 3 refinement even if enabled in YAML.",
    )
    p.add_argument(
        "--sam3_mode",
        choices=["conservative", "balanced", "open"],
        default=None,
        help="SAM 3 fusion mode. Conservative is production-safe; open is experimental.",
    )
    p.add_argument(
        "--sam3_open",
        action="store_true",
        help="Deprecated alias for --sam3_mode open.",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dump-config", default=None,
                   help="Write the default config to this path and exit.")

    # Subcommands. We use a positional `mode` rather than argparse subparsers
    # so the help text stays compact and the `ui` invocation reads naturally.
    p.add_argument("mode", nargs="?", default="single",
                   choices=["single", "dir", "ui"],
                   help="Execution mode (default: single).")

    # Single-image flags.
    p.add_argument("--image", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--save_overlay", action="store_true")
    p.add_argument("--save_mask", action="store_true", default=True)
    p.add_argument("--no_save_mask", dest="save_mask", action="store_false")

    # Directory flags.
    p.add_argument("--image_dir", default=None)
    p.add_argument("--num", type=int, default=None,
                   help="Cap on the number of images to process (default: all).")
    p.add_argument("--gt_mask_dir", default=None,
                   help="Optional ground-truth mask dir for per-image metrics.")
    p.add_argument("--mask_mode", default="grayscale",
                   choices=["red", "non_black", "grayscale", "auto"])
    p.add_argument("--foreground_rgb", default="128,0,0")
    p.add_argument("--tolerance", type=int, default=0)

    # UI flags.
    p.add_argument("--server_name", default=None)
    p.add_argument("--server_port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)

    if args.dump_config:
        return _cmd_dump_config(args)

    cfg = _build_config(args)

    if args.mode == "ui":
        return _cmd_ui(args, cfg)
    if args.mode == "dir":
        return _cmd_directory(args, cfg)
    return _cmd_single(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())

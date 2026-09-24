"""Compare YOLO-only, conservative SAM 3, and open SAM 3 refinement."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waterseg_platform.config import load_config  # noqa: E402
from waterseg_platform.image_io import ensure_dir, read_image  # noqa: E402
from waterseg_platform.pipeline import SegmentationService  # noqa: E402
from waterseg_platform.visualization import save_panel  # noqa: E402


IMAGE_PATH = Path(r"C:\Users\17473\Desktop\test_image.png")
OUTPUT_DIR = ROOT / "runs" / "platform_compare" / "sam3_refinement"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Sample image not found: {IMAGE_PATH}")
    output_dir = ensure_dir(OUTPUT_DIR)
    image = read_image(IMAGE_PATH)
    config = load_config(str(ROOT / "configs" / "onnx_platform_sam3_trial.yaml"))
    service = SegmentationService(config)

    modes = [
        ("yolo_only", False, "conservative"),
        ("sam3_conservative", True, "conservative"),
        ("sam3_balanced", True, "balanced"),
        ("sam3_open", True, "open"),
    ]
    report = {
        "image": str(IMAGE_PATH),
        "shape": list(image.shape),
        "model": config.model_path,
        "checkpoint": config.sam3_checkpoint,
        "use_tiling": False,
        "modes": {},
    }

    for name, enabled, mode in modes:
        started = time.perf_counter()
        mask, info = service.segment_array(
            image,
            sam3_enabled=enabled,
            sam3_mode=mode,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        mask_path = output_dir / f"{name}_mask.png"
        overlay_path = output_dir / f"{name}_overlay.jpg"
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        save_panel(overlay_path, image, gt=None, pred=mask)
        report["modes"][name] = {
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "elapsed_ms": elapsed_ms,
            "area": int(mask.sum()),
            "area_ratio": float(mask.mean()),
            "info": _json_safe(info),
        }
        print(
            f"{name}: area_ratio={float(mask.mean()):.4f}, "
            f"elapsed_ms={elapsed_ms:.1f}, sam3={info.get('sam3')}"
        )

    report_path = output_dir / "comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

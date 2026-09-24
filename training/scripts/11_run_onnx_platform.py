#!/usr/bin/env python
"""Thin script entry point for the ONNX water-segmentation platform.

Equivalent to running `python -m waterseg_platform.cli` from the project
root. Kept as a separate script so the platform can be invoked without
remembering the `python -m ...` syntax.

Examples:
    python scripts/11_run_onnx_platform.py --image path/to.jpg --output_dir out/
    python scripts/11_run_onnx_platform.py dir --image_dir data/.../test --output_dir out/
    python scripts/11_run_onnx_platform.py ui
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable so `waterseg_platform` resolves.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from waterseg_platform.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

"""Visualization helpers re-exported from ``src/visualization.py``.

The functions here are the ones that the .NET port will replace with
``OverlayRenderer.cs`` (OpenCvSharp4).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.visualization import (  # noqa: E402,F401
    comparison_panel,
    draw_contours,
    make_panel,
    mask_panel,
    overlay_mask,
    overlay_mask_with_contour,
    save_panel,
)

__all__ = [
    "comparison_panel",
    "draw_contours",
    "make_panel",
    "mask_panel",
    "overlay_mask",
    "overlay_mask_with_contour",
    "save_panel",
]

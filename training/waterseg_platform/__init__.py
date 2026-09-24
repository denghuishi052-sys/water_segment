"""ONNX-based water segmentation platform.

This package exposes a pure-NumPy + OpenCV + ONNX Runtime inference pipeline
that has no runtime dependency on PyTorch or Ultralytics. The module
boundaries are designed to map 1:1 onto .NET classes (see
``waterseg_platform/README.md`` for the translation table).

Note: the package is named ``waterseg_platform`` (not ``platform``) to avoid
shadowing the stdlib ``platform`` module when running from the project root.

Public entry points (all lazy-loaded so the package can be imported even if
optional dependencies are missing):
    from waterseg_platform import PlatformConfig, OnnxSegmenter, SegmentationService
    from waterseg_platform.preprocessing import letterbox
    from waterseg_platform.postprocessing import decode
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "PlatformConfig",
    "OnnxSegmenter",
    "SegmentationService",
    "load_config",
]

# Eager: pure-Python modules that have no heavy deps.
from waterseg_platform.config import PlatformConfig, load_config  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - only for static type checkers
    from waterseg_platform.engine import OnnxSegmenter
    from waterseg_platform.pipeline import SegmentationService


def __getattr__(name: str):
    """Lazy attribute access for heavy / optional submodules."""
    if name == "OnnxSegmenter":
        from waterseg_platform.engine import OnnxSegmenter

        return OnnxSegmenter
    if name == "SegmentationService":
        from waterseg_platform.pipeline import SegmentationService

        return SegmentationService
    raise AttributeError(f"module 'waterseg_platform' has no attribute {name!r}")

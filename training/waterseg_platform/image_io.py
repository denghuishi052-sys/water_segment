"""Image I/O wrappers re-exported from ``src/dataset_utils.py``.

This module is the only place the platform talks to the file system. It
mirrors the .NET ``ImageIo.cs`` (OpenCvSharp4).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``src/`` importable so we can reuse the existing functions verbatim.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataset_utils import (  # noqa: E402
    IMAGE_EXTS,
    ensure_dir,
    list_files,
    read_image,
    read_mask_binary,
)

__all__ = [
    "IMAGE_EXTS",
    "ensure_dir",
    "list_files",
    "read_image",
    "read_mask_binary",
]

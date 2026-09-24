"""Metrics re-exported from ``src/metrics.py``.

.NET equivalent: ``Metrics.cs`` (pure C# — no NuGet dependency).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.metrics import (  # noqa: E402,F401
    SegMetrics,
    aggregate_metrics,
    compute_binary_metrics,
    compute_counts,
)

__all__ = [
    "SegMetrics",
    "aggregate_metrics",
    "compute_binary_metrics",
    "compute_counts",
]

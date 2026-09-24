"""Portable proposal-benefit selector exported as plain JSON."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_NAMES = [
    "score",
    "area_ratio",
    "overlap",
    "coverage",
    "iou",
    "area_growth",
    "distance_ratio",
    "candidate_area_ratio",
    "yolo_area_ratio",
    "yolo_confidence",
    "is_text_box",
    "is_interactive_points",
    "is_global_text",
]

NONFINITE_FEATURE_SENTINELS = {
    # distance_ratio is normalized by image diagonal in the proposal cache.
    # Infinity means there is no YOLO foreground to measure from, which is a
    # useful signal for global proposals / empty-YOLO recovery. Keep it as a
    # finite out-of-range sentinel so JSON selectors and sklearn can consume it.
    "distance_ratio": 2.0,
}


def sanitize_selector_feature(name: str, value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isfinite(numeric):
        return numeric
    if math.isnan(numeric):
        return 0.0
    sentinel = NONFINITE_FEATURE_SENTINELS.get(name)
    if sentinel is not None:
        return sentinel if numeric > 0 else -sentinel
    return 1_000_000.0 if numeric > 0 else -1_000_000.0


def selector_feature_vector(
    features: dict,
    *,
    source: str,
    candidate_area_ratio: float,
    yolo_area_ratio: float,
    yolo_confidence: float = 0.0,
) -> np.ndarray:
    values = {
        **features,
        "candidate_area_ratio": candidate_area_ratio,
        "yolo_area_ratio": yolo_area_ratio,
        "yolo_confidence": yolo_confidence,
        "is_text_box": float(source == "text_box"),
        "is_interactive_points": float(source == "interactive_points"),
        "is_global_text": float(source == "global_text"),
    }
    return np.asarray(
        [sanitize_selector_feature(name, values.get(name, 0.0)) for name in FEATURE_NAMES],
        dtype=np.float64,
    )


@dataclass
class ProposalSelector:
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    accept_threshold: float

    @classmethod
    def load(cls, path: str | Path) -> "ProposalSelector":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "waterseg_proposal_selector_v1":
            raise ValueError("Unsupported proposal selector format")
        if payload["feature_names"] != FEATURE_NAMES:
            raise ValueError("Proposal selector feature order mismatch")
        return cls(
            feature_names=payload["feature_names"],
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=float(payload["bias"]),
            accept_threshold=float(payload["accept_threshold"]),
        )

    def predict_probability(self, vector: np.ndarray) -> float:
        safe_vector = np.asarray(
            [
                sanitize_selector_feature(name, value)
                for name, value in zip(self.feature_names, np.asarray(vector))
            ],
            dtype=np.float64,
        )
        standardized = (safe_vector - self.mean) / np.maximum(
            self.scale, 1e-12
        )
        logit = float(standardized @ self.weights + self.bias)
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)

    def accepts(self, vector: np.ndarray) -> tuple[bool, float]:
        probability = self.predict_probability(vector)
        return probability >= self.accept_threshold, probability

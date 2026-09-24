"""Training and validation utilities for the portable proposal selector."""
from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from waterseg_platform.sam3_selector import FEATURE_NAMES, sanitize_selector_feature


def feature_matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            [
                sanitize_selector_feature(name, row["features"].get(name, 0.0))
                for name in FEATURE_NAMES
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


@lru_cache(maxsize=65536)
def _read_binary_cached(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 0).astype(np.uint8)


def read_binary(path: str) -> np.ndarray:
    return _read_binary_cached(path).copy()


def micro_metrics(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    tp = fp = fn = 0
    for pred, gt in pairs:
        pred = np.asarray(pred) > 0
        gt = np.asarray(gt) > 0
        tp += int(np.logical_and(pred, gt).sum())
        fp += int(np.logical_and(pred, ~gt).sum())
        fn += int(np.logical_and(~pred, gt).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return {"precision": precision, "recall": recall, "iou": iou}


def evaluate_threshold(
    rows: list[dict], probabilities: np.ndarray, threshold: float
) -> dict:
    grouped: dict[int, list[tuple[dict, float]]] = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        grouped[int(row["sample_id"])].append((row, float(probability)))
    pairs = []
    for sample_rows in grouped.values():
        first = sample_rows[0][0]
        output = read_binary(first["yolo_mask"])
        gt = read_binary(first["gt_mask"])
        choices: dict[str, tuple[dict, float]] = {}
        for row, probability in sample_rows:
            key = (
                f"candidate:{row['candidate_index']}"
                if row["candidate_index"] is not None
                else "global"
            )
            if probability >= threshold and (
                key not in choices or probability > choices[key][1]
            ):
                choices[key] = (row, probability)
        for key, (row, _) in choices.items():
            proposal = read_binary(row["proposal_mask"])
            if key.startswith("candidate:") and row["candidate_mask"]:
                candidate = read_binary(row["candidate_mask"])
                output[candidate > 0] = 0
            output |= proposal
        pairs.append((output, gt))
    return micro_metrics(pairs)


def train_selector(
    train_rows: list[dict],
    val_rows: list[dict],
    *,
    precision_tolerance: float = 0.01,
) -> tuple[dict, dict]:
    x_train = feature_matrix(train_rows)
    y_train = np.asarray(
        [int(bool(row["beneficial"])) for row in train_rows],
        dtype=np.int64,
    )
    if len(np.unique(y_train)) != 2:
        raise ValueError("Selector training needs beneficial and harmful proposals")
    scaler = StandardScaler().fit(x_train)
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=0
    ).fit(scaler.transform(x_train), y_train)
    x_val = feature_matrix(val_rows)
    probabilities = classifier.predict_proba(
        scaler.transform(x_val)
    )[:, 1]
    baseline_pairs = []
    seen = set()
    for row in val_rows:
        sample_id = int(row["sample_id"])
        if sample_id in seen:
            continue
        seen.add(sample_id)
        baseline_pairs.append(
            (read_binary(row["yolo_mask"]), read_binary(row["gt_mask"]))
        )
    baseline = micro_metrics(baseline_pairs)
    minimum_precision = baseline["precision"] - precision_tolerance
    best = None
    evaluations = []
    thresholds = list(np.linspace(0.0, 1.0, 201)) + [1.000001]
    for threshold in thresholds:
        metrics = evaluate_threshold(val_rows, probabilities, float(threshold))
        metrics["threshold"] = float(threshold)
        evaluations.append(metrics)
        if metrics["precision"] + 1e-12 < minimum_precision:
            continue
        key = (metrics["iou"], metrics["recall"], metrics["precision"])
        if best is None or key > best[0]:
            best = (key, metrics)
    if best is None:
        # Threshold 1.0 is an exact "keep YOLO" policy and should always pass,
        # but retain an explicit guard for malformed validation caches.
        raise RuntimeError("No selector threshold satisfies precision constraint")
    selected = best[1]
    payload = {
        "format": "waterseg_proposal_selector_v1",
        "feature_names": FEATURE_NAMES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "weights": classifier.coef_[0].tolist(),
        "bias": float(classifier.intercept_[0]),
        "accept_threshold": selected["threshold"],
        "training_rows": len(train_rows),
        "validation_rows": len(val_rows),
    }
    report = {
        "baseline": baseline,
        "minimum_precision": minimum_precision,
        "selected": selected,
        "threshold_evaluations": evaluations,
    }
    return payload, report


def load_rows(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

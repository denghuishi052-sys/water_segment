from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class SegMetrics:
    precision: float
    recall: float
    f1: float
    iou: float
    dice: float
    tp: int
    fp: int
    fn: int
    tn: int


def compute_binary_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> SegMetrics:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    if tp == 0 and fp == 0 and fn == 0:
        return SegMetrics(1.0, 1.0, 1.0, 1.0, 1.0, tp, fp, fn, tn)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return SegMetrics(precision, recall, f1, iou, dice, tp, fp, fn, tn)


def aggregate_metrics(rows: List[dict]) -> Dict[str, float]:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    tp = int(df["tp"].sum())
    fp = int(df["fp"].sum())
    fn = int(df["fn"].sum())
    tn = int(df["tn"].sum())
    micro = compute_counts(tp, fp, fn, tn)
    macro = {
        "macro_precision": float(df["precision"].mean()),
        "macro_recall": float(df["recall"].mean()),
        "macro_f1": float(df["f1"].mean()),
        "macro_iou": float(df["iou"].mean()),
        "macro_dice": float(df["dice"].mean()),
    }
    macro.update(_macro_subset(df, "foreground_macro", df["gt_area"].gt(0) | df["pred_area"].gt(0)))
    macro.update(_macro_subset(df, "gt_positive_macro", df["gt_area"].gt(0)))
    macro.update(
        {
            "num_images": int(len(df)),
            "num_gt_positive": int(df["gt_area"].gt(0).sum()),
            "num_gt_negative": int(df["gt_area"].eq(0).sum()),
            "num_true_negative": int(((df["gt_area"] == 0) & (df["pred_area"] == 0)).sum()),
            "num_false_positive_images": int(((df["gt_area"] == 0) & (df["pred_area"] > 0)).sum()),
            "num_false_negative_images": int(((df["gt_area"] > 0) & (df["pred_area"] == 0)).sum()),
        }
    )
    micro.update(macro)
    return micro


def _macro_subset(df: pd.DataFrame, prefix: str, mask: pd.Series) -> Dict[str, float]:
    subset = df.loc[mask]
    if subset.empty:
        return {
            f"{prefix}_precision": float("nan"),
            f"{prefix}_recall": float("nan"),
            f"{prefix}_f1": float("nan"),
            f"{prefix}_iou": float("nan"),
            f"{prefix}_dice": float("nan"),
        }
    return {
        f"{prefix}_precision": float(subset["precision"].mean()),
        f"{prefix}_recall": float(subset["recall"].mean()),
        f"{prefix}_f1": float(subset["f1"].mean()),
        f"{prefix}_iou": float(subset["iou"].mean()),
        f"{prefix}_dice": float(subset["dice"].mean()),
    }


def compute_counts(tp: int, fp: int, fn: int, tn: int, eps: float = 1e-8) -> Dict[str, float]:
    if tp == 0 and fp == 0 and fn == 0:
        precision = recall = f1 = iou = dice = 1.0
    else:
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
    return {
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(f1),
        "micro_iou": float(iou),
        "micro_dice": float(dice),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

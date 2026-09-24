from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _safe_train_test_split(items, test_size, seed, stratify=None):
    if len(items) <= 2:
        return list(items), []
    try:
        train, test = train_test_split(items, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify)
    except ValueError:
        train, test = train_test_split(items, test_size=test_size, random_state=seed, shuffle=True, stratify=None)
    return list(train), list(test)


def split_dataframe(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_col: str | None = None,
    stratify_col: str = "bucket",
) -> pd.DataFrame:
    total = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total
    df = df.copy().reset_index(drop=True)

    if group_col and group_col in df.columns:
        # Split groups, using each group's dominant bucket for approximate stratification.
        group_df = df.groupby(group_col)[stratify_col].agg(lambda x: x.value_counts().index[0]).reset_index()
        groups = group_df[group_col].tolist()
        group_buckets = group_df[stratify_col].tolist()
        train_groups, temp_groups = _safe_train_test_split(
            groups, test_size=(1 - train_ratio), seed=seed, stratify=group_buckets
        )
        temp_df = group_df[group_df[group_col].isin(temp_groups)]
        if len(temp_groups) > 1:
            val_fraction_in_temp = val_ratio / (val_ratio + test_ratio)
            val_groups, test_groups = _safe_train_test_split(
                temp_df[group_col].tolist(),
                test_size=(1 - val_fraction_in_temp),
                seed=seed + 1,
                stratify=temp_df[stratify_col].tolist(),
            )
        else:
            val_groups, test_groups = temp_groups, []
        df["split"] = "train"
        df.loc[df[group_col].isin(val_groups), "split"] = "val"
        df.loc[df[group_col].isin(test_groups), "split"] = "test"
        return df

    indices = list(range(len(df)))
    train_idx, temp_idx = _safe_train_test_split(
        indices, test_size=(1 - train_ratio), seed=seed, stratify=df[stratify_col].tolist()
    )
    temp = df.iloc[temp_idx]
    if len(temp_idx) > 1:
        val_fraction_in_temp = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = _safe_train_test_split(
            temp_idx, test_size=(1 - val_fraction_in_temp), seed=seed + 1, stratify=temp[stratify_col].tolist()
        )
    else:
        val_idx, test_idx = temp_idx, []
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df

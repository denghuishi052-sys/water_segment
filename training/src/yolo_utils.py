from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import os

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_yolo_from_config(
    config_path: str | Path,
    overrides: Optional[Dict[str, Any]] = None,
):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ultralytics import YOLO

    cfg = load_yaml(config_path)
    if overrides:
        cfg.update(overrides)
    model_name = cfg.pop("model")
    model = YOLO(model_name)
    return model.train(**cfg)

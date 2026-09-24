from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from training.sam3_lora.data import (
    SAMPLING_RATIOS,
    build_manifest_records,
    validate_source_splits,
)
from waterseg_platform.config import PlatformConfig
from waterseg_platform.sam3_refinement import Sam3Refiner
from waterseg_platform.sam3_selector import (
    FEATURE_NAMES,
    ProposalSelector,
    sanitize_selector_feature,
    selector_feature_vector,
)


def test_lora_qv_only_disabled_parity_and_gradients() -> None:
    code = r"""
import torch
from torch import nn
from training.sam3_lora.lora import LoRAQKVLinear, inject_vision_qv_lora
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 12)
    def forward(self, x):
        return self.qkv(x)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.block = Block()
    def forward(self, x):
        return self.backbone.block(x)
torch.manual_seed(0)
model = Model()
x = torch.randn(2, 4)
expected = model(x).detach()
targets = inject_vision_qv_lora(model, rank=2, alpha=4, dropout=0)
actual = model(x)
assert targets == ["backbone.block.qkv"]
assert torch.allclose(actual, expected)
actual.square().mean().backward()
wrapper = model.backbone.block.qkv
assert isinstance(wrapper, LoRAQKVLinear)
assert wrapper.q_b.weight.grad is not None
assert wrapper.v_b.weight.grad is not None
assert wrapper.base.weight.grad is None
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_lora_save_load_round_trip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base checkpoint")
    code = rf"""
import torch
from torch import nn
from training.sam3_lora.lora import inject_vision_qv_lora, load_lora_adapter, save_lora_adapter
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 12)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.block = Block()
model = Model()
targets = inject_vision_qv_lora(model, rank=2, alpha=4, dropout=0)
with torch.no_grad():
    model.backbone.block.qkv.q_b.weight.fill_(0.25)
output = save_lora_adapter(
    model, {str(tmp_path / "adapter")!r},
    base_checkpoint={str(checkpoint)!r}, target_modules=targets,
    rank=2, alpha=4, dropout=0,
)
fresh = Model()
load_lora_adapter(fresh, output, base_checkpoint={str(checkpoint)!r})
assert torch.allclose(
    fresh.backbone.block.qkv.q_b.weight,
    model.backbone.block.qkv.q_b.weight,
)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _write_pair(root: Path, split: str, stem: str, positive: bool) -> None:
    image_dir = root / "images" / split
    mask_dir = root / "masks" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_dir / f"{stem}.jpg"), np.zeros((20, 20, 3), np.uint8))
    mask = np.zeros((20, 20), np.uint8)
    if positive:
        mask[5:10, 5:10] = 255
    cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)


def test_manifest_seals_test_and_preserves_sampling_ratios(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    rows = [
        ("train", "a", "a_pos", "foreground", 25, 0.0625),
        ("train", "b", "b_boundary", "boundary", 25, 0.0625),
        ("train", "c", "c_empty", "global", 0, 0),
        ("train", "d", "d_small", "foreground", 1, 0.0025),
        ("val", "e", "e_val", "global", 25, 0.0625),
        ("test", "f", "f_test", "global", 25, 0.0625),
    ]
    for split, _, stem, _, area, _ in rows:
        if split != "test":
            _write_pair(processed, split, stem, area > 0)
    report = tmp_path / "report.csv"
    fields = [
        "split",
        "source_stem",
        "output_stem",
        "crop_mode",
        "crop_foreground_area",
        "crop_foreground_ratio",
    ]
    with report.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for values in rows:
            writer.writerow(dict(zip(fields, values)))
    records, summary = build_manifest_records(report, processed)
    assert all(record["split"] != "test" for record in records)
    assert summary["test_images_read"] == 0
    weight_by_bucket = {
        record["sampling_bucket"]: record["sample_weight"]
        for record in records
        if record["split"] == "train"
    }
    assert weight_by_bucket == SAMPLING_RATIOS


def test_source_leakage_is_rejected() -> None:
    rows = [
        {"split": "train", "source_stem": "same"},
        {"split": "val", "source_stem": "same"},
    ]
    try:
        validate_source_splits(rows)
    except ValueError as exc:
        assert "leakage" in str(exc)
    else:
        raise AssertionError("Expected source leakage failure")


def test_selector_json_standardization_and_threshold(tmp_path: Path) -> None:
    path = tmp_path / "selector.json"
    payload = {
        "format": "waterseg_proposal_selector_v1",
        "feature_names": FEATURE_NAMES,
        "mean": [0.0] * len(FEATURE_NAMES),
        "scale": [1.0] * len(FEATURE_NAMES),
        "weights": [1.0] + [0.0] * (len(FEATURE_NAMES) - 1),
        "bias": 0.0,
        "accept_threshold": 0.7,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    selector = ProposalSelector.load(path)
    vector = selector_feature_vector(
        {"score": 1.0},
        source="text_box",
        candidate_area_ratio=0.1,
        yolo_area_ratio=0.2,
        yolo_confidence=0.25,
    )
    accepted, probability = selector.accepts(vector)
    assert accepted
    assert np.isclose(probability, 1 / (1 + np.exp(-1)))


def test_selector_sanitizes_nonfinite_features(tmp_path: Path) -> None:
    assert sanitize_selector_feature("distance_ratio", float("inf")) == 2.0
    vector = selector_feature_vector(
        {"score": 1.0, "distance_ratio": float("inf")},
        source="global_text",
        candidate_area_ratio=0.0,
        yolo_area_ratio=0.0,
    )
    assert np.isfinite(vector).all()
    assert vector[FEATURE_NAMES.index("distance_ratio")] == 2.0
    payload = {
        "format": "waterseg_proposal_selector_v1",
        "feature_names": FEATURE_NAMES,
        "mean": [0.0] * len(FEATURE_NAMES),
        "scale": [1.0] * len(FEATURE_NAMES),
        "weights": [0.0] * len(FEATURE_NAMES),
        "bias": 0.0,
        "accept_threshold": 0.5,
    }
    path = tmp_path / "selector.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    selector = ProposalSelector.load(path)
    accepted, probability = selector.accepts(
        np.full(len(FEATURE_NAMES), float("inf"), dtype=np.float64)
    )
    assert accepted
    assert probability == 0.5


def test_selector_load_error_returns_original_yolo(tmp_path: Path) -> None:
    yolo = np.zeros((20, 20), dtype=np.uint8)
    yolo[5:10, 5:10] = 1
    refiner = Sam3Refiner(
        PlatformConfig(
            sam3_selector_enabled=True,
            sam3_selector_path=str(tmp_path / "missing.json"),
        ),
        backend_factory=lambda config: object(),
    )
    refined, info = refiner.refine(
        np.zeros((20, 20, 3), dtype=np.uint8),
        yolo,
        enabled=True,
        mode="conservative",
    )
    assert np.array_equal(refined, yolo)
    assert info["fallback_stage"] == "yolo"
    assert info["selector_error"]

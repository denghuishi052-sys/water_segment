"""Configuration loader for the ONNX segmentation platform."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


SAM3_MODES = ("conservative", "balanced", "open")


@dataclass
class PlatformConfig:
    """Single source of truth for inference-time parameters."""

    model_path: str = "onnx/floodnet_binary_aug_yolov8m_1024.onnx"

    imgsz: int = 1024
    conf: float = 0.25
    iou: float = 0.5
    mask_thres: float = 0.5
    mask_box_expand_ratio: float = 0.0
    min_area_ratio: float = 0.0005
    morph_close: bool = True
    max_det: int = 300
    cascade_enabled: bool = True
    cascade_model_paths: List[str] = field(
        default_factory=lambda: [
            "onnx/waterlogging_yolov8m_base_640.onnx",
            "onnx/waterlogging_yolov8m_hard_finetune_640.onnx",
        ]
    )
    cascade_imgsz: int = 640
    cascade_conf: float = 0.15
    cascade_min_aspect_ratio: float = 1.15
    cascade_trigger_area_ratio: float = 0.005
    cascade_min_consensus_area_ratio: float = 0.005
    nc: int = 1
    nm: int = 32
    # Empty preserves the legacy binary decoder. For a multi-class model,
    # provide the class IDs that should be emitted as flood.
    target_class_ids: List[int] = field(default_factory=list)

    tile_size: int = 1024
    tile_overlap_px: int = 256
    tile_weight: float = 0.7
    tile_stitch_mode: str = "max"
    tile_weight_mode: str = "hann"
    tile_edge_weight: float = 0.3

    global_context_enabled: bool = True
    global_context_strength: float = 0.5

    border_suppression_enabled: bool = True
    border_margin_ratio: float = 0.01
    border_margin_min_px: int = 12
    border_touch_ratio_thresh: float = 0.35
    border_max_component_area_ratio: float = 0.08

    # SAM 3 is opt-in. YOLO remains the production-safe default.
    sam3_enabled: bool = False
    sam3_mode: str = "conservative"
    sam3_checkpoint: str = "D:/BaiduNetdiskDownload/课程- 权重(1)/sam3.pt"
    sam3_device: str = "cuda"
    sam3_use_child_process: bool = True
    sam3_compile: bool = False
    sam3_confidence: float = 0.5

    sam3_local_refine_enabled: bool = True
    sam3_prompt_strategy: str = "text_box_points"
    sam3_box_margin_ratio: float = 0.15
    sam3_min_component_area_ratio: float = 0.0003
    sam3_max_candidates: int = 12
    sam3_positive_points_per_component: int = 5
    sam3_negative_points_per_component: int = 8
    sam3_negative_ring_ratio: float = 0.20
    sam3_local_score_thresh: float = 0.25

    sam3_global_text_enabled: bool = True
    sam3_global_max_side: int = 1280
    sam3_global_score_thresh: float = 0.35
    sam3_global_max_area_ratio: float = 0.35

    sam3_min_yolo_overlap: float = 0.15
    sam3_min_yolo_coverage: float = 0.30
    sam3_balanced_max_distance_ratio: float = 0.15
    sam3_max_area_growth_conservative: float = 2.5
    sam3_max_area_growth_balanced: float = 4.0
    sam3_max_area_growth_open: float = 8.0
    sam3_text_prompts: List[str] = field(
        default_factory=lambda: [
            "standing water",
            "flood water",
            "flooded road",
            "water-covered road",
            "puddle on road",
            "urban waterlogging",
            "inundated pavement",
        ]
    )

    # Domain adapter and learned proposal selector remain opt-in.
    sam3_adapter_enabled: bool = False
    sam3_adapter_type: str = "qv_lora"
    sam3_lora_path: str = ""
    sam3_lora_rank: int = 4
    sam3_lora_alpha: float = 16.0
    sam3_lora_dropout: float = 0.05
    sam3_lora_target_modules: List[str] = field(default_factory=list)
    sam3_adapter_strict_fingerprint: bool = True
    sam3_selector_enabled: bool = False
    sam3_selector_path: str = ""
    sam3_selector_accept_threshold: Optional[float] = None

    sam3_debug_dir: str = ""
    sam3_save_debug_proposals: bool = False

    providers: List[str] = field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    bgr_input: bool = True

    def __post_init__(self) -> None:
        self.sam3_mode = str(self.sam3_mode).lower()
        if self.sam3_mode not in SAM3_MODES:
            raise ValueError(
                "sam3_mode must be one of: conservative, balanced, open"
            )
        self.tile_stitch_mode = str(self.tile_stitch_mode).lower()
        if self.tile_stitch_mode not in {"average", "max"}:
            raise ValueError("tile_stitch_mode must be one of: average, max")
        self.tile_weight_mode = str(self.tile_weight_mode).lower()
        if self.tile_weight_mode not in {"uniform", "hann"}:
            raise ValueError("tile_weight_mode must be one of: uniform, hann")
        self.tile_edge_weight = float(min(max(self.tile_edge_weight, 0.0), 1.0))
        self.global_context_strength = float(
            min(max(self.global_context_strength, 0.0), 1.0)
        )
        self.border_margin_ratio = float(max(self.border_margin_ratio, 0.0))
        self.border_margin_min_px = int(max(self.border_margin_min_px, 0))
        self.border_touch_ratio_thresh = float(
            min(max(self.border_touch_ratio_thresh, 0.0), 1.0)
        )
        self.border_max_component_area_ratio = float(
            min(max(self.border_max_component_area_ratio, 0.0), 1.0)
        )
        self.mask_box_expand_ratio = float(
            min(max(self.mask_box_expand_ratio, 0.0), 2.0)
        )
        self.target_class_ids = [
            int(class_id) for class_id in self.target_class_ids if int(class_id) >= 0
        ]

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Optional[str] = None) -> PlatformConfig:
    """Load a config file, using dataclass defaults for omitted fields."""
    defaults = PlatformConfig()
    if path is None:
        return defaults
    config_path = Path(path)
    if not config_path.exists():
        return defaults
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    # Migrate the original two-mode config without keeping two sources of truth.
    if "sam3_mode" not in raw and "sam3_conservative" in raw:
        raw["sam3_mode"] = (
            "conservative" if raw["sam3_conservative"] else "open"
        )
    valid_keys = set(defaults.to_dict())
    filtered = {key: value for key, value in raw.items() if key in valid_keys}
    return PlatformConfig(**filtered)


def dump_config(cfg: PlatformConfig, path: str) -> None:
    """Write a config as YAML."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            cfg.to_dict(),
            file,
            sort_keys=False,
            allow_unicode=True,
        )

from pathlib import Path

import numpy as np

from waterseg_platform.config import PlatformConfig, load_config
from waterseg_platform.pipeline import SegmentationService


NEW_MODEL_PATH = "onnx/floodnet_binary_aug_yolov8m_1024.onnx"


def assert_binary_1024_defaults(cfg: PlatformConfig) -> None:
    assert cfg.model_path == NEW_MODEL_PATH
    assert cfg.imgsz == 1024
    assert cfg.tile_size == 1024
    assert cfg.nc == 1
    assert cfg.nm == 32


def test_platform_config_defaults_to_latest_binary_model() -> None:
    cfg = PlatformConfig()
    assert_binary_1024_defaults(cfg)
    assert cfg.cascade_enabled is True
    assert cfg.tile_weight == 0.7
    assert cfg.tile_stitch_mode == "max"
    assert cfg.tile_weight_mode == "hann"
    assert cfg.tile_edge_weight == 0.3
    assert cfg.global_context_enabled is True
    assert cfg.global_context_strength == 0.5
    assert cfg.border_suppression_enabled is True
    assert cfg.border_margin_ratio == 0.01
    assert cfg.border_margin_min_px == 12
    assert cfg.border_touch_ratio_thresh == 0.35
    assert cfg.border_max_component_area_ratio == 0.08
    assert cfg.sam3_enabled is False
    assert cfg.sam3_mode == "conservative"
    assert cfg.sam3_prompt_strategy == "text_box_points"
    assert cfg.sam3_box_margin_ratio == 0.15
    assert cfg.sam3_min_component_area_ratio == 0.0003
    assert cfg.sam3_max_candidates == 12
    assert cfg.sam3_positive_points_per_component == 5
    assert cfg.sam3_negative_points_per_component == 8
    assert cfg.sam3_global_score_thresh == 0.35
    assert cfg.sam3_max_area_growth_balanced == 4.0
    assert cfg.sam3_adapter_enabled is False
    assert cfg.sam3_selector_enabled is False
    assert cfg.sam3_lora_rank == 4
    assert len(cfg.sam3_text_prompts) == 7


def test_yaml_and_trial_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    production = load_config(str(root / "configs" / "onnx_platform.yaml"))
    trial = load_config(
        str(root / "configs" / "onnx_platform_sam3_trial.yaml")
    )
    assert_binary_1024_defaults(production)
    assert production.sam3_enabled is False
    assert trial.sam3_enabled is True
    assert trial.sam3_mode == "conservative"


def test_old_two_mode_yaml_is_migrated(tmp_path) -> None:
    path = tmp_path / "old.yaml"
    path.write_text("sam3_conservative: false\n", encoding="utf-8")
    assert load_config(str(path)).sam3_mode == "open"


class RecordingEngine:
    providers_active = ["CPUExecutionProvider"]
    input_shape = (1, 3, 1024, 1024)
    output0_shape = (1, 37, 1)
    output1_shape = (1, 32, 256, 256)

    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask
        self.kwargs = {}

    def predict_mask(self, image, **kwargs):
        self.kwargs = kwargs
        mask = self.mask.copy()
        return mask, {
            "providers": self.providers_active,
            "pred_area": int(mask.sum()),
            "pred_area_ratio": float(mask.mean()),
        }

    def run(self, tensor):
        return (
            np.zeros((1, 37, 1), dtype=np.float32),
            np.zeros((1, 32, 256, 256), dtype=np.float32),
        )


class RecordingRefiner:
    def __init__(self, refined_mask: np.ndarray) -> None:
        self.refined_mask = refined_mask
        self.calls = []

    def refine(self, image, mask, *, enabled, mode):
        self.calls.append({"enabled": enabled, "mode": mode})
        return self.refined_mask.copy(), {
            "enabled": enabled,
            "ran": True,
            "mode": mode,
            "accepted": 1,
            "rejected": 0,
            "fallback": False,
        }


def _service(primary, refined=None, sam3_enabled=True):
    service = SegmentationService.__new__(SegmentationService)
    service.config = PlatformConfig(
        cascade_enabled=False, sam3_enabled=sam3_enabled
    )
    service.engine = RecordingEngine(primary)
    service._cascade_engines = None
    service._sam3_refiner = (
        RecordingRefiner(refined) if refined is not None else None
    )
    return service


def test_segmentation_service_passes_model_layout_to_engine() -> None:
    service = _service(np.zeros((8, 8), dtype=np.uint8), sam3_enabled=False)
    service.segment_array(np.zeros((8, 8, 3), dtype=np.uint8))
    assert service.engine.kwargs["nc"] == 1
    assert service.engine.kwargs["nm"] == 32


def test_segment_array_dispatches_all_sam3_modes() -> None:
    primary = np.zeros((12, 12), dtype=np.uint8)
    primary[4:8, 4:8] = 1
    refined = primary.copy()
    refined[3, 4:8] = 1
    for mode in ("conservative", "balanced", "open"):
        service = _service(primary, refined)
        mask, info = service.segment_array(
            np.zeros((12, 12, 3), dtype=np.uint8),
            sam3_mode=mode,
        )
        assert np.array_equal(mask, refined)
        assert service._sam3_refiner.calls == [
            {"enabled": True, "mode": mode}
        ]
        assert info["sam3"]["mode"] == mode


def test_segment_array_skips_sam3_with_per_call_override() -> None:
    primary = np.zeros((12, 12), dtype=np.uint8)
    service = _service(primary, np.ones_like(primary))
    mask, info = service.segment_array(
        np.zeros((12, 12, 3), dtype=np.uint8),
        sam3_enabled=False,
    )
    assert np.array_equal(mask, primary)
    assert service._sam3_refiner.calls == []
    assert info["sam3"]["enabled"] is False


def test_segment_array_tiled_applies_balanced_after_tiling() -> None:
    primary = np.zeros((32, 32), dtype=np.uint8)
    primary[10:20, 10:20] = 1
    refined = primary.copy()
    refined[9, 10:20] = 1
    service = _service(primary, refined)
    service.config.imgsz = 1024
    service.config.tile_size = 1024
    service.config.tile_overlap_px = 0
    mask, info = service.segment_array_tiled(
        np.zeros((32, 32, 3), dtype=np.uint8),
        sam3_mode="balanced",
    )
    assert np.array_equal(mask, refined)
    assert service._sam3_refiner.calls == [
        {"enabled": True, "mode": "balanced"}
    ]
    assert info["sam3"]["mode"] == "balanced"

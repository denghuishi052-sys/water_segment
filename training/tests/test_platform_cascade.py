import numpy as np

from waterseg_platform.config import PlatformConfig
from waterseg_platform.pipeline import SegmentationService
from waterseg_platform.pipeline import choose_cascade_mask, soft_fuse_masks


class FakeEngine:
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask
        self.calls = 0

    def predict_mask(self, image, **kwargs):
        self.calls += 1
        mask = self.mask.copy()
        return mask, {
            "pred_area": int(mask.sum()),
            "pred_area_ratio": float(mask.mean()),
        }


def test_cascade_does_not_run_when_primary_area_is_large() -> None:
    primary = np.ones((10, 10), dtype=np.uint8)
    fallback_a = np.zeros_like(primary)
    fallback_b = np.zeros_like(primary)

    mask, info = choose_cascade_mask(
        primary,
        fallback_a,
        fallback_b,
        trigger_area_ratio=0.005,
        min_consensus_area_ratio=0.005,
    )

    assert np.array_equal(mask, primary)
    assert info["triggered"] is False
    assert info["accepted"] is False


def test_cascade_accepts_conservative_fallback_intersection() -> None:
    primary = np.zeros((20, 20), dtype=np.uint8)
    fallback_a = np.zeros_like(primary)
    fallback_b = np.zeros_like(primary)
    fallback_a[2:12, 2:12] = 1
    fallback_b[5:15, 5:15] = 1

    mask, info = choose_cascade_mask(
        primary,
        fallback_a,
        fallback_b,
        trigger_area_ratio=0.005,
        min_consensus_area_ratio=0.005,
    )

    expected = np.logical_and(fallback_a, fallback_b).astype(np.uint8)
    assert np.array_equal(mask, expected)
    assert info["triggered"] is True
    assert info["accepted"] is True
    assert info["consensus_area_ratio"] == expected.mean()


def test_cascade_rejects_tiny_fallback_consensus() -> None:
    primary = np.zeros((100, 100), dtype=np.uint8)
    fallback_a = np.zeros_like(primary)
    fallback_b = np.zeros_like(primary)
    fallback_a[0:5, 0:5] = 1
    fallback_b[0:5, 0:5] = 1

    mask, info = choose_cascade_mask(
        primary,
        fallback_a,
        fallback_b,
        trigger_area_ratio=0.005,
        min_consensus_area_ratio=0.005,
    )

    assert np.array_equal(mask, primary)
    assert info["triggered"] is True
    assert info["accepted"] is False


def test_soft_fusion_allows_strong_tile_outside_empty_coarse_mask() -> None:
    coarse = np.zeros((4, 4), dtype=np.uint8)
    tiled_probability = np.zeros((4, 4), dtype=np.float32)
    tiled_probability[1:3, 1:3] = 0.9

    fused = soft_fuse_masks(
        coarse,
        tiled_probability,
        tile_weight=0.7,
        mask_thres=0.5,
    )

    assert fused[1:3, 1:3].all()
    assert fused.sum() == 4


def test_segment_array_uses_accepted_fallback_consensus() -> None:
    primary = np.zeros((20, 30), dtype=np.uint8)
    fallback_a = np.zeros_like(primary)
    fallback_b = np.zeros_like(primary)
    fallback_a[2:15, 2:15] = 1
    fallback_b[5:18, 5:18] = 1

    service = SegmentationService.__new__(SegmentationService)
    service.config = PlatformConfig(
        cascade_enabled=True,
        cascade_trigger_area_ratio=0.005,
        cascade_min_consensus_area_ratio=0.005,
    )
    service.engine = FakeEngine(primary)
    service._cascade_engines = (FakeEngine(fallback_a), FakeEngine(fallback_b))

    mask, info = service.segment_array(np.zeros((20, 30, 3), dtype=np.uint8))

    assert np.array_equal(mask, np.logical_and(fallback_a, fallback_b))
    assert info["cascade"]["triggered"] is True
    assert info["cascade"]["accepted"] is True
    assert info["pred_area"] == int(mask.sum())


def test_segment_array_skips_fallback_when_primary_is_large() -> None:
    primary = np.ones((20, 30), dtype=np.uint8)
    fallback_a = FakeEngine(np.zeros_like(primary))
    fallback_b = FakeEngine(np.zeros_like(primary))

    service = SegmentationService.__new__(SegmentationService)
    service.config = PlatformConfig(cascade_enabled=True)
    service.engine = FakeEngine(primary)
    service._cascade_engines = (fallback_a, fallback_b)

    mask, info = service.segment_array(np.zeros((20, 30, 3), dtype=np.uint8))

    assert np.array_equal(mask, primary)
    assert fallback_a.calls == fallback_b.calls == 0
    assert info["cascade"]["triggered"] is False


def test_segment_array_skips_fallback_when_cascade_is_disabled() -> None:
    primary = np.zeros((20, 30), dtype=np.uint8)
    fallback_a = FakeEngine(np.ones_like(primary))
    fallback_b = FakeEngine(np.ones_like(primary))

    service = SegmentationService.__new__(SegmentationService)
    service.config = PlatformConfig(cascade_enabled=False)
    service.engine = FakeEngine(primary)
    service._cascade_engines = (fallback_a, fallback_b)

    mask, info = service.segment_array(np.zeros((20, 30, 3), dtype=np.uint8))

    assert np.array_equal(mask, primary)
    assert fallback_a.calls == fallback_b.calls == 0
    assert info["cascade"]["enabled"] is False


def test_segment_array_skips_road_cascade_for_square_aerial_image() -> None:
    primary = np.zeros((20, 20), dtype=np.uint8)
    fallback_a = FakeEngine(np.ones_like(primary))
    fallback_b = FakeEngine(np.ones_like(primary))

    service = SegmentationService.__new__(SegmentationService)
    service.config = PlatformConfig(
        cascade_enabled=True,
        cascade_min_aspect_ratio=1.15,
    )
    service.engine = FakeEngine(primary)
    service._cascade_engines = (fallback_a, fallback_b)

    mask, info = service.segment_array(np.zeros((20, 20, 3), dtype=np.uint8))

    assert np.array_equal(mask, primary)
    assert fallback_a.calls == fallback_b.calls == 0
    assert info["cascade"]["aspect_eligible"] is False

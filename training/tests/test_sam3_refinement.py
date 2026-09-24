from __future__ import annotations

import numpy as np

from waterseg_platform.cli import _build_config, build_parser
from waterseg_platform.config import PlatformConfig
from waterseg_platform.sam3_refinement import (
    Sam3Proposal,
    Sam3Refiner,
    extract_candidates,
    fuse_sam3_proposals,
    proposal_features,
)


def _candidate(mask: np.ndarray):
    return extract_candidates(
        mask,
        min_component_area_ratio=0.0,
        max_candidates=12,
        box_margin_ratio=0.5,
        positive_points_per_component=5,
        negative_points_per_component=8,
        negative_ring_ratio=0.2,
    )[0]


def test_candidate_points_are_valid_and_spread() -> None:
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[20:40, 25:55] = 1
    candidate = _candidate(mask)

    assert 1 <= len(candidate.positive_points) <= 5
    assert 1 <= len(candidate.negative_points) <= 8
    assert len(set(candidate.positive_points)) == len(candidate.positive_points)
    assert len(set(candidate.negative_points)) == len(candidate.negative_points)
    assert all(mask[y, x] == 1 for x, y in candidate.positive_points)
    assert all(mask[y, x] == 0 for x, y in candidate.negative_points)
    x1, y1, x2, y2 = candidate.box_xyxy
    assert all(
        x1 <= x < x2 and y1 <= y < y2
        for x, y in candidate.negative_points
    )


def test_extract_candidates_sorts_filters_and_limits() -> None:
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[0:4, 0:5] = 1
    mask[10:16, 20:28] = 1
    mask[18:20, 28:30] = 1
    candidates = extract_candidates(mask, 0.02, 2, 0.25)

    assert [candidate.area for candidate in candidates] == [48, 20]
    assert candidates[0].box_xyxy == (18, 8, 30, 18)
    assert candidates[1].box_xyxy == (0, 0, 7, 5)


def test_proposal_features_report_fusion_evidence() -> None:
    reference = np.zeros((20, 20), dtype=np.uint8)
    reference[8:12, 8:12] = 1
    proposal = np.zeros_like(reference)
    proposal[7:13, 7:13] = 1

    features = proposal_features(proposal, reference, score=0.8)

    assert features["score"] == 0.8
    assert features["coverage"] == 1.0
    assert 0.0 < features["overlap"] < 1.0
    assert features["area_growth"] == 36 / 16
    assert features["distance_ratio"] == 0.0


def test_conservative_replaces_only_supported_component_and_keeps_other_yolo() -> None:
    yolo = np.zeros((40, 40), dtype=np.uint8)
    yolo[5:10, 5:10] = 1
    yolo[30:32, 30:32] = 1
    candidates = extract_candidates(yolo, 0.01, 1, 0.5)
    refined = np.zeros_like(yolo)
    refined[4:11, 4:11] = 1
    config = PlatformConfig(
        sam3_local_score_thresh=0.1,
        sam3_min_yolo_overlap=0.1,
        sam3_min_yolo_coverage=0.3,
        sam3_max_area_growth_conservative=3.0,
    )

    fused, info = fuse_sam3_proposals(
        yolo,
        candidates,
        [Sam3Proposal(refined, 0.9, "text_box", 0, "standing water")],
        mode="conservative",
        config=config,
    )

    assert fused[30:32, 30:32].all()
    assert fused[4:11, 4:11].all()
    assert info["accepted"] == 1


def test_conservative_rejects_remote_local_mask() -> None:
    yolo = np.zeros((40, 40), dtype=np.uint8)
    yolo[5:10, 5:10] = 1
    remote = np.zeros_like(yolo)
    remote[25:30, 25:30] = 1
    config = PlatformConfig()

    fused, info = fuse_sam3_proposals(
        yolo,
        [_candidate(yolo)],
        [Sam3Proposal(remote, 0.99, "interactive_points", 0)],
        mode="conservative",
        config=config,
    )

    assert np.array_equal(fused, yolo)
    assert info["proposal_decisions"][0]["reason"] in {
        "empty_mask",
        "low_overlap",
    }


def test_balanced_accepts_nearby_global_and_rejects_far_global() -> None:
    yolo = np.zeros((100, 100), dtype=np.uint8)
    yolo[45:55, 45:55] = 1
    nearby = np.zeros_like(yolo)
    nearby[55:60, 45:55] = 1
    far = np.zeros_like(yolo)
    far[0:5, 0:5] = 1
    config = PlatformConfig(
        sam3_global_score_thresh=0.3,
        sam3_balanced_max_distance_ratio=0.1,
        sam3_max_area_growth_balanced=4.0,
    )

    fused, info = fuse_sam3_proposals(
        yolo,
        [_candidate(yolo)],
        [
            Sam3Proposal(nearby, 0.9, "global_text", prompt="flood water"),
            Sam3Proposal(far, 0.9, "global_text", prompt="flood water"),
        ],
        mode="balanced",
        config=config,
    )

    assert fused[55:60, 45:55].all()
    assert not fused[0:5, 0:5].any()
    assert [item["reason"] for item in info["proposal_decisions"]] == [
        None,
        "too_far_from_yolo",
    ]


def test_open_accepts_remote_proposal_but_rejects_excessive_area() -> None:
    yolo = np.zeros((20, 20), dtype=np.uint8)
    remote = np.zeros_like(yolo)
    remote[1:4, 1:4] = 1
    huge = np.ones_like(yolo)
    config = PlatformConfig(sam3_global_max_area_ratio=0.35)

    fused, info = fuse_sam3_proposals(
        yolo,
        [],
        [
            Sam3Proposal(remote, 0.9, "global_text", prompt="puddle on road"),
            Sam3Proposal(huge, 0.9, "global_text", prompt="flood water"),
        ],
        mode="open",
        config=config,
    )

    assert fused[1:4, 1:4].all()
    assert info["proposal_decisions"][1]["reason"] == "excessive_image_area"


class FakeBackend:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_refiner_passes_text_box_points_and_mode_to_backend() -> None:
    yolo = np.zeros((30, 30), dtype=np.uint8)
    yolo[10:20, 10:20] = 1
    proposal = np.zeros_like(yolo)
    proposal[9:21, 9:21] = 1
    backend = FakeBackend(
        {
            "proposals": [
                {
                    "mask": proposal,
                    "score": 0.9,
                    "source": "interactive_points",
                    "candidate_index": 0,
                    "prompt": None,
                }
            ]
        }
    )
    refiner = Sam3Refiner(
        PlatformConfig(
            sam3_min_yolo_overlap=0.1,
            sam3_min_yolo_coverage=0.3,
            sam3_max_area_growth_conservative=3.0,
        ),
        backend_factory=lambda config: backend,
    )

    refined, info = refiner.refine(
        np.zeros((30, 30, 3), dtype=np.uint8),
        yolo,
        enabled=True,
        mode="conservative",
    )

    assert backend.calls[0]["mode"] == "conservative"
    candidate = backend.calls[0]["candidates"][0]
    assert candidate.positive_points
    assert candidate.negative_points
    assert refined.sum() >= yolo.sum()
    assert info["fallback"] is False


def test_refiner_records_text_box_fallback_when_points_fail() -> None:
    yolo = np.zeros((20, 20), dtype=np.uint8)
    yolo[5:10, 5:10] = 1
    backend = FakeBackend(
        {
            "interactive_error": "RuntimeError: points unavailable",
            "proposals": [
                {
                    "mask": yolo,
                    "score": 0.9,
                    "source": "text_box",
                    "candidate_index": 0,
                    "prompt": "standing water",
                }
            ],
        }
    )
    refiner = Sam3Refiner(
        PlatformConfig(), backend_factory=lambda config: backend
    )

    refined, info = refiner.refine(
        np.zeros((20, 20, 3), dtype=np.uint8),
        yolo,
        enabled=True,
        mode="conservative",
    )

    assert np.array_equal(refined, yolo)
    assert info["fallback_stage"] == "text_box"
    assert info["fallback"] is False


def test_refiner_returns_yolo_when_all_sam_branches_fail() -> None:
    yolo = np.zeros((20, 20), dtype=np.uint8)
    yolo[5:10, 5:10] = 1
    backend = FakeBackend(
        {
            "pcs_error": "PCS failed",
            "interactive_error": "points failed",
            "proposals": [],
        }
    )
    refiner = Sam3Refiner(
        PlatformConfig(), backend_factory=lambda config: backend
    )

    refined, info = refiner.refine(
        np.zeros((20, 20, 3), dtype=np.uint8),
        yolo,
        enabled=True,
        mode="balanced",
    )

    assert np.array_equal(refined, yolo)
    assert info["fallback"] is True
    assert info["fallback_stage"] == "yolo"


def test_cli_sam3_mode_overrides_config() -> None:
    balanced = build_parser().parse_args(
        ["--sam3", "--sam3_mode", "balanced"]
    )
    open_alias = build_parser().parse_args(["--sam3", "--sam3_open"])

    assert _build_config(balanced).sam3_mode == "balanced"
    assert _build_config(open_alias).sam3_mode == "open"


def test_cli_without_sam3_mode_preserves_yaml_default() -> None:
    args = build_parser().parse_args(
        ["--config", "configs/onnx_platform_sam3_trial.yaml"]
    )
    config = _build_config(args)
    assert config.sam3_enabled is True
    assert config.sam3_mode == "conservative"

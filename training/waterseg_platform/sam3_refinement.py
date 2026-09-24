"""YOLO-guided SAM 3 prompting, proposal scoring, and safe fusion."""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

VALID_MODES = ("conservative", "balanced", "open")


@dataclass(frozen=True)
class Sam3Candidate:
    """One YOLO component and all prompts derived from it."""

    component_mask: np.ndarray
    box_xyxy: Tuple[int, int, int, int]
    box_cxcywh: Tuple[float, float, float, float]
    positive_points: Tuple[Tuple[int, int], ...]
    negative_points: Tuple[Tuple[int, int], ...]
    area: int


@dataclass
class Sam3Proposal:
    """One mask proposal returned by a SAM 3 prompting branch."""

    mask: np.ndarray
    score: float
    source: str
    candidate_index: Optional[int] = None
    prompt: Optional[str] = None


def _spread_points(
    valid_mask: np.ndarray,
    count: int,
    priority: Optional[np.ndarray] = None,
) -> Tuple[Tuple[int, int], ...]:
    """Select deterministic, spatially spread points from a binary region."""
    valid = np.asarray(valid_mask) > 0
    coordinates = np.argwhere(valid)
    if count <= 0 or len(coordinates) == 0:
        return ()

    if priority is None:
        priority = cv2.distanceTransform(
            valid.astype(np.uint8), cv2.DIST_L2, 5
        )
    first = coordinates[
        int(np.argmax(priority[coordinates[:, 0], coordinates[:, 1]]))
    ]
    chosen = [first.astype(np.float32)]
    while len(chosen) < min(int(count), len(coordinates)):
        points = np.stack(chosen)
        squared = (
            (coordinates[:, None, :] - points[None, :, :]) ** 2
        ).sum(axis=2)
        min_squared = squared.min(axis=1)
        # Prefer distance-transform maxima when separation is tied.
        rank = min_squared + 0.01 * priority[
            coordinates[:, 0], coordinates[:, 1]
        ]
        next_point = coordinates[int(np.argmax(rank))].astype(np.float32)
        if any(np.array_equal(next_point, point) for point in chosen):
            break
        chosen.append(next_point)
    return tuple((int(point[1]), int(point[0])) for point in chosen)


def sample_prompt_points(
    component_mask: np.ndarray,
    box_xyxy: Tuple[int, int, int, int],
    positive_count: int,
    negative_count: int,
    negative_ring_ratio: float,
) -> tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]:
    """Sample foreground points and an exterior negative ring."""
    component = (np.asarray(component_mask) > 0).astype(np.uint8)
    positive = _spread_points(component, positive_count)

    x1, y1, x2, y2 = box_xyxy
    box_mask = np.zeros_like(component)
    box_mask[y1:y2, x1:x2] = 1
    radius = max(
        1,
        int(
            round(
                max(x2 - x1, y2 - y1)
                * max(float(negative_ring_ratio), 0.01)
            )
        ),
    )
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    dilated = cv2.dilate(component, kernel)
    ring = np.logical_and(dilated > 0, component == 0)
    ring = np.logical_and(ring, box_mask > 0).astype(np.uint8)
    if not ring.any():
        ring = np.logical_and(box_mask > 0, component == 0).astype(np.uint8)
    negative = _spread_points(ring, negative_count)
    return positive, negative


def extract_candidates(
    yolo_mask: np.ndarray,
    min_component_area_ratio: float,
    max_candidates: int,
    box_margin_ratio: float,
    positive_points_per_component: int = 5,
    negative_points_per_component: int = 8,
    negative_ring_ratio: float = 0.20,
) -> list[Sam3Candidate]:
    """Convert the largest YOLO connected components into SAM 3 prompts."""
    binary = (np.asarray(yolo_mask) > 0).astype(np.uint8)
    height, width = binary.shape[:2]
    min_area = float(height * width) * float(min_component_area_ratio)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    components = []
    for label in range(1, count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[label]
        )
        if area >= min_area:
            components.append(
                (area, label, x, y, component_width, component_height)
            )
    components.sort(key=lambda item: (-item[0], item[1]))

    candidates = []
    for area, label, x, y, component_width, component_height in components[
        : max(0, int(max_candidates))
    ]:
        margin_x = int(np.ceil(component_width * float(box_margin_ratio)))
        margin_y = int(np.ceil(component_height * float(box_margin_ratio)))
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(width, x + component_width + margin_x)
        y2 = min(height, y + component_height + margin_y)
        component = (labels == label).astype(np.uint8)
        positive, negative = sample_prompt_points(
            component,
            (x1, y1, x2, y2),
            positive_points_per_component,
            negative_points_per_component,
            negative_ring_ratio,
        )
        candidates.append(
            Sam3Candidate(
                component_mask=component,
                box_xyxy=(x1, y1, x2, y2),
                box_cxcywh=(
                    ((x1 + x2) * 0.5) / max(width, 1),
                    ((y1 + y2) * 0.5) / max(height, 1),
                    (x2 - x1) / max(width, 1),
                    (y2 - y1) / max(height, 1),
                ),
                positive_points=positive,
                negative_points=negative,
                area=area,
            )
        )
    return candidates


def _validated_mask(
    mask: np.ndarray, expected_shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    array = np.asarray(mask)
    if array.shape != expected_shape or not np.isfinite(array).all():
        return None
    return (array > 0).astype(np.uint8)


def _distance_ratio(mask: np.ndarray, reference: np.ndarray) -> float:
    proposal = np.asarray(mask) > 0
    target = np.asarray(reference) > 0
    if not proposal.any():
        return float("inf")
    if not target.any():
        return float("inf")
    if np.logical_and(proposal, target).any():
        return 0.0
    distance = cv2.distanceTransform(
        (~target).astype(np.uint8), cv2.DIST_L2, 5
    )
    diagonal = float(np.hypot(*target.shape))
    return float(distance[proposal].min()) / max(diagonal, 1.0)


def proposal_features(
    proposal_mask: np.ndarray,
    reference_mask: np.ndarray,
    score: float,
) -> dict:
    """Compute the evidence used by all fusion modes."""
    proposal = (np.asarray(proposal_mask) > 0).astype(np.uint8)
    reference = (np.asarray(reference_mask) > 0).astype(np.uint8)
    proposal_area = int(proposal.sum())
    reference_area = int(reference.sum())
    intersection = int(np.logical_and(proposal, reference).sum())
    union = int(np.logical_or(proposal, reference).sum())
    return {
        "score": float(score),
        "proposal_area": proposal_area,
        "reference_area": reference_area,
        "area_ratio": proposal_area / float(max(proposal.size, 1)),
        "overlap": intersection / float(max(proposal_area, 1)),
        "coverage": intersection / float(max(reference_area, 1)),
        "iou": intersection / float(max(union, 1)),
        "area_growth": proposal_area / float(max(reference_area, 1)),
        "distance_ratio": _distance_ratio(proposal, reference),
    }


def _max_growth(config, mode: str) -> float:
    return float(
        getattr(config, f"sam3_max_area_growth_{mode}")
    )


def fuse_sam3_proposals(
    yolo_mask: np.ndarray,
    candidates: Sequence[Sam3Candidate],
    proposals: Sequence[Sam3Proposal],
    *,
    mode: str,
    config,
    selector=None,
) -> tuple[np.ndarray, dict]:
    """Score SAM proposals and fuse only those allowed by the selected mode."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported SAM 3 mode: {mode}")
    yolo = (np.asarray(yolo_mask) > 0).astype(np.uint8)
    output = yolo.copy()
    decisions = []
    accepted_count = 0
    rejected_count = 0
    yolo_area_ratio = float(yolo.sum()) / float(max(yolo.size, 1))

    def apply_selector(features, source, candidate_area_ratio):
        if selector is None:
            return None, None
        from waterseg_platform.sam3_selector import selector_feature_vector

        vector = selector_feature_vector(
            features,
            source=source,
            candidate_area_ratio=candidate_area_ratio,
            yolo_area_ratio=yolo_area_ratio,
            yolo_confidence=float(getattr(config, "conf", 0.0)),
        )
        accepted, probability = selector.accepts(vector)
        return bool(accepted), float(probability)

    local_by_candidate: dict[int, list[Sam3Proposal]] = {}
    global_proposals = []
    for proposal in proposals:
        if proposal.candidate_index is None:
            global_proposals.append(proposal)
        else:
            local_by_candidate.setdefault(
                int(proposal.candidate_index), []
            ).append(proposal)

    for index, candidate in enumerate(candidates):
        accepted_options = []
        for proposal in local_by_candidate.get(index, []):
            mask = _validated_mask(proposal.mask, yolo.shape)
            reason = None
            if mask is None:
                features = {}
                reason = "malformed_mask"
            else:
                if mode == "conservative":
                    x1, y1, x2, y2 = candidate.box_xyxy
                    clipped = np.zeros_like(mask)
                    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                    mask = clipped
                features = proposal_features(
                    mask, candidate.component_mask, proposal.score
                )
                if not mask.any():
                    reason = "empty_mask"
                elif features["score"] < float(config.sam3_local_score_thresh):
                    reason = "low_score"
                elif features["overlap"] < float(config.sam3_min_yolo_overlap):
                    reason = "low_overlap"
                elif features["coverage"] < float(config.sam3_min_yolo_coverage):
                    reason = "low_coverage"
                elif features["area_growth"] > _max_growth(config, mode):
                    reason = "excessive_growth"
                elif (
                    mode == "balanced"
                    and features["distance_ratio"]
                    > float(config.sam3_balanced_max_distance_ratio)
                ):
                    reason = "too_far_from_yolo"
                if reason is None and selector is not None:
                    selected, probability = apply_selector(
                        features,
                        proposal.source,
                        candidate.area / float(max(yolo.size, 1)),
                    )
                    if not selected:
                        reason = "selector_rejected"
                else:
                    probability = None
            decision = {
                "source": proposal.source,
                "candidate_index": index,
                "prompt": proposal.prompt,
                "accepted": reason is None,
                "reason": reason,
                "selector_probability": probability if mask is not None else None,
                **features,
            }
            decisions.append(decision)
            if reason is None:
                quality = (
                    probability
                    if probability is not None
                    else (
                        features["score"]
                        + features["coverage"]
                        + features["iou"]
                        - 0.05 * features["area_growth"]
                    )
                )
                accepted_options.append((quality, mask, decision))
            else:
                rejected_count += 1

        if accepted_options:
            _, best_mask, best_decision = max(
                accepted_options, key=lambda item: item[0]
            )
            # Other valid local proposals were considered but not selected.
            for _, _, decision in accepted_options:
                if decision is not best_decision:
                    decision["accepted"] = False
                    decision["reason"] = "lower_ranked_local_proposal"
                    rejected_count += 1
            if mode == "conservative":
                output[candidate.component_mask > 0] = 0
            output |= best_mask
            accepted_count += 1

    for proposal in global_proposals:
        mask = _validated_mask(proposal.mask, yolo.shape)
        reason = None
        probability = None
        if mode == "conservative":
            features = {}
            reason = "global_disabled_in_conservative"
        elif mask is None:
            features = {}
            reason = "malformed_mask"
            probability = None
        else:
            features = proposal_features(mask, yolo, proposal.score)
            if not mask.any():
                reason = "empty_mask"
            elif features["score"] < float(config.sam3_global_score_thresh):
                reason = "low_score"
            elif features["area_ratio"] > float(
                config.sam3_global_max_area_ratio
            ):
                reason = "excessive_image_area"
            elif yolo.any() and features["area_growth"] > _max_growth(
                config, mode
            ):
                reason = "excessive_growth"
            elif mode == "balanced" and not yolo.any():
                reason = "balanced_requires_yolo_support"
            elif (
                mode == "balanced"
                and features["distance_ratio"]
                > float(config.sam3_balanced_max_distance_ratio)
            ):
                reason = "too_far_from_yolo"
            if reason is None and selector is not None:
                selected, probability = apply_selector(
                    features, proposal.source, 0.0
                )
                if not selected:
                    reason = "selector_rejected"
            else:
                probability = None
        decisions.append(
            {
                "source": proposal.source,
                "candidate_index": None,
                "prompt": proposal.prompt,
                "accepted": reason is None,
                "reason": reason,
                "selector_probability": probability,
                **features,
            }
        )
        if reason is None:
            output |= mask
            accepted_count += 1
        else:
            rejected_count += 1

    return output.astype(np.uint8), {
        "accepted": accepted_count,
        "rejected": rejected_count,
        "proposal_decisions": decisions,
    }


def fuse_sam3_masks(
    yolo_mask: np.ndarray,
    candidates: Sequence[Sam3Candidate],
    candidate_masks: Sequence[np.ndarray],
    *,
    conservative: bool,
    min_yolo_overlap: float,
    max_area_growth: float,
    max_image_area_ratio: float,
    global_masks: Optional[Sequence[np.ndarray]] = None,
) -> tuple[np.ndarray, dict]:
    """Compatibility wrapper for the original two-mode pure fusion API."""
    class Config:
        sam3_local_score_thresh = 0.0
        sam3_min_yolo_overlap = min_yolo_overlap
        sam3_min_yolo_coverage = min_yolo_overlap
        sam3_balanced_max_distance_ratio = 1.0
        sam3_global_score_thresh = 0.0
        sam3_global_max_area_ratio = max_image_area_ratio
        sam3_max_area_growth_conservative = max_area_growth
        sam3_max_area_growth_balanced = max_area_growth
        sam3_max_area_growth_open = max_area_growth

    proposals = [
        Sam3Proposal(mask, 1.0, "legacy_local", index)
        for index, mask in enumerate(candidate_masks)
    ]
    proposals.extend(
        Sam3Proposal(mask, 1.0, "legacy_global")
        for mask in (global_masks or ())
    )
    return fuse_sam3_proposals(
        yolo_mask,
        candidates,
        proposals,
        mode="conservative" if conservative else "open",
        config=Config(),
    )


class Sam3Refiner:
    """Lazy process-isolated SAM 3 adapter with strict YOLO fallback."""

    def __init__(self, config, backend_factory=None) -> None:
        self.config = config
        self._backend_factory = backend_factory or self._build_backend
        self._backend = None
        self._selector = None
        self._selector_error = None

    @staticmethod
    def _build_backend(config):
        return Sam3WorkerClient(config)

    def _ensure_loaded(self):
        if self._backend is None:
            self._backend = self._backend_factory(self.config)
            if isinstance(self._backend, tuple):
                # Preserve compatibility with existing fake factories.
                model, processor = self._backend
                self._backend = _InProcessBackend(model, processor, self.config)
        return self._backend

    def _ensure_selector_loaded(self):
        if not bool(getattr(self.config, "sam3_selector_enabled", False)):
            return None
        if self._selector_error is not None:
            raise RuntimeError(self._selector_error)
        if self._selector is None:
            try:
                from waterseg_platform.sam3_selector import ProposalSelector

                self._selector = ProposalSelector.load(
                    self.config.sam3_selector_path
                )
                threshold = getattr(
                    self.config, "sam3_selector_accept_threshold", None
                )
                if threshold is not None:
                    self._selector.accept_threshold = float(threshold)
            except Exception as exc:
                self._selector_error = f"{type(exc).__name__}: {exc}"
                raise
        return self._selector

    def _save_debug(
        self,
        image_bgr: np.ndarray,
        yolo: np.ndarray,
        refined: np.ndarray,
        candidates: Sequence[Sam3Candidate],
        proposals: Sequence[Sam3Proposal],
        info: dict,
    ) -> None:
        if not self.config.sam3_save_debug_proposals:
            return
        root = Path(self.config.sam3_debug_dir or "runs/sam3_debug")
        request_dir = root / time.strftime("%Y%m%d_%H%M%S")
        request_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(request_dir / "image.jpg"), image_bgr)
        cv2.imwrite(str(request_dir / "yolo.png"), yolo * 255)
        cv2.imwrite(str(request_dir / "refined.png"), refined * 255)
        for index, proposal in enumerate(proposals):
            mask = _validated_mask(proposal.mask, yolo.shape)
            if mask is not None:
                cv2.imwrite(
                    str(request_dir / f"proposal_{index:03d}.png"),
                    mask * 255,
                )
        metadata = dict(info)
        metadata["candidates"] = [
            {
                "box_xyxy": list(candidate.box_xyxy),
                "positive_points": [list(point) for point in candidate.positive_points],
                "negative_points": [list(point) for point in candidate.negative_points],
                "area": candidate.area,
            }
            for candidate in candidates
        ]
        (request_dir / "diagnostics.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        info["debug_dir"] = str(request_dir)

    def refine(
        self,
        image_bgr: np.ndarray,
        yolo_mask: np.ndarray,
        *,
        enabled: bool,
        mode: str,
    ) -> tuple[np.ndarray, dict]:
        yolo = (np.asarray(yolo_mask) > 0).astype(np.uint8)
        info = {
            "enabled": bool(enabled),
            "ran": False,
            "mode": mode,
            "loaded": self._backend is not None,
            "candidate_count": 0,
            "accepted": 0,
            "rejected": 0,
            "fallback": False,
            "fallback_stage": None,
            "pcs_error": None,
            "interactive_error": None,
            "adapter_loaded": False,
            "adapter_error": None,
            "selector_enabled": bool(
                getattr(self.config, "sam3_selector_enabled", False)
            ),
            "selector_loaded": False,
            "selector_error": None,
            "elapsed_ms": 0.0,
            "peak_gpu_memory_mb": None,
            "error": None,
        }
        if not enabled:
            return yolo, info
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported SAM 3 mode: {mode}")

        candidates = extract_candidates(
            yolo,
            self.config.sam3_min_component_area_ratio,
            self.config.sam3_max_candidates,
            self.config.sam3_box_margin_ratio,
            self.config.sam3_positive_points_per_component,
            self.config.sam3_negative_points_per_component,
            self.config.sam3_negative_ring_ratio,
        )
        info["candidate_count"] = len(candidates)
        info["candidate_prompts"] = [
            {
                "box_xyxy": list(candidate.box_xyxy),
                "positive_points": [
                    list(point) for point in candidate.positive_points
                ],
                "negative_points": [
                    list(point) for point in candidate.negative_points
                ],
                "area": candidate.area,
            }
            for candidate in candidates
        ]
        if mode == "conservative" and not candidates:
            return yolo, info

        started = time.perf_counter()
        try:
            selector = self._ensure_selector_loaded()
            info["selector_loaded"] = selector is not None
            backend = self._ensure_loaded()
            info["loaded"] = True
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            response = backend.predict(
                image_rgb=image_rgb,
                candidates=candidates,
                text_prompts=list(self.config.sam3_text_prompts),
                mode=mode,
                local_enabled=bool(self.config.sam3_local_refine_enabled),
                global_enabled=bool(self.config.sam3_global_text_enabled),
            )
            info["pcs_error"] = response.get("pcs_error")
            info["interactive_error"] = response.get("interactive_error")
            info["peak_gpu_memory_mb"] = response.get("peak_gpu_memory_mb")
            info["adapter_loaded"] = bool(response.get("adapter_loaded", False))
            info["adapter_error"] = response.get("adapter_error")
            proposals = [
                Sam3Proposal(
                    mask=np.asarray(item["mask"]),
                    score=float(item.get("score", 0.0)),
                    source=str(item["source"]),
                    candidate_index=item.get("candidate_index"),
                    prompt=item.get("prompt"),
                )
                for item in response.get("proposals", [])
            ]
            if not proposals and (
                info["pcs_error"] or info["interactive_error"]
            ):
                info["fallback"] = True
                info["fallback_stage"] = "yolo"
                info["error"] = info["pcs_error"] or info["interactive_error"]
                return yolo, info

            refined, fusion_info = fuse_sam3_proposals(
                yolo,
                candidates,
                proposals,
                mode=mode,
                config=self.config,
                selector=selector,
            )
            info.update(fusion_info)
            info["ran"] = True
            if info["interactive_error"] and not info["pcs_error"]:
                info["fallback_stage"] = "text_box"
            self._save_debug(
                image_bgr, yolo, refined, candidates, proposals, info
            )
            return refined, info
        except Exception as exc:
            info["fallback"] = True
            info["fallback_stage"] = "yolo"
            info["error"] = f"{type(exc).__name__}: {exc}"
            if self._selector_error:
                info["selector_error"] = self._selector_error
            return yolo, info
        finally:
            info["elapsed_ms"] = (time.perf_counter() - started) * 1000.0


class _InProcessBackend:
    """Small adapter used only by unit-test fake processors."""

    def __init__(self, model, processor, config) -> None:
        self.model = model
        self.processor = processor
        self.config = config

    def predict(
        self,
        image_rgb,
        candidates,
        text_prompts,
        mode,
        local_enabled,
        global_enabled,
    ):
        if self.processor is None and hasattr(self.model, "predict"):
            return self.model.predict(
                image_rgb=image_rgb,
                candidates=candidates,
                text_prompts=text_prompts,
                mode=mode,
                local_enabled=local_enabled,
                global_enabled=global_enabled,
            )
        from PIL import Image

        state = self.processor.set_image(Image.fromarray(image_rgb))
        proposals = []
        if local_enabled:
            for index, candidate in enumerate(candidates):
                for prompt in text_prompts:
                    self.processor.reset_all_prompts(state)
                    state = self.processor.set_text_prompt(prompt, state)
                    state = self.processor.add_geometric_prompt(
                        list(candidate.box_cxcywh), True, state
                    )
                    masks = np.asarray(state.get("masks", []))
                    scores = np.asarray(state.get("scores", [])).reshape(-1)
                    if len(masks):
                        best = int(np.argmax(scores)) if len(scores) else 0
                        proposals.append(
                            {
                                "mask": masks[best],
                                "score": float(scores[best]) if len(scores) else 1.0,
                                "source": "text_box",
                                "candidate_index": index,
                                "prompt": prompt,
                            }
                        )
        if mode != "conservative" and global_enabled:
            for prompt in text_prompts:
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(prompt, state)
                masks = np.asarray(state.get("masks", []))
                scores = np.asarray(state.get("scores", [])).reshape(-1)
                if len(masks):
                    best = int(np.argmax(scores)) if len(scores) else 0
                    proposals.append(
                        {
                            "mask": masks[best],
                            "score": float(scores[best]) if len(scores) else 1.0,
                            "source": "global_text",
                            "prompt": prompt,
                        }
                    )
        return {"proposals": proposals}


class Sam3WorkerClient:
    """Persistent process-isolated SAM 3 inference client."""

    def __init__(self, config, startup_timeout: float = 60.0) -> None:
        checkpoint = Path(config.sam3_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM 3 checkpoint not found: {checkpoint}")
        self._authkey = os.urandom(16)
        self._listener = Listener(("127.0.0.1", 0), authkey=self._authkey)
        socket_obj = getattr(
            getattr(self._listener, "_listener", None), "_socket", None
        )
        if socket_obj is not None:
            socket_obj.settimeout(float(startup_timeout))
        host, port = self._listener.address
        log_file = tempfile.NamedTemporaryFile(
            prefix="waterseg_sam3_worker_", suffix=".log", delete=False
        )
        self.log_path = log_file.name
        creationflags = (
            subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "waterseg_platform.sam3_worker",
                "--host",
                str(host),
                "--port",
                str(port),
                "--authkey",
                self._authkey.hex(),
                "--checkpoint",
                str(checkpoint),
                "--device",
                str(config.sam3_device),
                "--confidence",
                str(
                    min(
                        config.sam3_confidence,
                        config.sam3_local_score_thresh,
                        config.sam3_global_score_thresh,
                    )
                ),
                "--compile",
                "1" if config.sam3_compile else "0",
                "--adapter-enabled",
                "1" if config.sam3_adapter_enabled else "0",
                "--adapter-path",
                str(config.sam3_lora_path),
                "--adapter-strict-fingerprint",
                "1" if config.sam3_adapter_strict_fingerprint else "0",
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        log_file.close()
        try:
            self._connection = self._listener.accept()
        except Exception:
            self.close()
            raise RuntimeError(
                f"SAM 3 worker failed to connect; log: {self.log_path}"
            )
        finally:
            self._listener.close()
        atexit.register(self.close)

    def _worker_error(self, message: str) -> RuntimeError:
        detail = message
        try:
            log_text = Path(self.log_path).read_text(
                encoding="utf-8", errors="replace"
            )
            if log_text.strip():
                detail = f"{message}; worker log: {log_text[-2000:]}"
        except OSError:
            pass
        return RuntimeError(detail)

    def predict(
        self,
        *,
        image_rgb,
        candidates,
        text_prompts,
        mode,
        local_enabled,
        global_enabled,
        timeout: float = 600.0,
    ) -> dict:
        if self._process.poll() is not None:
            raise self._worker_error(
                f"SAM 3 worker exited with code {self._process.returncode}"
            )
        self._connection.send(
            {
                "type": "predict",
                "image_rgb": np.asarray(image_rgb, dtype=np.uint8),
                "candidates": [
                    {
                        "box_cxcywh": list(candidate.box_cxcywh),
                        "box_xyxy": list(candidate.box_xyxy),
                        "positive_points": [
                            list(point) for point in candidate.positive_points
                        ],
                        "negative_points": [
                            list(point) for point in candidate.negative_points
                        ],
                    }
                    for candidate in candidates
                ],
                "text_prompts": list(text_prompts),
                "mode": mode,
                "local_enabled": bool(local_enabled),
                "global_enabled": bool(global_enabled),
            }
        )
        if not self._connection.poll(float(timeout)):
            raise TimeoutError(
                f"SAM 3 worker timed out after {timeout:.0f}s"
            )
        response = self._connection.recv()
        if not response.get("ok"):
            raise self._worker_error(response.get("error", "unknown error"))
        return response

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            try:
                connection.send({"type": "close"})
            except Exception:
                pass
            try:
                connection.close()
            except Exception:
                pass
            self._connection = None
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()
            self._process = None

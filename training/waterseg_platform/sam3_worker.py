"""Process-isolated SAM 3 worker for text/box and interactive point prompts."""
from __future__ import annotations

import argparse
import gc
import traceback
from multiprocessing.connection import Client

# PyTorch must be loaded before other large native runtimes in this process.
import torch
import numpy as np
from PIL import Image


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _state_best(state: dict):
    masks = state.get("masks")
    if masks is None:
        return None
    array = _to_numpy(masks)
    while array.ndim > 3 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[0] == 0:
        return None
    scores = _to_numpy(state.get("scores", np.ones(array.shape[0]))).reshape(-1)
    index = int(np.argmax(scores[: array.shape[0]]))
    return (array[index] > 0).astype(np.uint8), float(scores[index])


def _interactive_best(masks, scores):
    mask_array = _to_numpy(masks)
    score_array = _to_numpy(scores).reshape(-1)
    if mask_array.ndim == 2:
        mask_array = mask_array[None, ...]
    if mask_array.ndim != 3 or len(mask_array) == 0:
        return None
    index = int(np.argmax(score_array[: len(mask_array)]))
    return (
        (mask_array[index] > 0).astype(np.uint8),
        float(score_array[index]),
    )


class WorkerModel:
    def __init__(
        self,
        checkpoint: str,
        device: str,
        confidence: float,
        compile_model: bool,
        adapter_enabled: bool = False,
        adapter_path: str = "",
        adapter_strict_fingerprint: bool = True,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.confidence = confidence
        self.compile_model = compile_model
        self.adapter_enabled = bool(adapter_enabled)
        self.adapter_path = adapter_path
        self.adapter_strict_fingerprint = bool(adapter_strict_fingerprint)
        self.adapter_loaded = False
        self.adapter_error = None
        self.model = None
        self.processor = None
        self.interactive_available = False
        self.interactive_load_error = None

    def _build(self, enable_interactive: bool):
        from sam3 import build_sam3_image_model

        return build_sam3_image_model(
            checkpoint_path=self.checkpoint,
            load_from_HF=False,
            device=self.device,
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=enable_interactive,
            compile=self.compile_model,
        )

    def _ensure_loaded(self) -> None:
        if self.processor is not None:
            return
        from sam3.model.sam3_image_processor import Sam3Processor

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SAM 3 requested CUDA, but CUDA is unavailable")
        try:
            self.model = self._build(enable_interactive=True)
            self.interactive_available = (
                self.model.inst_interactive_predictor is not None
            )
        except Exception as exc:
            self.interactive_load_error = (
                f"{type(exc).__name__}: {exc}"
            )
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # The PCS text/box branch must remain usable when the optional
            # instance-interactive branch cannot be constructed.
            self.model = self._build(enable_interactive=False)
            self.interactive_available = False
        if self.adapter_enabled:
            try:
                from training.sam3_lora.lora import load_lora_adapter

                load_lora_adapter(
                    self.model,
                    self.adapter_path,
                    base_checkpoint=self.checkpoint,
                    strict_fingerprint=self.adapter_strict_fingerprint,
                )
                self.adapter_loaded = True
            except Exception as exc:
                self.adapter_error = f"{type(exc).__name__}: {exc}"
                # A partially injected model must never be used as the base
                # fallback. Rebuild a clean base SAM 3 model.
                self.model = None
                self.processor = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    self.model = self._build(enable_interactive=True)
                    self.interactive_available = (
                        self.model.inst_interactive_predictor is not None
                    )
                except Exception:
                    self.model = self._build(enable_interactive=False)
                    self.interactive_available = False
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=self.confidence,
        )

    def _text_box_proposals(
        self,
        state: dict,
        candidates: list[dict],
        prompts: list[str],
    ) -> list[dict]:
        proposals = []
        for candidate_index, candidate in enumerate(candidates):
            for prompt in prompts:
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(prompt, state)
                state = self.processor.add_geometric_prompt(
                    candidate["box_cxcywh"], True, state
                )
                best = _state_best(state)
                if best is not None:
                    mask, score = best
                    proposals.append(
                        {
                            "mask": mask,
                            "score": score,
                            "source": "text_box",
                            "candidate_index": candidate_index,
                            "prompt": prompt,
                        }
                    )
        return proposals

    def _interactive_proposals(
        self,
        state: dict,
        candidates: list[dict],
    ) -> list[dict]:
        proposals = []
        for candidate_index, candidate in enumerate(candidates):
            positive = candidate.get("positive_points", [])
            negative = candidate.get("negative_points", [])
            points = np.asarray(positive + negative, dtype=np.float32)
            labels = np.asarray(
                [1] * len(positive) + [0] * len(negative),
                dtype=np.int32,
            )
            if len(points) == 0:
                points = None
                labels = None
            masks, scores, _ = self.model.predict_inst(
                state,
                point_coords=points,
                point_labels=labels,
                box=np.asarray(candidate["box_xyxy"], dtype=np.float32),
                multimask_output=True,
            )
            best = _interactive_best(masks, scores)
            if best is not None:
                mask, score = best
                proposals.append(
                    {
                        "mask": mask,
                        "score": score,
                        "source": "interactive_points",
                        "candidate_index": candidate_index,
                        "prompt": None,
                    }
                )
        return proposals

    def _global_proposals(
        self, state: dict, prompts: list[str]
    ) -> list[dict]:
        proposals = []
        for prompt in prompts:
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(prompt, state)
            best = _state_best(state)
            if best is not None:
                mask, score = best
                proposals.append(
                    {
                        "mask": mask,
                        "score": score,
                        "source": "global_text",
                        "candidate_index": None,
                        "prompt": prompt,
                    }
                )
        return proposals

    def predict(self, request: dict) -> dict:
        self._ensure_loaded()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        image_rgb = np.asarray(request["image_rgb"], dtype=np.uint8)
        candidates = list(request.get("candidates", []))
        prompts = list(request.get("text_prompts", [])) or ["standing water"]
        mode = str(request.get("mode", "conservative"))
        proposals = []
        pcs_error = None
        interactive_error = self.interactive_load_error

        use_cuda_amp = self.device == "cuda"
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_cuda_amp,
        ):
            state = self.processor.set_image(Image.fromarray(image_rgb))
            if request.get("local_enabled", True) and candidates:
                try:
                    proposals.extend(
                        self._text_box_proposals(state, candidates, prompts)
                    )
                except Exception as exc:
                    pcs_error = f"{type(exc).__name__}: {exc}"

                if self.interactive_available:
                    try:
                        proposals.extend(
                            self._interactive_proposals(state, candidates)
                        )
                    except Exception as exc:
                        interactive_error = (
                            f"{type(exc).__name__}: {exc}"
                        )

            if (
                mode != "conservative"
                and request.get("global_enabled", True)
            ):
                try:
                    proposals.extend(
                        self._global_proposals(state, prompts)
                    )
                except Exception as exc:
                    if pcs_error is None:
                        pcs_error = f"{type(exc).__name__}: {exc}"

        peak_mb = None
        if torch.cuda.is_available():
            peak_mb = float(torch.cuda.max_memory_allocated()) / (1024**2)
        return {
            "ok": True,
            "proposals": proposals,
            "pcs_error": pcs_error,
            "interactive_error": interactive_error,
            "interactive_available": self.interactive_available,
            "adapter_loaded": self.adapter_loaded,
            "adapter_error": self.adapter_error,
            "peak_gpu_memory_mb": peak_mb,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--authkey", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--compile", type=int, default=0)
    parser.add_argument("--adapter-enabled", type=int, default=0)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--adapter-strict-fingerprint", type=int, default=1)
    args = parser.parse_args()

    connection = Client(
        (args.host, args.port),
        authkey=bytes.fromhex(args.authkey),
    )
    worker = WorkerModel(
        args.checkpoint,
        args.device,
        args.confidence,
        bool(args.compile),
        bool(args.adapter_enabled),
        args.adapter_path,
        bool(args.adapter_strict_fingerprint),
    )
    while True:
        request = connection.recv()
        if request.get("type") == "close":
            break
        try:
            connection.send(worker.predict(request))
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            connection.send(
                {
                    "ok": False,
                    "error": (
                        f"{type(exc).__name__}: {exc}\n"
                        f"{traceback.format_exc(limit=5)}"
                    ),
                }
            )
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

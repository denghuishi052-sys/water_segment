"""ONNX Runtime wrapper for the YOLOv8-seg water model.

This module is the only place the platform talks to ONNX Runtime. The
:class:`OnnxSegmenter` class:

* loads the exported model with the requested provider list
* auto-falls-back to the next provider in the list if the preferred one
  is not available
* exposes :meth:`run` for raw inference, returning ``(output0, output1)``
  numpy arrays in the canonical YOLOv8-seg layout
* exposes :meth:`predict_mask` for convenience: pre-process + run + decode
  + post-process

.NET equivalent: ``OnnxSegmenter.cs`` (``Microsoft.ML.OnnxRuntime``).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from waterseg_platform.preprocessing import LetterboxMeta, letterbox, to_nchw_float

LOG = logging.getLogger(__name__)


class OnnxSegmenter:
    """Thin wrapper around an ONNX Runtime ``InferenceSession`` for YOLOv8-seg.

    Example:
        >>> seg = OnnxSegmenter("onnx/floodnet_binary_aug_yolov8m_1024.onnx",
        ...                     providers=["CUDAExecutionProvider",
        ...                                "CPUExecutionProvider"])
        >>> mask = seg.predict_mask(image_bgr)  # HxW uint8 binary mask
    """

    def __init__(
        self,
        model_path: str,
        providers: Optional[Sequence[str]] = None,
        imgsz: int = 1024,
    ) -> None:
        if providers is None:
            providers = ("CUDAExecutionProvider", "CPUExecutionProvider")

        # On Windows, ``onnxruntime`` advertises ``CUDAExecutionProvider``
        # as available whenever the wheel was built with CUDA support,
        # but the actual cuBLAS / cuDNN / cuFFT runtime DLLs are loaded
        # lazily from PATH. The official NVIDIA CUDA 12 runtime is
        # distributed as pip packages (nvidia-cublas-cu12, nvidia-cudnn-cu12,
        # ...); we auto-discover them under site-packages/nvidia/*/bin
        # and add them to PATH + ``os.add_dll_directory`` so CUDA just
        # works after ``pip install``. This is a no-op on Linux/macOS
        # and harmless if no nvidia-* package is installed.
        self._bootstrap_cuda_dlls()

        # Imported lazily so this module is testable without onnxruntime.
        import onnxruntime as ort

        self._imgsz = int(imgsz)
        self._model_path = str(Path(model_path).resolve())

        available = set(ort.get_available_providers())
        chosen: List[str] = []
        for p in providers:
            if p in available:
                chosen.append(p)
            else:
                LOG.warning("Provider %s requested but not available; skipping.", p)
        if not chosen:
            raise RuntimeError(
                f"None of the requested providers {list(providers)} are available. "
                f"Available: {sorted(available)}"
            )
        # Always keep CPU as the very last resort.
        if "CPUExecutionProvider" not in chosen:
            chosen.append("CPUExecutionProvider")

        LOG.info("Loading ONNX model %s with providers=%s", self._model_path, chosen)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            self._model_path, sess_options=sess_options, providers=chosen
        )
        self._providers_active = list(self._session.get_providers())
        self._input_name = self._session.get_inputs()[0].name
        self._input_shape = tuple(self._session.get_inputs()[0].shape)
        outputs = self._session.get_outputs()
        out0, out1 = outputs[0], outputs[1]
        self._output0_name = out0.name
        self._output1_name = out1.name
        self._output0_shape = tuple(out0.shape)
        self._output1_shape = tuple(out1.shape)
        LOG.info(
            "ONNX ready. in=%s shape=%s | out[0]=%s shape=%s | out[1]=%s shape=%s",
            self._input_name, self._input_shape,
            self._output0_name, self._output0_shape,
            self._output1_name, self._output1_shape,
        )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bootstrap_cuda_dlls() -> None:
        """Auto-discover the NVIDIA CUDA 12 runtime pip packages on Windows.

        ``onnxruntime``'s CUDA provider DLL (``onnxruntime_providers_cuda.dll``)
        dynamically links against ``cublasLt64_12.dll``, ``cudnn64_9.dll``,
        ``cufft64_11.dll``, and friends. Those DLLs ship in the
        ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12`` / ... pip packages
        (e.g. installed automatically by PyTorch wheels). On Windows the
        OS resolves DLL dependencies from PATH, so we have to add the
        package ``bin/`` directories to PATH before the first
        ``ort.InferenceSession(...)`` call — otherwise ORT falls back
        silently to the CPU provider.

        This method walks ``site-packages/nvidia/*/bin`` and prepends each
        existing directory to ``os.environ['PATH']`` plus
        ``os.add_dll_directory`` (the Win 3.8+ replacement for
        ``os.environ['PATH']``). It is a no-op on non-Windows or when
        none of the nvidia-* packages are present.
        """
        if sys.platform != "win32":
            return
        try:
            import site
            site_dirs = list(site.getsitepackages())
        except Exception:
            site_dirs = []
        # `site.getsitepackages()` may miss the user site on some
        # layouts; add it explicitly.
        try:
            import site as _site
            us = _site.getusersitepackages()
            if us and us not in site_dirs:
                site_dirs.append(us)
        except Exception:
            pass
        nvidia_pkgs = (
            "cublas", "cudnn", "cuda_runtime", "cuda_nvrtc",
            "cufft", "curand", "cusolver", "cusparse",
            "nvjitlink", "nvtx",
        )
        added: List[str] = []
        for sd in site_dirs:
            nvidia_root = os.path.join(sd, "nvidia")
            if not os.path.isdir(nvidia_root):
                continue
            for pkg in nvidia_pkgs:
                bin_dir = os.path.join(nvidia_root, pkg, "bin")
                if not os.path.isdir(bin_dir):
                    continue
                if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                # os.add_dll_directory is the Win 3.8+ way; skip on older.
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(bin_dir)
                    except Exception:
                        pass
                added.append(bin_dir)
        if added:
            LOG.info(
                "Auto-added %d CUDA 12 runtime dir(s) from pip "
                "nvidia-*-cu12 packages: %s",
                len(added), added,
            )

    @property
    def providers_active(self) -> List[str]:
        """The provider list actually in use (after auto-fallback)."""
        return list(self._providers_active)

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self._input_shape

    @property
    def output_names(self) -> List[str]:
        return [self._output0_name, self._output1_name]

    @property
    def output0_shape(self) -> Tuple[int, ...]:
        """Expected ``output0`` shape: ``(1, 4+nc+nm, A)`` for YOLOv8-seg."""
        return self._output0_shape

    @property
    def output1_shape(self) -> Tuple[int, ...]:
        """Expected ``output1`` shape: ``(1, nm, mh, mw)`` (mask prototypes)."""
        return self._output1_shape

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def preprocess(
        self, image_bgr: np.ndarray, rect: bool = False,
    ) -> Tuple[np.ndarray, LetterboxMeta]:
        """Letterbox + NCHW float conversion. Returns the tensor and meta."""
        canvas, meta = letterbox(image_bgr, imgsz=self._imgsz, rect=rect)
        tensor = to_nchw_float(canvas, assume_rgb=False)
        return tensor, meta

    def run(self, tensor_nchw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run the model on a pre-built NCHW tensor.

        Returns ``(output0, output1)`` as numpy arrays with the canonical
        YOLOv8-seg layout:

        * ``output0``: ``(1, 4+nc+nm, A)`` -- raw logits
        * ``output1``: ``(1, nm, mh, mw)`` -- mask prototypes
        """
        if tensor_nchw.dtype != np.float32:
            tensor_nchw = tensor_nchw.astype(np.float32)
        if not tensor_nchw.flags["C_CONTIGUOUS"]:
            tensor_nchw = np.ascontiguousarray(tensor_nchw)
        outputs = self._session.run(
            [self._output0_name, self._output1_name],
            {self._input_name: tensor_nchw},
        )
        return outputs[0], outputs[1]

    def predict_mask(
        self,
        image_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.5,
        mask_thres: float = 0.5,
        mask_box_expand_ratio: float = 0.0,
        min_area_ratio: float = 0.0005,
        morph_close: bool = True,
        max_det: int = 300,
        nc: int = 1,
        nm: int = 32,
    ) -> Tuple[np.ndarray, dict]:
        """End-to-end: image -> binary mask + diagnostic info.

        Imports the decoder lazily to keep this module dependency-light.
        """
        # Lazy import to break the cycle between engine and postprocessing.
        from waterseg_platform.postprocessing import decode_segmentation

        h, w = image_bgr.shape[:2]
        tensor, meta = self.preprocess(image_bgr)
        output0, output1 = self.run(tensor)
        mask = decode_segmentation(
            output0=output0,
            output1=output1,
            letterbox_meta=meta,
            original_shape=(h, w),
            conf=conf,
            iou=iou,
            mask_thres=mask_thres,
            mask_box_expand_ratio=mask_box_expand_ratio,
            min_area_ratio=min_area_ratio,
            morph_close=morph_close,
            max_det=max_det,
            nc=nc,
            nm=nm,
            class_scores_sigmoided=True,  # ONNX export already applies sigmoid
        )
        info = {
            "providers": self.providers_active,
            "input_shape": self.input_shape,
            "output0_shape": self.output0_shape,
            "output1_shape": self.output1_shape,
            "letterbox": {
                "r": meta.r, "pad_w": meta.pad_w, "pad_h": meta.pad_h,
                "new_w": meta.new_w, "new_h": meta.new_h,
            },
            "pred_area": int(mask.sum()),
            "pred_area_ratio": float(mask.sum()) / float(h * w),
        }
        return mask, info

"""Image preprocessing for the YOLOv8-seg ONNX platform.

The two functions in this module are the only image transforms the platform
applies before the ONNX forward pass:

* :func:`letterbox` -- resize-keep-aspect to a square ``imgsz x imgsz`` canvas,
  pad with constant 114 (the YOLO default). Returns the canvas and the
  ``(r, pad_w, pad_h)`` triple needed to map coordinates back to the
  original image.

* :func:`to_nchw_float` -- convert a BGR uint8 image (or a pre-letterboxed
  canvas) into the ``(1, 3, H, W) float32`` NCHW tensor the model expects:
  BGR->RGB, divide by 255, transpose, ensure contiguous memory.

.NET equivalent: ``Preprocessor.cs`` (OpenCvSharp4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


# Default YOLO letterbox pad colour (RGB 114, 114, 114). OpenCV uses BGR.
_PAD_COLOR_BGR: Tuple[int, int, int] = (114, 114, 114)


@dataclass(frozen=True)
class LetterboxMeta:
    """Geometry of a letterbox operation, sufficient to invert it.

    Attributes:
        r:        ``imgsz / max(H, W)`` of the source image.
        pad_w:    Pixels of left padding applied to the resized image.
        pad_h:    Pixels of top padding applied to the resized image.
        new_w:    Width of the resized (pre-pad) image.
        new_h:    Height of the resized (pre-pad) image.
        imgsz:    The square canvas size the image was letterboxed into.
    """

    r: float
    pad_w: int
    pad_h: int
    new_w: int
    new_h: int
    imgsz: int

    def to_original(self, x: float, y: float) -> Tuple[float, float]:
        """Map a (x, y) point in letterbox coords back to original-image coords."""
        return (x - self.pad_w) / self.r, (y - self.pad_h) / self.r

    def to_letterbox(self, x: float, y: float) -> Tuple[float, float]:
        """Map a (x, y) point in original coords into letterbox coords."""
        return x * self.r + self.pad_w, y * self.r + self.pad_h


def letterbox(
    image_bgr: np.ndarray,
    imgsz: int = 704,
    color: Tuple[int, int, int] = _PAD_COLOR_BGR,
    rect: bool = False,
    stride: int = 32,
) -> Tuple[np.ndarray, "LetterboxMeta"]:
    """Letterbox ``image_bgr`` into a square ``imgsz x imgsz`` canvas.

    Args:
        image_bgr: HxWx3 uint8 BGR image (OpenCV native).
        imgsz:     Target size. For square mode, the canvas side length.
                   For rect mode, the maximum side length.
        color:     Pad color in BGR (only used when rect=False).
        rect:      If True, use rect-mode letterbox: resize to fit within
                   imgsz without padding, rounding dimensions up to the
                   nearest multiple of ``stride``. No gray border is added.
                   This matches Ultralytics's ``rect=True`` predict mode
                   and produces sharper results because the model sees no
                   padding artifacts.
        stride:    Grid stride for rect-mode dimension rounding. Default 32.

    Returns:
        A 2-tuple ``(canvas, meta)`` where ``canvas`` is the resized image
        and ``meta`` carries the geometry needed to invert the transform.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            f"letterbox expects an HxWx3 BGR image, got shape {image_bgr.shape}"
        )
    h, w = image_bgr.shape[:2]
    r = imgsz / float(max(h, w))

    if rect:
        # Rect-mode: resize to fit within imgsz, no padding.
        # Round dimensions up to the nearest multiple of stride.
        new_w = max(stride, int(np.ceil(w * r / stride) * stride))
        new_h = max(stride, int(np.ceil(h * r / stride) * stride))
        interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR
        canvas = cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)
        meta = LetterboxMeta(
            r=float(r),
            pad_w=0,
            pad_h=0,
            new_w=int(new_w),
            new_h=int(new_h),
            imgsz=max(new_h, new_w),
        )
        return canvas, meta

    # Square-mode: original letterbox with center padding.
    new_w = int(round(w * r))
    new_h = int(round(h * r))

    # Resize-keep-aspect: AREA when downscaling, LINEAR when upscaling.
    if (new_w, new_h) != (w, h):
        interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)
    else:
        resized = image_bgr

    # Center-pad the resized image to (imgsz, imgsz).
    pad_w = (imgsz - new_w) // 2
    pad_h = (imgsz - new_h) // 2
    canvas = np.full((imgsz, imgsz, 3), color, dtype=np.uint8)
    canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    meta = LetterboxMeta(
        r=float(r),
        pad_w=int(pad_w),
        pad_h=int(pad_h),
        new_w=int(new_w),
        new_h=int(new_h),
        imgsz=int(imgsz),
    )
    return canvas, meta


def to_nchw_float(image_bgr_or_rgb: np.ndarray, assume_rgb: bool = False) -> np.ndarray:
    """Convert an HxWx3 uint8 image into the (1, 3, H, W) float32 NCHW tensor.

    The YOLOv8 ONNX export expects RGB, normalized to [0, 1] by /255, in
    NCHW float32 contiguous layout. This function does exactly that.

    Args:
        image_bgr_or_rgb: HxWx3 uint8 image.
        assume_rgb:       If True, skip the BGR->RGB conversion (use this when
                          the input has already been converted upstream).

    Returns:
        ``np.ndarray`` of shape ``(1, 3, H, W)`` and dtype ``float32``,
        contiguous in memory.
    """
    if image_bgr_or_rgb.dtype != np.uint8:
        arr = image_bgr_or_rgb.astype(np.float32)
    else:
        arr = image_bgr_or_rgb.astype(np.float32)

    if not assume_rgb:
        # OpenCV / letterbox produces BGR. The ONNX model expects RGB.
        arr = arr[..., ::-1]

    arr /= 255.0
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))
    # Add batch dim and ensure contiguous memory.
    arr = np.ascontiguousarray(arr[None, ...])
    return arr

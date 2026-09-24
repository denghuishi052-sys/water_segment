#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def add_torch_cuda_dlls() -> None:
    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
    except Exception:
        return
    if not torch_lib.exists():
        return
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(torch_lib))


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test a YOLOv8-seg ONNX model with ONNX Runtime GPU.")
    parser.add_argument(
        "--model",
        default="runs/segment/runs/segment/waterlogging_yolov8m_640_b8-3/weights/best.onnx",
    )
    parser.add_argument("--image", default=r"C:\Users\17473\Downloads\Flood_1028.jpg")
    parser.add_argument("--output_dir", default="runs/single_image_tests/onnx_gpu_smoke_test")
    parser.add_argument("--imgsz", type=int, default=704)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    add_torch_cuda_dlls()

    import onnxruntime as ort
    from ultralytics import YOLO

    from src.dataset_utils import ensure_dir
    from src.mask_utils import postprocess_mask
    from src.visualization import save_panel

    args = parse_args()
    model_path = Path(args.model).resolve()
    image_path = Path(args.image).resolve()
    out_dir = ensure_dir(Path(args.output_dir).resolve())

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"CUDAExecutionProvider is unavailable. Available providers: {providers}")

    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]

    model = YOLO(model_path)
    result = model.predict(
        image_rgb,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=0,
        verbose=False,
    )[0]

    pred = np.zeros((h, w), dtype=np.uint8)
    if result.masks is not None and result.masks.data is not None:
        for mask in result.masks.data.detach().cpu().numpy():
            resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            pred |= (resized >= 0.5).astype(np.uint8)
    pred = postprocess_mask(pred)

    save_panel(out_dir / "prediction_panel.jpg", image_bgr, gt=None, pred=pred)
    cv2.imwrite(str(out_dir / "pred_mask.png"), (pred * 255).astype(np.uint8))

    instances = 0 if result.masks is None or result.masks.data is None else int(result.masks.data.shape[0])
    confidences = (
        []
        if result.boxes is None or result.boxes.conf is None
        else [float(x) for x in result.boxes.conf.detach().cpu().numpy()]
    )
    print(f"onnxruntime providers: {providers}")
    print(f"model: {model_path}")
    print(f"image: {image_path}")
    print(f"instances: {instances}")
    print("confidences:", ", ".join(f"{x:.4f}" for x in confidences) if confidences else "none")
    print(f"pred_area_ratio: {int(pred.sum()) / (h * w):.6f}")
    print(f"output_dir: {out_dir}")


if __name__ == "__main__":
    main()

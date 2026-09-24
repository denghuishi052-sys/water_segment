from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dual_context_water.inference import DualContextOnnxPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="onnx/dual_context_water_convnext_tiny_1024.onnx")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_dir", default="runs/dual_context_prediction")
    parser.add_argument("--low_threshold", type=float, default=0.40)
    parser.add_argument("--high_threshold", type=float, default=0.65)
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    predictor = DualContextOnnxPredictor(ROOT / args.model)
    mask, probability, info = predictor.predict_mask(
        image,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{image_path.stem}_pred.png"
    probability_path = output_dir / f"{image_path.stem}_probability.png"
    overlay_path = output_dir / f"{image_path.stem}_overlay.jpg"
    cv2.imwrite(str(mask_path), mask * 255)
    cv2.imwrite(str(probability_path), np.clip(probability * 255.0, 0, 255).astype(np.uint8))
    overlay = image.copy()
    red = np.zeros_like(image)
    red[:, :, 2] = 255
    selected = mask > 0
    overlay[selected] = cv2.addWeighted(image, 0.45, red, 0.55, 0)[selected]
    cv2.imwrite(str(overlay_path), overlay)
    print(json.dumps({
        **info,
        "mask_path": str(mask_path),
        "probability_path": str(probability_path),
        "overlay_path": str(overlay_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

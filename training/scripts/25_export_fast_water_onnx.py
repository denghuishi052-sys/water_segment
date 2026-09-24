"""Export the fast water model and validate the unchanged YOLOv8 .NET contract."""
from __future__ import annotations

import shutil
from pathlib import Path

import onnxruntime as ort
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs" / "segment" / "runs" / "segment" / "yolov8l_water_fast_10h" / "weights" / "best.pt"
TARGET = ROOT / "onnx" / "water_yolov8l_fast_10h_1024.onnx"
EXPECTED_OUTPUTS = [(1, 37, 21504), (1, 32, 256, 256)]


def main() -> None:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    exported = Path(YOLO(str(WEIGHTS)).export(format="onnx", imgsz=1024, opset=17, simplify=True))
    if exported.resolve() != TARGET.resolve():
        shutil.copy2(exported, TARGET)
    session = ort.InferenceSession(str(TARGET), providers=["CPUExecutionProvider"])
    output_shapes = [tuple(output.shape) for output in session.get_outputs()]
    if output_shapes != EXPECTED_OUTPUTS:
        raise RuntimeError(f"Unexpected ONNX outputs: {output_shapes}")
    print(f"Validated ONNX: {TARGET}")
    print(f"Output shapes: {output_shapes}")


if __name__ == "__main__":
    main()

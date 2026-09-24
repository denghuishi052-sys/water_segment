param(
    [string]$Python = "C:\Users\17473\miniforge3\envs\torch_env\python.exe",
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$Phase1 = Join-Path $Root "runs\segment\runs\segment\yolov8x_water_multidomain_phase1_640\weights\best.pt"
$Phase2Dir = Join-Path $Root "runs\segment\runs\segment\yolov8x_water_multidomain_phase2_1024"
$Phase2 = Join-Path $Phase2Dir "weights\best.pt"
$OnnxDir = Join-Path $Root "onnx"
$OnnxPath = Join-Path $OnnxDir "water_yolov8x_multidomain_1024.onnx"

while (-not (Test-Path $Phase1)) {
    Start-Sleep -Seconds $PollSeconds
}

& $Python scripts/04_train_yolov8.py --config configs/train_yolov8x_water_multidomain_phase2_1024.yaml
if ($LASTEXITCODE -ne 0) { throw "Phase 2 training failed with exit code $LASTEXITCODE." }
if (-not (Test-Path $Phase2)) { throw "Phase 2 completed without best.pt: $Phase2" }

New-Item -ItemType Directory -Force $OnnxDir | Out-Null
& $Python -c @"
from pathlib import Path
import onnxruntime as ort
from ultralytics import YOLO

weights = Path(r'$Phase2')
target = Path(r'$OnnxPath')
exported = Path(YOLO(str(weights)).export(format='onnx', imgsz=1024, opset=17, simplify=True))
if exported.resolve() != target.resolve():
    target.write_bytes(exported.read_bytes())
session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
shapes = [tuple(output.shape) for output in session.get_outputs()]
expected = [(1, 37, 21504), (1, 32, 256, 256)]
if shapes != expected:
    raise RuntimeError(f'Unexpected ONNX output shapes: {shapes}, expected {expected}')
print(f'ONNX validated: {target}')
print(f'Output shapes: {shapes}')
"@
if ($LASTEXITCODE -ne 0) { throw "ONNX export or validation failed with exit code $LASTEXITCODE." }

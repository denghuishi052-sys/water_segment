# YOLO-only .NET Export

This package contains only the YOLO ONNX prediction path for .NET development.
It does not include Python source code, Python dependency files, SAM3, or LoRA assets.

## Contents

- `dotnet/`: .NET solution, class library, CLI sample, and WinForms demo.
- `configs/yolo_only.json`: YOLO-only runtime configuration.
- `onnx/floodnet_binary_aug_yolov8m_1024.onnx`: YOLO segmentation model.
- `samples/valid_water_0003.jpg`: smoke-test image.

## Extension Points

The public entry point is `PredictionService`. It accepts image files and writes
mask/overlay outputs. Model-specific logic is behind `IImageSegmentationPredictor`.

Detailed .NET integration documentation is available in `docs/DOTNET_INTERFACE.md`.
Config and pipeline extension guidance is available in `docs/DOTNET_CONFIG_EXTENSION.md`.
End-to-end image input/output flow is documented in `docs/IMAGE_INPUT_OUTPUT_FLOW.md`.

To add another image segmentation model later:

1. Add a predictor class that implements `IImageSegmentationPredictor`.
2. Return a `SegmentationPrediction` containing the binary mask and optional probability map.
3. Register the new `modelType` in `ImageSegmentationPredictorFactory`.
4. Set `modelType` and model-specific paths/options in the JSON config.

CLI and WinForms callers do not need to change as long as the input remains an image.

## Run CLI

```powershell
cd D:\project\water_segment\platform_prediction_yolo_dotnet_export\dotnet
dotnet run --project samples\WaterSegmentation.PlatformPrediction.Cli\WaterSegmentation.PlatformPrediction.Cli.csproj -c Release -- `
  "D:\project\water_segment\platform_prediction_yolo_dotnet_export" `
  "D:\project\water_segment\platform_prediction_yolo_dotnet_export\samples\valid_water_0003.jpg" `
  "D:\project\water_segment\platform_prediction_yolo_dotnet_export\dotnet_runs\smoke" `
  "D:\project\water_segment\platform_prediction_yolo_dotnet_export\configs\yolo_only.json"
```

## Run GUI Demo

```powershell
cd D:\project\water_segment\platform_prediction_yolo_dotnet_export\dotnet
dotnet run --project samples\WaterSegmentation.PlatformPrediction.WinForms\WaterSegmentation.PlatformPrediction.WinForms.csproj -c Release
```

## Latest Fix

- Fixed YOLO mask placement so prototype masks are sampled back through the letterbox coordinates instead of being stretched inside the detection box.
- Set `maskBoxExpandRatio` to `0.5` to avoid the right-side water region being cut off by an overly tight detection box.
- Set `maskThreshold` to `0.65` to reduce part of the land over-segmentation while preserving the river region in the provided test image.
- Added `IImageSegmentationPredictor`, `PredictionRequest`, `SegmentationPrediction`, and `ImageSegmentationPredictorFactory` so future models can be plugged in without changing the image input/output layer.

## Verified On 2026-07-03

- `dotnet build WaterSegmentation.PlatformPrediction.sln -c Release`: 0 warnings, 0 errors.
- Provided aerial image: `maskArea=56319`, `predAreaRatio=0.669388`, `modelType=yolo-onnx`.
- Smoke sample: `maskArea=220471`, `predAreaRatio=0.165271`, `modelType=yolo-onnx`.
- Source scan confirms no Python files are present in the export package.

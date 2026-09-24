# .NET Config Extension Guide

本文档补充说明配置扩展策略。核心原则是：后续模型、预处理、后处理或输出链路变化时，不应把参数写死在业务代码里，而应通过 JSON 配置和 predictor 实现解耦。

## 1. 设计原则

调用方业务代码只依赖稳定入口：

```csharp
PredictionService
```

模型和处理链由配置决定：

```json
{
  "modelType": "yolo-onnx",
  "pipeline": {},
  "modelOptions": {},
  "preprocessOptions": {},
  "postprocessOptions": {},
  "outputOptions": {}
}
```

业务层不应写死：

- `maskThreshold`
- `tileSize`
- `overlapPixels`
- 模型路径
- 模型类型
- 后处理开关
- 输出文件命名规则
- 未来新增的处理链步骤

## 2. PredictionConfig 的扩展字段

`PredictionConfig` 当前支持两类配置：

1. 当前 YOLO 实现已经使用的强类型兼容字段。
2. 面向未来处理链的动态扩展字段。

动态扩展字段包括：

```csharp
public PipelineConfig Pipeline { get; set; } = new();
public Dictionary<string, JsonElement> ModelOptions { get; set; } = [];
public Dictionary<string, JsonElement> PreprocessOptions { get; set; } = [];
public Dictionary<string, JsonElement> PostprocessOptions { get; set; } = [];
public Dictionary<string, JsonElement> OutputOptions { get; set; } = [];

[JsonExtensionData]
public Dictionary<string, JsonElement>? ExtensionData { get; set; }
```

含义：

| 字段 | 用途 |
|---|---|
| `pipeline` | 描述当前处理链由哪些 step 组成。 |
| `modelOptions` | 模型运行相关参数，例如 runtime、输入输出格式、provider、batch 等。 |
| `preprocessOptions` | 预处理参数，例如 resize、normalize、pad、crop。 |
| `postprocessOptions` | 后处理参数，例如阈值、连通域、形态学、拼接策略。 |
| `outputOptions` | 输出控制，例如文件名、编码方式、是否返回概率图。 |
| `ExtensionData` | 捕获未来 JSON 中尚未定义的顶层字段，避免反序列化丢失扩展信息。 |

## 3. Pipeline 配置结构

当前默认 JSON 中已加入 pipeline：

```json
{
  "pipeline": {
    "type": "segmentation",
    "steps": [
      {
        "name": "preprocess",
        "type": "letterbox",
        "enabled": true,
        "parameters": {
          "imageSize": 1024,
          "padColor": [114, 114, 114]
        }
      },
      {
        "name": "inference",
        "type": "yolo-onnx",
        "enabled": true,
        "parameters": {
          "modelPath": "onnx/floodnet_binary_aug_yolov8m_1024.onnx",
          "confidenceThreshold": 0.25,
          "iouThreshold": 0.5
        }
      },
      {
        "name": "postprocess",
        "type": "binary-mask",
        "enabled": true,
        "parameters": {
          "maskThreshold": 0.65,
          "minAreaRatio": 0.0005,
          "morphClose": true
        }
      }
    ]
  }
}
```

当前 YOLO predictor 为兼容旧代码，仍使用强类型字段；后续新的 predictor 可以直接读取 `pipeline.steps`。

## 4. 读取动态参数

`PredictionConfig` 提供了通用读取方法：

```csharp
var runtime = config.GetModelOption("runtime", "onnxruntime");
var resizeMode = config.GetPreprocessOption("resizeMode", "letterbox");
var stitchMode = config.GetPostprocessOption("stitchMode", "max");
var maskSuffix = config.GetOutputOption("maskSuffix", "_pred.png");
```

也可以读取 pipeline step 的参数：

```csharp
var step = config.Pipeline.Steps
    .FirstOrDefault(x => x.Name == "postprocess" && x.Enabled);

if (step is not null)
{
    var threshold = PredictionConfig.GetOption(
        step.Parameters,
        "maskThreshold",
        0.65f);
}
```

## 5. 新模型接入建议

新增模型时，建议不要复用 YOLO 的强类型字段作为唯一参数来源。

推荐模式：

```csharp
public sealed class NewModelPredictor : IImageSegmentationPredictor
{
    private readonly PredictionConfig _config;

    public string ModelType => "new-model";

    public NewModelPredictor(string packageRoot, PredictionConfig config)
    {
        _config = config;

        var modelPath = config.GetModelOption("modelPath", config.ModelPath);
        var runtime = config.GetModelOption("runtime", "onnxruntime");
        var inputSize = config.GetPreprocessOption("imageSize", config.ImageSize);
    }

    public SegmentationPrediction Predict(PredictionRequest request)
    {
        var threshold = _config.GetPostprocessOption("maskThreshold", 0.5f);
        return new SegmentationPrediction(mask, probability, ModelType);
    }

    public void Dispose()
    {
    }
}
```

然后在工厂中注册：

```csharp
"new-model" => new NewModelPredictor(packageRoot, config)
```

## 6. 多处理链示例

如果未来处理链变成：

```text
resize -> coarse model -> ROI crop -> fine model -> morphology -> output
```

可以写成：

```json
{
  "modelType": "cascade-segmentation",
  "pipeline": {
    "type": "segmentation-cascade",
    "steps": [
      {
        "name": "coarse_preprocess",
        "type": "resize",
        "enabled": true,
        "parameters": {
          "maxSide": 1536
        }
      },
      {
        "name": "coarse_inference",
        "type": "onnx-segmentation",
        "enabled": true,
        "parameters": {
          "modelPath": "onnx/coarse.onnx",
          "threshold": 0.35
        }
      },
      {
        "name": "roi_builder",
        "type": "connected-components-to-roi",
        "enabled": true,
        "parameters": {
          "minAreaRatio": 0.001,
          "padding": 32
        }
      },
      {
        "name": "fine_inference",
        "type": "onnx-segmentation",
        "enabled": true,
        "parameters": {
          "modelPath": "onnx/fine.onnx",
          "threshold": 0.55
        }
      }
    ]
  }
}
```

这样业务入口仍然是：

```csharp
using var service = new PredictionService(packageRoot, config);
var result = service.PredictFile(imagePath, outputDir);
```

## 7. 当前兼容性

当前 YOLO-only predictor 仍兼容以下字段：

- `imageSize`
- `confidenceThreshold`
- `iouThreshold`
- `maskThreshold`
- `maskBoxExpandRatio`
- `minAreaRatio`
- `morphClose`
- `tiling`

这些字段保留是为了不破坏已验证版本。后续新 predictor 可以逐步改为只依赖 `pipeline` 和 options 字段。

## 8. 对 .NET 开发的要求

开发侧应遵守：

1. 不在业务代码中写死模型阈值和处理参数。
2. 不直接依赖具体 YOLO 类，优先依赖 `PredictionService` 或 `IImageSegmentationPredictor`。
3. 新模型参数放入 JSON 的 `pipeline.steps[*].parameters` 或 options 字段。
4. 新处理链通过新增 predictor 或 pipeline runner 扩展，不修改图片输入输出入口。
5. 配置文件应纳入版本管理，模型文件和配置文件保持一一对应。

## 9. 已实现的流程优化

当前包已实现以下工程化能力：

| 能力 | 实现位置 | 说明 |
|---|---|---|
| Pipeline Runner 骨架 | `PipelineRunner.cs` | 读取 `pipeline.steps`，生成执行链 trace，写入结果 metadata。 |
| 配置校验 | `PredictionConfigValidator.cs` | 在创建 predictor 前校验模型路径、阈值范围、tile 参数、注册模型类型、重复 step 名称等。 |
| 大图模式 | `YoloOnnxSegmentationPredictor.cs` | 支持 `full_quality`、`fast_preview`、`roi_refine`。 |
| 诊断信息 | `PredictionService.cs` | 输出图片尺寸、耗时、tile 数、pipeline trace、large image mode。 |
| 模型注册表 | `ModelRegistry` | 替代硬编码 switch，支持运行时注册新模型类型。 |
| 标准化输出 | `PredictionOutput.cs` | 每次预测输出 `*_result.json` 和 `summary.csv`。 |

### 9.1 大图模式配置

```json
{
  "largeImage": {
    "mode": "full_quality",
    "fastPreviewMaxSide": 1536,
    "roiCoarseThreshold": 0.35,
    "roiPaddingPixels": 64
  }
}
```

模式说明：

| mode | 说明 |
|---|---|
| `full_quality` | 默认模式。按 `tileSize/overlapPixels` 全分辨率切片推理，速度慢但保持当前已验证质量。 |
| `fast_preview` | 将大图缩放到 `fastPreviewMaxSide` 后推理，再把概率图映射回原图。速度快，适合预览或预筛，不保证精细结果。 |
| `roi_refine` | 先用快速预览找 ROI，再对 ROI 做高质量推理。适合目标区域相对集中的大图。 |

### 9.2 标准化输出

每张图片输出：

```text
<stem>_pred.png
<stem>_overlay.jpg
<stem>_result.json
summary.csv
```

`result.json` 包含：

- mask path
- overlay path
- mask area
- prediction area ratio
- model type
- image width/height
- elapsed milliseconds
- large image mode
- tile count
- pipeline enabled steps

### 9.3 新模型注册

```csharp
ModelRegistry.Register(
    "new-model",
    (packageRoot, config) => new NewModelPredictor(packageRoot, config));
```

注册后 JSON 中设置：

```json
{
  "modelType": "new-model"
}
```

即可由 `PredictionService` 自动创建对应 predictor。

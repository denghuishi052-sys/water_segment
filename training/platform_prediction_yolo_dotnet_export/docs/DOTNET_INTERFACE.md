# .NET Interface Documentation

本文档面向 .NET 开发集成使用，说明当前水体分割模型包的稳定接口、配置项、返回结果、大图处理逻辑和后续模型扩展方式。

如需查看单张图片从输入到输出的完整执行链路，请阅读 `docs/IMAGE_INPUT_OUTPUT_FLOW.md`。

当前交付包为纯 .NET + ONNX Runtime 版本，不依赖 Python 环境，不包含 Python 脚本、SAM3 或 LoRA 运行资产。

## 1. 项目定位

当前包提供一个图片分割接口：

- 输入：本地图片文件，或由调用方加载后的 `Image<Rgb24>`。
- 输出：二值 mask、叠加图 overlay、预测面积、预测面积比例、当前模型类型。
- 当前模型：YOLOv8-seg ONNX。
- 扩展方式：后续新增模型时，实现统一接口 `IImageSegmentationPredictor`，并通过 `modelType` 注册到工厂。

稳定集成入口是：

```csharp
PredictionService
```

模型扩展接口是：

```csharp
IImageSegmentationPredictor
```

## 2. 运行环境

类库项目：

```text
dotnet/src/WaterSegmentation.PlatformPrediction/WaterSegmentation.PlatformPrediction.csproj
```

目标框架：

```text
net9.0
```

NuGet 依赖：

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.22.1" />
<PackageReference Include="SixLabors.ImageSharp" Version="3.1.11" />
```

当前默认使用 `Microsoft.ML.OnnxRuntime` 的标准 CPU session 创建方式：

```csharp
new InferenceSession(modelPath)
```

配置中的 `providers` 字段目前作为配置保留项，当前代码没有显式创建 CUDA provider。若后续需要 GPU/CUDA，需要在 `OnnxYoloSegmenter` 中扩展 `SessionOptions`。

## 3. 推荐目录结构

交付包关键结构：

```text
platform_prediction_yolo_dotnet_export/
  configs/
    yolo_only.json
  dotnet/
    WaterSegmentation.PlatformPrediction.sln
    src/
      WaterSegmentation.PlatformPrediction/
        PredictionService.cs
        PredictionAbstractions.cs
        PredictionConfig.cs
        YoloOnnxSegmentationPredictor.cs
        OnnxYoloSegmenter.cs
        YoloV8SegPostprocessor.cs
        MaskPostprocessor.cs
        ImageOutput.cs
        Preprocessor.cs
  onnx/
    floodnet_binary_aug_yolov8m_1024.onnx
```

调用方集成时至少需要：

- `WaterSegmentation.PlatformPrediction` 类库。
- `configs/yolo_only.json`。
- `onnx/floodnet_binary_aug_yolov8m_1024.onnx`。

## 4. 核心接口

### 4.1 PredictionService

文件：

```text
dotnet/src/WaterSegmentation.PlatformPrediction/PredictionService.cs
```

职责：

- 作为业务侧稳定入口。
- 接收图片路径。
- 调用内部模型 predictor。
- 保存 mask 和 overlay。
- 计算预测面积和面积比例。

构造函数：

```csharp
public PredictionService(string packageRoot, PredictionConfig config)
```

参数说明：

| 参数 | 类型 | 说明 |
|---|---|---|
| `packageRoot` | `string` | 模型包根目录。相对路径模型文件会基于该目录解析。 |
| `config` | `PredictionConfig` | 预测配置对象。 |

该构造函数内部会调用：

```csharp
ImageSegmentationPredictorFactory.Create(packageRoot, config)
```

根据 `config.ModelType` 创建具体模型实现。

另一个构造函数：

```csharp
public PredictionService(IImageSegmentationPredictor predictor)
```

用于调用方自行注入模型实现，适合依赖注入或高级扩展场景。

### 4.2 PredictFile

```csharp
public PredictionResult PredictFile(
    string imagePath,
    string outputDir,
    bool saveOverlay = true)
```

参数说明：

| 参数 | 类型 | 说明 |
|---|---|---|
| `imagePath` | `string` | 输入图片路径。支持 ImageSharp 可读取的常见图片格式。 |
| `outputDir` | `string` | 输出目录。不存在时会自动创建。 |
| `saveOverlay` | `bool` | 是否保存叠加图。默认 `true`。 |

输出文件命名：

假设输入图片为：

```text
abc.png
```

则输出：

```text
abc_pred.png
abc_overlay.jpg
```

返回：

```csharp
PredictionResult
```

### 4.3 PredictionResult

定义：

```csharp
public sealed record PredictionResult(
    string MaskPath,
    string OverlayPath,
    int MaskArea,
    float PredAreaRatio,
    string ModelType);
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `MaskPath` | `string` | 二值 mask 图片路径。前景为 255，背景为 0。 |
| `OverlayPath` | `string` | 叠加图路径。若 `saveOverlay=false`，路径仍会按规则生成，但文件不会保存。 |
| `MaskArea` | `int` | mask 前景像素数量。 |
| `PredAreaRatio` | `float` | `MaskArea / (image.Width * image.Height)`。 |
| `ModelType` | `string` | 实际使用的模型类型。当前为 `yolo-onnx`。 |

## 5. 模型扩展接口

文件：

```text
dotnet/src/WaterSegmentation.PlatformPrediction/PredictionAbstractions.cs
```

### 5.1 IImageSegmentationPredictor

定义：

```csharp
public interface IImageSegmentationPredictor : IDisposable
{
    string ModelType { get; }

    SegmentationPrediction Predict(PredictionRequest request);
}
```

职责：

- 封装具体模型推理逻辑。
- 输入统一为图片请求。
- 输出统一为分割预测结果。

调用约定：

- `Predict` 不负责保存文件。
- `Predict` 只负责返回 mask、概率图和模型元信息。
- 文件输出由 `PredictionService` 统一处理。

这样做的目的是：后续更换模型时，不影响外层文件输入输出接口。

### 5.2 PredictionRequest

定义：

```csharp
public sealed record PredictionRequest(
    Image<Rgb24> Image,
    string? ImagePath = null,
    IReadOnlyDictionary<string, string>? Metadata = null);
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `Image` | `Image<Rgb24>` | 已加载图片。 |
| `ImagePath` | `string?` | 原始图片路径，可为空。 |
| `Metadata` | `IReadOnlyDictionary<string, string>?` | 预留扩展元数据。 |

### 5.3 SegmentationPrediction

定义：

```csharp
public sealed record SegmentationPrediction(
    byte[,] Mask,
    float[,]? Probability,
    string ModelType,
    IReadOnlyDictionary<string, string>? Metadata = null);
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `Mask` | `byte[,]` | 二值 mask。数组维度为 `[height, width]`，前景为 1，背景为 0。 |
| `Probability` | `float[,]?` | 概率图。数组维度为 `[height, width]`，取值通常为 0 到 1。某些模型没有概率图时可为 `null`。 |
| `ModelType` | `string` | 模型类型。 |
| `Metadata` | `IReadOnlyDictionary<string, string>?` | 预留扩展元数据。 |

### 5.4 ImageSegmentationPredictorFactory

定义：

```csharp
public static class ImageSegmentationPredictorFactory
{
    public static IImageSegmentationPredictor Create(
        string packageRoot,
        PredictionConfig config)
    {
        return config.ModelType.ToLowerInvariant() switch
        {
            "yolo-onnx" or "yolo" => new YoloOnnxSegmentationPredictor(packageRoot, config),
            _ => throw new NotSupportedException($"Unsupported modelType: {config.ModelType}")
        };
    }
}
```

职责：

- 根据 `config.ModelType` 创建具体 predictor。
- 当前支持：
  - `yolo-onnx`
  - `yolo`

后续新增模型时，在此处注册新的 `modelType`。

## 6. 配置文件说明

重要约定：模型参数、预处理参数、后处理参数和输出参数不应写死在业务代码中。当前包保留 YOLO 强类型字段用于兼容已验证版本，同时也提供 `pipeline`、`modelOptions`、`preprocessOptions`、`postprocessOptions`、`outputOptions` 和 `ExtensionData` 用于未来处理链扩展。详细说明见 `docs/DOTNET_CONFIG_EXTENSION.md`。

默认配置：

```text
configs/yolo_only.json
```

当前内容：

```json
{
  "modelType": "yolo-onnx",
  "modelPath": "onnx/floodnet_binary_aug_yolov8m_1024.onnx",
  "imageSize": 1024,
  "confidenceThreshold": 0.25,
  "iouThreshold": 0.5,
  "maskThreshold": 0.65,
  "maskBoxExpandRatio": 0.5,
  "minAreaRatio": 0.0005,
  "morphClose": true,
  "suppressLargeBorderComponents": false,
  "largeBorderComponentMinAreaRatio": 0.15,
  "largeBorderComponentMarginRatio": 0.03,
  "maxDetections": 300,
  "classCount": 1,
  "maskPrototypeCount": 32,
  "cascade": {
    "enabled": false,
    "modelPaths": [],
    "imageSize": 640,
    "confidenceThreshold": 0.15,
    "minAspectRatio": 1.15,
    "triggerAreaRatio": 0.005,
    "minConsensusAreaRatio": 0.005
  },
  "tiling": {
    "tileSize": 1024,
    "overlapPixels": 256,
    "weight": 0.7,
    "stitchMode": "max"
  },
  "providers": [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
  ]
}
```

### 6.1 基础模型字段

| 字段 | 类型 | 当前默认值 | 说明 |
|---|---:|---:|---|
| `modelType` | `string` | `yolo-onnx` | 模型类型，用于工厂创建 predictor。 |
| `modelPath` | `string` | `onnx/floodnet_binary_aug_yolov8m_1024.onnx` | 模型路径。相对路径基于 `packageRoot`。 |
| `imageSize` | `int` | `1024` | YOLO 输入尺寸。 |
| `classCount` | `int` | `1` | 类别数量。当前为二分类水体分割。 |
| `maskPrototypeCount` | `int` | `32` | YOLOv8-seg prototype mask 数量。需与模型输出一致。 |

### 6.2 YOLO 检测与 mask 字段

| 字段 | 类型 | 当前默认值 | 说明 |
|---|---:|---:|---|
| `confidenceThreshold` | `float` | `0.25` | 检测置信度阈值。 |
| `iouThreshold` | `float` | `0.5` | NMS IoU 阈值。 |
| `maxDetections` | `int` | `300` | 最大保留检测数。 |
| `maskThreshold` | `float` | `0.65` | 概率图转二值 mask 的阈值。越高越保守。 |
| `maskBoxExpandRatio` | `float` | `0.5` | 扩大检测框参与 mask 采样，避免水域被过紧检测框截断。 |

### 6.3 后处理字段

| 字段 | 类型 | 当前默认值 | 说明 |
|---|---:|---:|---|
| `minAreaRatio` | `float` | `0.0005` | 小连通域过滤阈值，占整图面积比例。 |
| `morphClose` | `bool` | `true` | 是否做形态学闭运算。 |
| `suppressLargeBorderComponents` | `bool` | `false` | 是否抑制贴边大连通域。当前默认关闭，避免误删真实大水面。 |
| `largeBorderComponentMinAreaRatio` | `float` | `0.15` | 贴边大连通域抑制面积阈值。仅在上一项开启时生效。 |
| `largeBorderComponentMarginRatio` | `float` | `0.03` | 贴边判断边距比例。仅在抑制开启时生效。 |

### 6.4 大图切片字段

| 字段 | 类型 | 当前默认值 | 说明 |
|---|---:|---:|---|
| `tiling.tileSize` | `int` | `1024` | 切片尺寸。 |
| `tiling.overlapPixels` | `int` | `256` | 切片重叠像素。 |
| `tiling.weight` | `float` | `0.7` | 预留字段，当前拼接代码未使用。 |
| `tiling.stitchMode` | `string` | `max` | 拼接策略说明字段。当前实现为 max。 |

## 7. 大图处理逻辑

当前 `YoloOnnxSegmentationPredictor` 中的大图处理逻辑如下：

1. 判断图片最长边是否大于 `tiling.tileSize`。
2. 若不大于，则整图进入 YOLO。
3. 若大于，则按滑窗切片。
4. 每个 tile 单独推理。
5. 将 tile 概率图拼回原图尺寸。
6. 重叠区域取最大概率值。
7. 拼接后的整图概率图统一做阈值化和后处理。

当前步长：

```text
step = tileSize - overlapPixels
```

默认配置：

```text
tileSize = 1024
overlapPixels = 256
step = 768
```

重叠区域拼接方式：

```csharp
full[y, x] = Math.Max(full[y, x], tileProb[y, x]);
```

注意：

- 当前策略保证大图能完整处理。
- 大图耗时会明显增加。
- 已测试桌面大图，`pakistan_oli2_2022240_lrg.jpg` 耗时约 358 秒，`thailand_tmo_2011346_lrg.jpg` 耗时约 173 秒。
- 若生产环境需要高吞吐，应增加快速模式，例如降采样预筛、ROI 二次精推、并行 tile 推理、GPU provider 或中心权重融合。

## 8. 最小集成示例

以下不是 demo 工程，只是接口调用方式说明：

```csharp
using WaterSegmentation.PlatformPrediction;

var packageRoot = @"D:\project\water_segment\platform_prediction_yolo_dotnet_export";
var configPath = Path.Combine(packageRoot, "configs", "yolo_only.json");
var imagePath = @"C:\path\to\input.png";
var outputDir = @"C:\path\to\output";

var config = PredictionConfig.Load(configPath);

using var service = new PredictionService(packageRoot, config);
var result = service.PredictFile(imagePath, outputDir);

Console.WriteLine(result.MaskPath);
Console.WriteLine(result.OverlayPath);
Console.WriteLine(result.MaskArea);
Console.WriteLine(result.PredAreaRatio);
Console.WriteLine(result.ModelType);
```

## 9. 直接使用 Predictor

如果调用方不希望 `PredictionService` 保存文件，可以直接使用 predictor：

```csharp
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WaterSegmentation.PlatformPrediction;

var config = PredictionConfig.Load(configPath);

using var predictor = ImageSegmentationPredictorFactory.Create(packageRoot, config);
using var image = Image.Load<Rgb24>(imagePath);

var prediction = predictor.Predict(new PredictionRequest(image, imagePath));

byte[,] mask = prediction.Mask;
float[,]? probability = prediction.Probability;
string modelType = prediction.ModelType;
```

这种方式适合：

- 调用方自己管理输出。
- 服务端返回 mask 数组或自定义编码。
- 与现有业务系统的图片存储逻辑对接。

## 10. 新增模型的接入方式

假设后续新增一个模型类型：

```text
new-model-onnx
```

推荐步骤：

### 10.1 新增 predictor 类

```csharp
public sealed class NewModelSegmentationPredictor : IImageSegmentationPredictor
{
    public string ModelType => "new-model-onnx";

    public NewModelSegmentationPredictor(string packageRoot, PredictionConfig config)
    {
        // 初始化模型、读取路径、创建 session
    }

    public SegmentationPrediction Predict(PredictionRequest request)
    {
        // 1. 前处理 request.Image
        // 2. 模型推理
        // 3. 后处理为 byte[,] mask
        // 4. 可选返回 float[,] probability
        return new SegmentationPrediction(mask, probability, ModelType);
    }

    public void Dispose()
    {
        // 释放 session 或其它资源
    }
}
```

### 10.2 注册到 Factory

```csharp
return config.ModelType.ToLowerInvariant() switch
{
    "yolo-onnx" or "yolo" => new YoloOnnxSegmentationPredictor(packageRoot, config),
    "new-model-onnx" => new NewModelSegmentationPredictor(packageRoot, config),
    _ => throw new NotSupportedException($"Unsupported modelType: {config.ModelType}")
};
```

### 10.3 新增配置文件

例如：

```json
{
  "modelType": "new-model-onnx",
  "modelPath": "onnx/new_model.onnx",
  "imageSize": 1024,
  "maskThreshold": 0.5,
  "tiling": {
    "tileSize": 1024,
    "overlapPixels": 256,
    "weight": 0.7,
    "stitchMode": "max"
  }
}
```

只要外层仍然使用 `PredictionService`，调用方业务代码无需关心内部模型变化。

## 11. 输出 mask 约定

内部 mask：

```text
byte[,] mask
```

维度：

```text
mask[height, width]
```

取值：

```text
0 = background
1 = foreground
```

保存为图片时：

```text
0 -> 0
1 -> 255
```

即 `_pred.png` 为单通道二值图。

## 12. 线程与资源管理

`PredictionService` 和 `IImageSegmentationPredictor` 均实现资源释放链路：

```csharp
using var service = new PredictionService(packageRoot, config);
```

或：

```csharp
using var predictor = ImageSegmentationPredictorFactory.Create(packageRoot, config);
```

当前 `YoloOnnxSegmentationPredictor` 内部持有 `OnnxYoloSegmenter`，`OnnxYoloSegmenter` 内部持有 `InferenceSession`。

必须释放：

```text
PredictionService -> predictor -> InferenceSession
```

服务端长期运行时建议：

- 不要每张图片都重新创建 `PredictionService`。
- 可以按配置创建单例或池化 predictor。
- 注意并发调用同一个 `InferenceSession` 前，需要根据 ONNX Runtime 使用策略做压测确认。

## 13. 错误处理

常见异常来源：

| 场景 | 可能异常 |
|---|---|
| 图片路径不存在 | ImageSharp 加载异常 |
| 模型路径不存在 | ONNX Runtime session 创建异常 |
| `modelType` 未注册 | `NotSupportedException` |
| ONNX 输出结构不符合 YOLOv8-seg | `InvalidOperationException` |
| `maskPrototypeCount` 与模型输出不一致 | `InvalidOperationException` |

建议业务层捕获：

```csharp
try
{
    var result = service.PredictFile(imagePath, outputDir);
}
catch (Exception ex)
{
    // 记录 imagePath/config/modelType/异常堆栈
    throw;
}
```

## 14. 当前验证结果

验证日期：2026-07-03

构建：

```text
dotnet build WaterSegmentation.PlatformPrediction.sln -c Release
0 warnings, 0 errors
```

接口批量测试：

```text
C:\Users\17473\Desktop
8 images
8 passed
```

典型输出：

```text
微信图片_20260602144629_184_4.png
maskArea=56319
predAreaRatio=0.669388
modelType=yolo-onnx
```

大图测试：

```text
pakistan_oli2_2022240_lrg.jpg
predAreaRatio=0.072406
elapsedSeconds=358.407

thailand_tmo_2011346_lrg.jpg
predAreaRatio=0.223829
elapsedSeconds=173.244
```

## 15. 集成建议

建议 .NET 开发侧按以下方式接入：

1. 将 `WaterSegmentation.PlatformPrediction` 作为类库引用。
2. 部署时保留 `configs/` 和 `onnx/` 目录。
3. 业务层只依赖 `PredictionService` 或 `IImageSegmentationPredictor`。
4. 不要在业务层直接依赖 `YoloOnnxSegmentationPredictor`，除非确实需要绑定当前模型。
5. 新模型通过 `IImageSegmentationPredictor` 扩展，不改外层图片输入输出。
6. 大图批处理场景需要额外评估耗时，并考虑 GPU、tile 并行或快速模式。

## 16. 流程优化实现状态

当前版本已补充：

- `PredictionConfigValidator`：配置校验。
- `PipelineRunner`：pipeline trace 骨架。
- `largeImage.mode`：`full_quality`、`fast_preview`、`roi_refine`。
- `ModelRegistry`：模型注册表。
- `PredictionOutput`：标准化 `result.json` 和 `summary.csv` 输出。
- `PredictionResult.Metadata`：诊断信息返回。

这些能力均通过配置驱动，不要求业务代码写死具体模型参数。

# Water Segment

面向道路积水、洪水与卫星水体的图像分割项目。

项目最初采用 YOLOv8-seg，目前研发主模型已演进为 **DualContextWaterNet v2 large 1024**。YOLOv8-seg 路线仍作为道路积水、FloodNet 快速推理以及现有 .NET 交付的重要基线保留。

完整的模型结构、历史实验、部署状态和交接事项见 [项目交接说明](docs/PROJECT_HANDOVER.md)。

## 当前模型

当前研发主模型：

```text
DualContextWaterNet v2 large 1024
```

| 项目 | 路径/数值 |
|---|---|
| PyTorch checkpoint | `runs/dual_context_water/v2_large_1024_finetune/best.pt` |
| ONNX | `onnx/dual_context_water_v2_large_1024_candidate.onnx` |
| 配置 | `configs/dual_context_water_v2_large_1024_finetune.yaml` |
| ONNX SHA256 | `60AE3651F3F556247640A604CED021638A58A13AE4B576A8704989DB3761B157` |
| 本地输入 | `local_image: (1,3,1024,1024)` |
| 全局输入 | `global_context: (1,4,512,512)` |
| 输出 | `water_logits`、`global_logits`、`quality_logits` |

该模型不是 YOLO。它直接输出逐像素水体 logit，不使用检测框、NMS 或 prototype mask。

### 模型结构

```text
1024×1024 local RGB
    → ConvNeXt-Small local encoder
                              ┐
512×512 global RGB + ROI mask │
    → ConvNeXt-Tiny encoder    │
    → ContextFiLM 调制局部特征 ┘
    → 256-channel GroupNorm FPN decoder
    → dense water logits
```

局部分支负责水体边界和小区域细节；全局分支同时提供整幅图语义以及当前 tile 在全图中的位置。大图推理采用 1024 tile、256 overlap 和 Hann 加权融合。

### 当前验证结果

在 220 个卫星验证 tile 上，固定阈值 `sigmoid(logits) >= 0.5` 的 micro 指标：

| IoU | Dice/F1 | Precision | Recall |
|---:|---:|---:|---:|
| **0.6144** | **0.7612** | **0.8288** | **0.7038** |

相对初始化 checkpoint，当前微调显著降低了误报，但 Recall 有所下降。生产推理使用 low `0.40`、high `0.65` 的 hysteresis，和上表固定阈值不是同一评估口径。

### DualContext 快速推理

本机实际使用的 Python 环境：

```text
C:\Users\17473\miniforge3\envs\torch_env\python.exe
```

```powershell
C:\Users\17473\miniforge3\envs\torch_env\python.exe scripts\29_predict_dual_context_onnx.py `
  --model onnx\dual_context_water_v2_large_1024_candidate.onnx `
  --image "<输入图片路径>" `
  --output_dir runs\dual_context_prediction `
  --low_threshold 0.40 `
  --high_threshold 0.65
```

输出：

```text
<stem>_pred.png
<stem>_probability.png
<stem>_overlay.jpg
```

评估 checkpoint：

```powershell
C:\Users\17473\miniforge3\envs\torch_env\python.exe scripts\31_evaluate_dual_context.py `
  --checkpoint runs\dual_context_water\v2_large_1024_finetune\best.pt `
  --manifest data\satellite_adaptation_1024_v1\satellite_val.csv
```

导出 ONNX：

```powershell
C:\Users\17473\miniforge3\envs\torch_env\python.exe scripts\28_export_dual_context_onnx.py `
  --checkpoint runs\dual_context_water\v2_large_1024_finetune\best.pt `
  --output onnx\dual_context_water_v2_large_1024_candidate.onnx `
  --local_size 1024 `
  --global_size 512
```

## 此前的 YOLOv8-seg 路线

项目早期以 YOLOv8-seg 为核心，先后尝试过：

```text
YOLOv8n-seg
YOLOv8s-seg
YOLOv8m-seg
YOLOv8l-seg
YOLOv8x-seg
```

相关实验包括道路积水基础模型、hard positive/hard negative 微调、FloodNet 二分类/三分类、多尺度裁剪、GF-FloodNet、Sen2GF3、多域训练、1024 输入、tiled inference、cascade 兜底以及 SAM3 精修。

早期数据与训练入口：

| 步骤 | 脚本 |
|---|---|
| 数据检查 | `scripts/01_check_dataset.py` |
| train/val/test 划分 | `scripts/02_split_dataset.py` |
| mask 转 YOLO polygon | `scripts/03_convert_mask_to_yolo_seg.py` |
| YOLO 训练 | `scripts/04_train_yolov8.py` |
| 独立 test 评估 | `scripts/05_eval_test.py` |
| 困难样本挖掘 | `scripts/07_hard_sample_mining.py` |
| 困难样本微调 | `scripts/08_finetune_hard_samples.py` |

### 当前保留的 YOLO ONNX

```text
onnx/floodnet_binary_aug_yolov8m_1024.onnx
```

Python 配置：`configs/onnx_platform.yaml`  
.NET 配置：`platform_prediction_yolo_dotnet_export/configs/yolo_only.json`

Python 生产推理位于 `waterseg_platform/`，运行时不依赖 PyTorch 或 Ultralytics，自行实现 letterbox、NMS、YOLOv8 mask 解码、大图切片、后处理和结果保存。

```powershell
C:\Users\17473\miniforge3\envs\torch_env\python.exe -m waterseg_platform.cli `
  --config configs\onnx_platform.yaml `
  single --image "<输入图片路径>" --output_dir runs\onnx_cli_smoke
```

### 代表性 YOLO 结果

838 张道路积水独立 test 集：

| 模型 | IoU | Dice/F1 | Precision | Recall | FP 图片 | FN 图片 |
|---|---:|---:|---:|---:|---:|---:|
| Base | **0.8539** | **0.9212** | 0.9172 | **0.9253** | **4** | 8 |
| Balanced freeze | 0.7312 | 0.8447 | 0.9225 | 0.7790 | 0 | 80 |
| Recall hard-sample | 0.8415 | 0.9139 | **0.9242** | 0.9038 | 7 | **3** |

Base 模型综合指标最好；Recall hard-sample 减少了完全漏检图片，但没有同时提升所有像素指标。

60 张 FloodNet test 集：

| 模式 | IoU | Dice/F1 | Precision | Recall | 平均耗时 |
|---|---:|---:|---:|---:|---:|
| YOLO-only | 0.6645 | 0.7984 | **0.8124** | 0.7849 | **82.5 ms** |
| YOLO + SAM3 LoRA + Selector | **0.7297** | **0.8437** | 0.8096 | **0.8808** | 3820.3 ms |

SAM3 精修提高了 IoU 和 Recall，但耗时约为 YOLO-only 的 46.3 倍，因此更适合离线高质量模式。

## 两条路线如何选择

| 场景 | 建议路线 |
|---|---|
| 大幅卫星水体 | DualContext v2 large 1024 |
| 道路积水、FloodNet、快速预览 | YOLOv8m-seg 1024 ONNX |
| 离线高质量边界与召回 | YOLO + SAM3 LoRA + Selector |
| 现有 .NET YOLO-only 应用 | 保持 YOLO 路线，除非完成 DualContext 迁移验收 |

DualContext 与 YOLO 的指标来自不同数据集，不能直接横向比较。模型选择应以目标业务数据、误报/漏报要求和目标硬件为准。

## .NET 状态

主要工程：

```text
platform_prediction_yolo_dotnet_export/dotnet/
```

```powershell
Set-Location platform_prediction_yolo_dotnet_export\dotnet
dotnet build WaterSegmentation.PlatformPrediction.sln -c Release
```

工程已包含 `yolo-onnx` 和 `dual-context-onnx` predictor。不过 `configs/dual_context.json` 当前仍指向较早的：

```text
onnx/dual_context_water_satellite_adapt_ian_to_ida_fp32.onnx
```

最新 DualContext candidate 尚未完成 .NET 全量 parity、阈值校准和正式性能验收。在这些工作完成前，不应直接覆盖现有 YOLO-only 交付包。

## 环境与验证

2026-08-17 已验证环境：

| 组件 | 版本/状态 |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.5.0+cu121 |
| CUDA | 12.1，可用 |
| Ultralytics | 8.4.56 |
| ONNX Runtime | 1.23.2 |
| OpenCV | 4.13.0 |
| .NET SDK | 9.0.308 |

已执行：

```text
Python 核心测试：42 passed
.NET Release 构建：0 warnings, 0 errors
DualContext ONNX checker：通过
DualContext 输出 shape 验证：通过
```

根目录 `requirements.txt` 是早期依赖清单，尚未完整覆盖后期模型依赖，也没有锁定版本。系统默认 Python 3.12 环境没有安装 PyTorch，训练和评估应使用 `torch_env`。

## 目录结构

```text
configs/                         训练与推理配置
data/                            原始数据、处理数据和 manifest
dual_context_water/              当前双上下文模型与推理
waterseg_platform/               YOLO ONNX Python 平台
training/sam3_lora/              SAM3 LoRA 与 selector 训练
scripts/                         数据、训练、评估、导出入口
tests/                           单元、集成与 parity 测试
onnx/                            ONNX 模型
runs/                            checkpoint、日志和实验结果
platform_prediction_yolo_dotnet_export/  当前主要 .NET 交付工程
docs/                            设计、运行和交接文档
```

## 文档索引

- [项目交接说明](docs/PROJECT_HANDOVER.md)
- [完整项目结构与实验结果](PROJECT_STRUCTURE.md)
- [YOLO ONNX Python 平台](waterseg_platform/README.md)
- [YOLO PT/ONNX 一致性记录](waterseg_platform/PARITY.md)
- [SAM3 LoRA/Selector 运行手册](docs/SAM3_LORA_SELECTOR_RUNBOOK.md)
- [.NET 接口说明](platform_prediction_yolo_dotnet_export/docs/DOTNET_INTERFACE.md)
- [.NET 图像输入输出流程](platform_prediction_yolo_dotnet_export/docs/IMAGE_INPUT_OUTPUT_FLOW.md)


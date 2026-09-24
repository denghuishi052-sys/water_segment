# Water Segment 项目交接说明

> 文档更新日期：2026-08-17  
> 项目目录：`D:\project\water_segment`  
> 本文用于替代此前以 YOLOv8-seg 为主的远端项目说明。

## 1. 项目现状

Water Segment 是一个面向道路积水、洪水和卫星水体的图像分割项目。项目最初采用 **YOLOv8-seg**，后续为解决大幅卫星影像中局部细节与全局场景难以兼顾的问题，开发了 **DualContextWaterNet 双上下文语义分割模型**。

目前项目的研发主模型是：

```text
DualContextWaterNet v2 large 1024
```

对应文件如下：

| 类型 | 路径 |
|---|---|
| PyTorch checkpoint | `runs/dual_context_water/v2_large_1024_finetune/best.pt` |
| ONNX 模型 | `onnx/dual_context_water_v2_large_1024_candidate.onnx` |
| 训练配置 | `configs/dual_context_water_v2_large_1024_finetune.yaml` |
| 模型代码 | `dual_context_water/model.py` |
| ONNX 推理代码 | `dual_context_water/inference.py` |
| 单图推理入口 | `scripts/29_predict_dual_context_onnx.py` |
| 评估入口 | `scripts/31_evaluate_dual_context.py` |
| ONNX 导出入口 | `scripts/28_export_dual_context_onnx.py` |

当前 ONNX 文件的校验信息：

```text
SHA256: 60AE3651F3F556247640A604CED021638A58A13AE4B576A8704989DB3761B157
大小:   339,409,636 bytes，约 323.7 MiB
```

需要特别区分两个概念：

- **当前研发主模型**是 DualContextWaterNet v2 large 1024；
- **现有 YOLO-only Python/.NET 交付链**仍然保留，并继续使用 YOLOv8-seg ONNX 模型。

因此，不能仅通过文件时间判断发布模型。对外交付前，应根据目标场景明确选择 DualContext 或 YOLO 路线，并使用对应的配置和后处理。

## 2. 当前模型：DualContextWaterNet v2 large 1024

### 2.1 为什么改用双上下文模型

YOLOv8-seg 在道路积水和常规 FloodNet 图像上表现稳定，但用于大幅卫星影像时存在几个限制：

1. 将整幅大图缩放到固定输入尺寸会丢失小水体和细窄边界；
2. 仅使用局部 tile 推理又容易丢失河流、海岸线和城市区域的全局语义；
3. YOLO 的检测框、NMS 和 prototype mask 解码更适合实例式目标，不完全适合连续大面积水体；
4. 大图切片时，局部纹理相似的阴影、道路、屋顶和植被容易产生误报。

DualContextWaterNet 的目标是让模型在处理局部高分辨率 tile 时，同时看到整幅图的缩略上下文和当前 tile 在全图中的位置。

### 2.2 模型输入与输出

该模型不是 YOLO，不包含检测框、anchor、NMS 或 prototype mask。

模型有两个输入：

| 输入 | Shape | 内容 |
|---|---|---|
| `local_image` | `(1, 3, 1024, 1024)` | 当前高分辨率局部 tile 的 RGB 图像 |
| `global_context` | `(1, 4, 512, 512)` | 整幅图缩略 RGB，加一层当前 tile 的 ROI 位置 mask |

模型有三个输出：

| 输出 | Shape | 内容 |
|---|---|---|
| `water_logits` | `(1, 1, 1024, 1024)` | 当前 tile 的逐像素水体 logit |
| `global_logits` | `(1, 1, 512, 512)` | 全局辅助分割输出，主要用于训练约束 |
| `quality_logits` | `(1, 1)` | 当前样本的质量/前景辅助判断 |

生产推理主要使用 `water_logits`，经过 sigmoid 得到逐像素概率图。

### 2.3 网络结构

```text
局部输入：1024×1024 RGB
    ↓
ConvNeXt-Small 局部编码器
    ↓
四层局部特征金字塔
                         ┐
全局输入：512×512 RGB + ROI mask
    ↓                    │
ConvNeXt-Tiny 全局编码器 │
    ↓                    │
全局上下文向量           │
    └── ContextFiLM 调制局部多尺度特征
                         ┘
    ↓
256 通道 GroupNorm FPN 解码器
    ↓
1024×1024 dense water logits
```

其中：

- 局部分支保留水体边界、小区域和纹理细节；
- 全局分支提供整幅图的场景语义；
- ROI mask 告诉全局分支当前 tile 位于整幅图的哪个位置；
- ContextFiLM 使用全局上下文调制局部多尺度特征；
- FPN 解码器直接输出逐像素水体概率，不再执行 YOLO mask 解码。

### 2.4 大图推理方式

实现位于 `dual_context_water/inference.py`。

默认参数：

```text
tile size:       1024
global size:     512
tile overlap:    256
stitching:       Hann 加权融合
low threshold:   0.40
high threshold:  0.65
```

推理流程如下：

1. 将整幅输入图缩放为 512×512，形成全局 RGB；
2. 将原图切成 1024×1024 tile，相邻 tile 重叠 256 像素；
3. 对每个 tile 生成其在全局图中的 ROI mask；
4. 同时输入 local RGB 和 global RGB + ROI；
5. 将每个 tile 的概率图使用 Hann 权重融合回原图；
6. 使用双阈值 hysteresis 生成最终二值 mask：
   - 高于 `0.65` 的像素作为强前景；
   - 高于 `0.40` 且与强前景连通的区域作为弱前景保留；
   - 没有强前景支持的弱响应被移除。

相比简单阈值，hysteresis 能保留与高置信水体相连的低置信边缘，同时抑制孤立误报。

### 2.5 训练数据与策略

当前模型使用以下 manifest：

```text
训练: data/satellite_adaptation_1024_v1/combined_train.csv
验证: data/satellite_adaptation_1024_v1/satellite_val.csv
```

训练集共 2830 条记录：

```text
base 样本:       2770
satellite 样本:    60
```

由于卫星样本数量很少，训练代码没有按原始比例直接采样，而是通过 `WeightedRandomSampler` 将卫星样本的目标采样比例提高到 `45%`。

关键训练参数：

| 参数 | 当前值 |
|---|---:|
| local backbone | `convnext_small` |
| global backbone | `convnext_tiny` |
| decoder channels | `256` |
| local size | `1024` |
| global size | `512` |
| batch size | `1` |
| gradient accumulation | `4` |
| learning rate | `5e-6` |
| epochs | `100` |
| satellite batch fraction | `0.45` |
| AMP dtype | `bfloat16` |

模型从以下 checkpoint 初始化：

```text
runs/dual_context_water/v2_large_convnext_small_tiny/best.pt
```

训练损失包含：

- local segmentation loss；
- global auxiliary segmentation loss；
- quality classification loss；
- Tversky loss；
- boundary loss。

### 2.6 当前验证结果

在 220 个卫星验证 tile 上，使用 `sigmoid(logits) >= 0.5` 的固定阈值，micro 指标为：

| IoU | Dice/F1 | Precision | Recall |
|---:|---:|---:|---:|
| **0.6144** | **0.7612** | **0.8288** | **0.7038** |

与初始化 checkpoint 中记录的指标相比：

| 指标 | 初始化 | 当前最佳 | 变化 |
|---|---:|---:|---:|
| IoU | 0.5523 | 0.6144 | +6.21 个百分点 |
| Dice | 0.7116 | 0.7612 | +4.96 个百分点 |
| Precision | 0.6200 | 0.8288 | +20.87 个百分点 |
| Recall | 0.8350 | 0.7038 | -13.12 个百分点 |

这次微调最明显的效果是减少卫星场景中的误报，但模型也变得更加保守。若业务更重视洪水漏检，需要在业务验证集上重新校准阈值，或调整 Tversky、负样本权重后再训练。

验证和导出记录位于：

```text
runs/dual_context_water/v2_large_1024_finetune/finalize.log
```

注意：训练执行了 100 个 epoch，但 `best.pt` 按验证 Dice 保存。`results.csv` 的最后一行不是最佳模型指标，不能用最后一个 epoch 的数值替代上述结果。

### 2.7 当前性能记录

2026-08-13 在本机 CUDA 环境中，对单张 1024×1024 图像进行 1 次预热和 5 次测量：

| 指标 | 延迟 |
|---|---:|
| 平均 | 195.1 ms |
| P50 | 192.9 ms |
| 最快 | 187.2 ms |
| 最慢 | 207.6 ms |

记录文件：

```text
tmp/dual_context_v2_large_1024_latency_20260813.json
```

该结果只是单 tile 本机抽测，不是完整性能基准。大图耗时会随 tile 数增加，正式交付需要在目标硬件和典型分辨率上重新测量 P50/P95。

## 3. 之前使用的 YOLOv8-seg 模型

### 3.1 YOLO 路线的项目定位

项目早期以 YOLOv8-seg 为核心，用于道路积水和 FloodNet/GF-FloodNet 图像分割。根目录 `README.md` 中记录的主要是这一阶段的流程。

完整处理链包括：

```text
原始 image/mask
    ↓
数据检查与 train/val/test 划分
    ↓
mask 转 YOLO segmentation polygon
    ↓
YOLOv8-seg 训练
    ↓
独立 test 评估
    ↓
hard sample mining
    ↓
二阶段微调
    ↓
ONNX 导出与 Python/.NET 部署
```

相关代码：

| 功能 | 入口 |
|---|---|
| 数据检查 | `scripts/01_check_dataset.py` |
| 数据划分 | `scripts/02_split_dataset.py` |
| mask 转 YOLO 标签 | `scripts/03_convert_mask_to_yolo_seg.py` |
| YOLO 训练 | `scripts/04_train_yolov8.py` |
| test 评估 | `scripts/05_eval_test.py` |
| 困难样本挖掘 | `scripts/07_hard_sample_mining.py` |
| 困难样本微调 | `scripts/08_finetune_hard_samples.py` |
| ONNX 生产推理 | `waterseg_platform/` |

### 3.2 尝试过的 YOLOv8-seg 规模

仓库保留了多个 Ultralytics 预训练权重和实验模型：

```text
yolov8n-seg.pt
yolov8s-seg.pt
yolov8m-seg.pt
yolov8l-seg-fresh.pt
yolov8x-seg.pt
yolov8x-seg-fresh.pt
```

早期方案面向 RTX 3060，首先尝试 YOLOv8s-seg 和 YOLOv8n-seg；随着数据规模和输入分辨率增加，主要实验逐步转向 YOLOv8m、YOLOv8l 和 YOLOv8x。

仓库中的主要 YOLO 实验方向包括：

1. 道路积水基础模型；
2. hard positive/hard negative 二阶段微调；
3. FloodNet 二分类和三分类实验；
4. FloodNet 多尺度裁剪与二值增强；
5. GF-FloodNet、Sen2GF3 和多域联合训练；
6. flood water 与 natural water 分类尝试；
7. 1024 输入和 tiled inference；
8. YOLO 主模型加 cascade 召回兜底；
9. YOLO mask 作为 SAM3 proposal 的初始先验。

### 3.3 当前保留的 YOLO ONNX 基线

目前 Python 和 .NET YOLO-only 链使用的主要模型是：

```text
onnx/floodnet_binary_aug_yolov8m_1024.onnx
```

Python 配置：

```text
configs/onnx_platform.yaml
```

.NET 配置：

```text
platform_prediction_yolo_dotnet_export/configs/yolo_only.json
```

Python ONNX 平台位于 `waterseg_platform/`。它不依赖 PyTorch 或 Ultralytics 执行生产推理，自行实现：

- letterbox；
- NCHW 输入转换；
- YOLOv8 detection tensor 解码；
- confidence filtering 和 NMS；
- mask coefficient × prototype 解码；
- mask 回贴与形态学处理；
- 大图切片和概率拼接；
- cascade、边界抑制和结果保存。

这条链路被设计为可以一对一移植到 .NET。

### 3.4 YOLO 基础模型与 hard-sample 实验

在 838 张道路积水独立 test 集上，已有三组代表性结果：

| 模型 | IoU | Dice/F1 | Precision | Recall | FP 图片 | FN 图片 |
|---|---:|---:|---:|---:|---:|---:|
| Base | **0.8539** | **0.9212** | 0.9172 | **0.9253** | **4** | 8 |
| Balanced freeze | 0.7312 | 0.8447 | 0.9225 | 0.7790 | 0 | 80 |
| Recall hard-sample | 0.8415 | 0.9139 | **0.9242** | 0.9038 | 7 | **3** |

这些结果说明：

- Base 模型在该测试集上的综合 IoU、Dice 和 Recall 最稳定；
- Balanced freeze 虽然消除了 FP 图片，但产生了大量 FN，不适合作为默认模型；
- Recall hard-sample 将完全漏检图片从 8 张降低到 3 张，但总体 IoU 和像素 Recall 略有下降；
- hard-sample 训练不能概括为“所有指标都提升”，它改变的是误报与漏报之间的权衡。

### 3.5 FloodNet YOLO-only 与 SAM3 精修

在同一组 60 张 FloodNet test 图像上：

| 模式 | IoU | Dice/F1 | Precision | Recall | 平均耗时 |
|---|---:|---:|---:|---:|---:|
| YOLO-only | 0.6645 | 0.7984 | **0.8124** | 0.7849 | **82.5 ms** |
| YOLO + SAM3 LoRA + Selector | **0.7297** | **0.8437** | 0.8096 | **0.8808** | 3820.3 ms |

SAM3 精修使 IoU 提升 6.52 个百分点、Recall 提升 9.59 个百分点，但平均耗时约为 YOLO-only 的 46.3 倍。因此：

- YOLO-only 适合快速推理和交互式应用；
- YOLO + SAM3 更适合离线高质量精修；
- SAM3 不是当前 DualContext 模型的一部分。

结果来源：

```text
runs/final_test/sam3_lora_selector_frozen_rerun_validated/summary.json
```

### 3.6 YOLO 路线的优点与局限

优点：

- 训练和部署生态成熟；
- YOLOv8m 1024 的速度明显快于当前大型 DualContext 模型；
- Python 与 .NET 已有完整的前后处理实现；
- 适合道路积水、FloodNet 和相对独立的水体目标；
- 可以输出实例候选，方便与 SAM3 proposal 对接。

局限：

- mask 输出依赖检测框、NMS 和 prototype 解码；
- 连续大水域可能被检测框截断或拆分；
- 整图缩放会损失大幅卫星影像中的小目标；
- 仅切片推理缺少整幅图语义；
- mask box expansion、confidence、NMS 和 mask threshold 会共同影响结果，跨语言 parity 较复杂。

## 4. 当前模型与 YOLOv8-seg 的区别

| 对比项 | 当前 DualContext v2 large 1024 | 之前的 YOLOv8-seg |
|---|---|---|
| 主要场景 | 大幅卫星影像水体 | 道路积水、FloodNet、常规航拍图 |
| 任务形式 | Dense semantic segmentation | Detection-guided instance segmentation |
| 输入 | 局部 RGB + 全局 RGB/ROI | 单张 RGB 图像 |
| 主干 | ConvNeXt-Small + ConvNeXt-Tiny | YOLOv8 n/s/m/l/x backbone |
| 输出 | 每像素 water logits | boxes/classes/mask coefficients + prototypes |
| NMS | 不需要 | 需要 |
| mask prototype 解码 | 不需要 | 需要 |
| 大图上下文 | 模型原生包含全局上下文 | 通过整图 coarse pass 或外部融合补充 |
| 拼接 | Hann 加权概率融合 | max/Hann/软融合，取决于平台配置 |
| 当前 ONNX 大小 | 约 323.7 MiB | YOLOv8m 1024 明显更小 |
| 单图速度记录 | 约 195.1 ms/1024 tile | 约 82.5 ms/图，测试口径不同 |
| .NET 状态 | 已有 predictor，最新候选未完成发布验收 | YOLO-only 包和接口较完整 |

表中的延迟来自不同测试集和运行流程，只能用于理解量级，不能作为严格的模型速度排名。正式比较必须固定同一输入、provider、预热次数和计时范围。

## 5. 当前部署状态

### 5.1 Python DualContext

Python 已支持最新候选模型的大图推理。

示例：

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

### 5.2 Python YOLO ONNX

```powershell
C:\Users\17473\miniforge3\envs\torch_env\python.exe -m waterseg_platform.cli `
  --config configs\onnx_platform.yaml `
  single --image "<输入图片路径>" --output_dir runs\onnx_cli_smoke
```

### 5.3 .NET

当前主要 .NET 工程：

```text
platform_prediction_yolo_dotnet_export/dotnet/
```

构建命令：

```powershell
Set-Location platform_prediction_yolo_dotnet_export\dotnet
dotnet build WaterSegmentation.PlatformPrediction.sln -c Release
```

工程支持：

- `yolo-onnx`；
- `dual-context-onnx`；
- CLI 示例；
- WinForms 示例；
- mask、overlay、JSON 和 summary CSV 输出。

但 `configs/dual_context.json` 当前指向的是较早模型：

```text
onnx/dual_context_water_satellite_adapt_ian_to_ida_fp32.onnx
```

它尚未切换到最新的 `dual_context_water_v2_large_1024_candidate.onnx`。正式切换前需要完成：

1. 复制并校验最新 ONNX；
2. 更新 `modelPath`；
3. 对齐 Python 与 .NET 的 normalization、reflect padding、ROI mask 和 Hann 权重；
4. 对齐阈值与连通域后处理；
5. 完成多图 Python/.NET parity；
6. 测试 CPU/CUDA provider、性能和内存；
7. 验收通过后再将 candidate 标记为正式发布模型。

## 6. 环境说明

项目实际使用的 Python 环境：

```text
C:\Users\17473\miniforge3\envs\torch_env\python.exe
```

2026-08-17 验证版本：

| 组件 | 版本 |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.5.0+cu121 |
| CUDA | 12.1 |
| Ultralytics | 8.4.56 |
| ONNX Runtime | 1.23.2 |
| OpenCV | 4.13.0 |
| .NET SDK | 9.0.308 |

当前系统默认 `python` 是 Python 3.12.12，且没有安装 PyTorch。不要直接使用裸 `python` 运行训练和模型评估。

根目录 `requirements.txt` 是早期依赖清单，未完整包含 PyTorch、ONNX Runtime GPU、timm、SAM3 等后期依赖，也没有锁定版本。重建环境时，应以已验证的 `torch_env` 为基准补充 lockfile。

## 7. 验证状态

2026-08-17 已执行：

```text
Python 核心测试：42 passed
.NET Release 构建：0 warnings, 0 errors
DualContext ONNX checker：通过
DualContext 输出 shape 验证：通过
```

Python 快速测试命令：

```powershell
C:\Users\17473\miniforge3\envs\torch_env\python.exe -m pytest `
  tests\test_platform_config.py `
  tests\test_tiling.py `
  tests\test_preprocessing.py `
  tests\test_postprocessing.py -q
```

上述测试不等于发布验收。最新 DualContext candidate 仍缺少完整的 Python/.NET 多图 parity 和正式 P95 性能测试。

## 8. 指标阅读注意事项

1. 必须注明测试集，不能把道路积水、FloodNet 和卫星验证集的指标直接比较；
2. 必须注明 micro 或 macro；本文的主要汇总结果为 micro 像素指标；
3. 必须注明输入尺寸、阈值、tile 参数和后处理；
4. DualContext 验证使用固定阈值 0.5，生产 Python 默认使用 0.40/0.65 hysteresis，两者不是同一口径；
5. YOLO Python 与 .NET 默认阈值也不完全相同，parity 前必须统一配置；
6. `best.pt` 按最佳验证指标保存，不能用训练 CSV 最后一行判断最佳模型；
7. full-image 与 tiled 输出之间的 IoU 是预测一致性，不是相对真实标注的精度。

## 9. 接手后的优先工作

### P0：确定正式模型

首先确认实际交付场景：

- 道路积水/FloodNet 快速推理：优先保留 YOLO-only；
- 大幅卫星水体：优先验证 DualContext v2 large 1024；
- 离线高质量边界精修：可使用 YOLO + SAM3。

确定路线后冻结：

```text
模型文件
SHA256
配置文件
代码 commit
测试集版本
阈值与后处理
```

### P0：完成 DualContext 发布验收

- [ ] 最新 ONNX 集成到唯一 .NET 交付目录；
- [ ] Python/.NET 使用完全一致的预处理和后处理；
- [ ] 多图 parity：报告最小 mask IoU 和最大 mismatch ratio；
- [ ] 带 GT 业务测试：报告 Precision、Recall、IoU、Dice；
- [ ] 性能测试：报告 warm-up 后 P50/P95、tile 数、内存和显存；
- [ ] 覆盖空 mask、全前景、超大图、窄长图、中文路径和 CPU fallback；
- [ ] 从干净目录解压交付包并独立完成 smoke test。

### P1：整理项目入口

1. 使用本文替换远端旧说明；
2. 在根 `README.md` 顶部明确当前模型，并将早期内容标记为 YOLO 历史流程；
3. 为模型建立版本 manifest；
4. 补充可复现的 Python 环境锁定文件；
5. 明确 `platform_prediction_*` 三个相似目录中哪个是唯一发布源。

## 10. 关键文件索引

| 文件 | 说明 |
|---|---|
| `PROJECT_STRUCTURE.md` | 项目结构、模型和实验结果的详细汇总 |
| `README.md` | 早期 YOLOv8-seg 训练流程 |
| `dual_context_water/model.py` | 当前双上下文模型结构 |
| `dual_context_water/dataset.py` | 双上下文数据读取与预处理 |
| `dual_context_water/inference.py` | 当前 ONNX 大图切片推理 |
| `configs/dual_context_water_v2_large_1024_finetune.yaml` | 当前模型训练配置 |
| `waterseg_platform/README.md` | YOLO ONNX Python 平台说明 |
| `waterseg_platform/PARITY.md` | YOLO PT/ONNX 一致性记录 |
| `docs/SAM3_LORA_SELECTOR_RUNBOOK.md` | YOLO + SAM3 精修运行手册 |
| `docs/model_error_analysis_and_optimization.md` | 历史误差分析和优化建议 |
| `platform_prediction_yolo_dotnet_export/docs/DOTNET_INTERFACE.md` | .NET 接口说明 |
| `platform_prediction_yolo_dotnet_export/docs/IMAGE_INPUT_OUTPUT_FLOW.md` | .NET 图像处理链说明 |

## 11. 最终交接结论

项目当前已经从“YOLOv8-seg 道路积水识别工程”演进为“以 DualContextWaterNet 为当前卫星水体主模型，同时保留成熟 YOLOv8-seg 推理链”的多路线项目。

两条模型路线不是简单的新旧替换关系：

- DualContext 解决的是大幅卫星影像的局部细节和全局语义问题；
- YOLOv8-seg 在道路积水、FloodNet、快速推理和现有 .NET 交付方面仍有实际价值；
- SAM3 是基于 YOLO proposal 的高质量精修扩展，不属于当前 DualContext 主模型；
- 在最新 DualContext 完成 .NET parity 和正式性能验收前，现有 YOLO-only 交付包仍应保留，不能直接删除或覆盖。


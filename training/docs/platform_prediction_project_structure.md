# Water Segmentation Platform 预测功能项目结构整理

更新时间：2026-07-02  
工作目录：`D:\project\water_segment`

本文档整理当前工程中与“图片预测、tile 拼接、YOLO/SAM3-LoRA 输出 mask/overlay”相关的核心文件夹与模块。

---

## 1. 当前预测入口

### 1.1 命令行入口

```text
scripts/11_run_onnx_platform.py
waterseg_platform/cli.py
```

常用命令：

```bash
python scripts/11_run_onnx_platform.py ^
  --config configs/onnx_platform_sam3_lora_frozen.yaml ^
  --image "C:\path\to\image.png" ^
  --output_dir runs\your_output_dir ^
  --save_mask --save_overlay
```

如果只想跑 YOLO，不启用 SAM3-LoRA：

```bash
python scripts/11_run_onnx_platform.py ^
  --config configs/onnx_platform_sam3_lora_frozen.yaml ^
  --no_sam3 ^
  --image "C:\path\to\image.png" ^
  --output_dir runs\your_output_dir ^
  --save_mask --save_overlay
```

---

## 2. 当前主配置文件

### 2.1 生产/冻结配置

```text
configs/onnx_platform_sam3_lora_frozen.yaml
```

当前关键参数：

```yaml
model_path: onnx/floodnet_binary_aug_yolov8m_1024.onnx
imgsz: 1024
conf: 0.25
iou: 0.5
mask_thres: 0.55
mask_box_expand_ratio: 0.0
min_area_ratio: 0.0005
morph_close: true
max_det: 300

tile_size: 1024
tile_overlap_px: 256
tile_weight: 0.7
tile_stitch_mode: max

sam3_enabled: true
sam3_mode: balanced
sam3_adapter_enabled: true
sam3_lora_path: runs/sam3_finetune/lora/best
sam3_selector_enabled: true
sam3_selector_path: runs/sam3_finetune/lora_selector.json
```

说明：

- `tile_stitch_mode: max`：tile overlap 区域取最大概率，减少水体被 average 拼接切断的问题。
- `mask_thres: 0.55`：根据蓝色 flood 标注图做过单图校准，比 `0.5` 更少误报。
- `mask_box_expand_ratio: 0.0`：目前不启用检测框扩张。实验表明扩框会补部分缺块，但也会吞掉非 flood 区域。

### 2.2 基础平台配置

```text
configs/onnx_platform.yaml
```

用途：YOLO-only 或基础平台默认配置。

---

## 3. 核心预测流水线模块

```text
waterseg_platform/
```

### 3.1 主调度

```text
waterseg_platform/pipeline.py
```

职责：

- 读取图像。
- 判断是否使用 tiled inference。
- 调用 YOLO ONNX 推理。
- 对 tile 概率图进行拼接。
- 执行 cascade fallback。
- 执行 SAM3-LoRA refinement。
- 保存 mask 和 overlay。

整体流程：

```text
input image
  ↓
read config
  ↓
single/tiled YOLO ONNX inference
  ↓
YOLO mask decode
  ↓
tile probability stitching
  ↓
postprocess
  ↓
cascade fallback，可选
  ↓
SAM3-LoRA + selector，可选
  ↓
save pred mask / overlay
```

### 3.2 ONNX 推理

```text
waterseg_platform/engine.py
```

职责：

- 加载 ONNX 模型。
- 选择 ONNX Runtime provider。
- 预处理输入。
- 执行模型推理。
- 调用 YOLO mask 解码。

### 3.3 图像预处理

```text
waterseg_platform/preprocessing.py
```

职责：

- `letterbox` resize。
- BGR/RGB 处理。
- 转换为 NCHW float tensor。

### 3.4 YOLO mask 解码与后处理

```text
waterseg_platform/postprocessing.py
src/mask_utils.py
```

职责：

- 解析 YOLOv8-seg 输出。
- 执行 NMS。
- 解码 mask prototype。
- 将 mask paste 回原图或 tile。
- 阈值化。
- 去小连通域。
- morphology close。

相关参数：

```yaml
mask_thres: 0.55
mask_box_expand_ratio: 0.0
min_area_ratio: 0.0005
morph_close: true
```

其中 `mask_box_expand_ratio` 是为诊断“局部 tile mask 被检测框规整裁掉”新增的可控参数。当前根据标注图校准后设为 `0.0`。

### 3.5 Tile 切块与拼接

```text
waterseg_platform/tiling.py
```

职责：

- 生成 tile 网格。
- 管理 overlap。
- 将每个 tile 的 probability map 拼回整图。

当前关键参数：

```yaml
tile_size: 1024
tile_overlap_px: 256
tile_stitch_mode: max
```

`tile_stitch_mode` 支持：

```text
average
max
```

当前使用 `max`，用于缓解连续水体在 tile overlap 区域被平均稀释后断裂的问题。

---

## 4. SAM3-LoRA 与 Proposal Selector

### 4.1 SAM3 refinement

```text
waterseg_platform/sam3_refinement.py
waterseg_platform/sam3_worker.py
```

职责：

- 根据 YOLO mask 生成局部候选区域。
- 生成 text/box/points proposals。
- 与 SAM3 子进程交互。
- 执行 Conservative / Balanced / Open 模式下的 proposal fusion。

### 4.2 LoRA adapter

```text
waterseg_platform/sam3_lora.py
runs/sam3_finetune/lora/best/
```

职责：

- 加载 SAM3 LoRA adapter。
- 只启用轻量 Q/V LoRA 参数。

### 4.3 Proposal selector

```text
waterseg_platform/sam3_selector.py
runs/sam3_finetune/lora_selector.json
```

职责：

- 根据 proposal 特征选择是否接受 SAM3 proposal。
- 不确定时保留 YOLO。

---

## 5. 模型与权重目录

### 5.1 ONNX 模型

```text
onnx/
```

当前主模型：

```text
onnx/floodnet_binary_aug_yolov8m_1024.onnx
```

cascade fallback 模型：

```text
onnx/waterlogging_yolov8m_base_640.onnx
onnx/waterlogging_yolov8m_hard_finetune_640.onnx
```

### 5.2 SAM3-LoRA 训练结果

```text
runs/sam3_finetune/
```

关键文件：

```text
runs/sam3_finetune/lora/best/adapter.pt
runs/sam3_finetune/lora_selector.json
```

---

## 6. 输出结果目录

预测输出一般位于：

```text
runs/
```

常见输出文件：

```text
*_pred.png       # 二值预测 mask
*_overlay.jpg    # 原图 + 预测区域叠加
summary.json     # 诊断/指标信息，可选
```

近期相关输出目录：

```text
runs/user_image_test_tuned_platform/
runs/user_image_chatgpt_flood_gt_tuned_platform/
runs/user_image_chatgpt_flood_max_stitch_yolo/
runs/user_image_pakistan_yolo_tiled_max_stitch/
```

### 6.1 test_image.png 当前输出

```text
runs/user_image_test_tuned_platform/test_image_pred.png
runs/user_image_test_tuned_platform/test_image_overlay.jpg
```

### 6.2 蓝色 GT 标注图校准输出

```text
runs/user_image_chatgpt_flood_gt_calibration/
runs/user_image_chatgpt_flood_gt_tuned_platform/
```

其中：

```text
runs/user_image_chatgpt_flood_gt_tuned_platform/metrics_vs_blue_gt.json
runs/user_image_chatgpt_flood_gt_tuned_platform/gt_error_map_green_tp_red_fp_blue_fn.jpg
```

---

## 7. Tile debug 诊断目录

当前 tile debug 是临时诊断脚本生成的输出，还没有固化成正式脚本。

相关目录：

```text
runs/user_image_chatgpt_flood_tile_debug/
runs/user_image_chatgpt_flood_tile_debug_box_expand/
runs/user_image_chatgpt_flood_tile_debug_box_expand075/
```

典型内容：

```text
tile_index_map.jpg                 # tile 编号索引图
tile_overlay_contact_sheet.jpg     # 每个 tile 的预测拼图
tile_debug.html                    # HTML 浏览页
tiles/
  tile_000_*_image.jpg             # 局部原图
  tile_000_*_overlay.jpg           # 局部预测叠加图
  tile_000_*_mask.png              # 局部 mask
  tile_000_*_prob.png              # 局部概率图
```

建议后续固化为：

```text
scripts/debug_tile_predictions.py
```

推荐功能：

- 导出每个 tile 原图。
- 导出每个 tile mask。
- 导出每个 tile probability map。
- 导出每个 tile overlay。
- 导出 tile bbox 诊断图。
- 生成 contact sheet。
- 生成 HTML 页面。
- 输出 summary.json。

---

## 8. 测试文件

```text
tests/
```

与预测功能强相关的测试：

```text
tests/test_tiling.py
tests/test_postprocessing.py
tests/test_platform_config.py
tests/test_platform_cascade.py
tests/test_sam3_refinement.py
tests/test_sam3_finetune.py
```

最近验证命令：

```bash
python -m pytest tests/test_postprocessing.py tests/test_tiling.py tests/test_platform_config.py tests/test_platform_cascade.py -q
```

最近通过结果：

```text
37 passed
```

---

## 9. 重要实验结论

### 9.1 Tile 拼接问题

Pakistan 大图暴露出 overlap 区域 average 拼接会造成连续水体断裂。

修复：

```yaml
tile_stitch_mode: max
```

效果：

- overlap 区域取最大概率。
- 连续水体不容易被相邻 tile 的低概率平均掉。

### 9.2 检测框裁剪问题

局部 tile debug 中出现规则矩形缺块，原因是 YOLOv8-seg 的 mask 解码会受检测框约束：

```text
YOLO box → mask proto crop → paste 回 tile
```

曾测试：

```yaml
mask_box_expand_ratio: 0.25 / 0.4 / 0.75
```

结论：

- 扩框可以减少局部 tile 的规整缺块。
- 但在蓝色 flood GT 标注图上，扩框会带来更多误报。
- 当前最佳配置是不扩框：

```yaml
mask_box_expand_ratio: 0.0
```

### 9.3 蓝色 flood 标注图校准

使用用户提供的蓝色 flood 标注图作为单图 GT 后，扫描得到当前较优配置：

```yaml
mask_thres: 0.55
mask_box_expand_ratio: 0.0
tile_stitch_mode: max
```

对应指标：

```text
Precision: 0.757
Recall:    0.830
IoU:       0.656
Dice:      0.792
```

---

## 10. 推荐整理后的工程心智模型

```text
D:\project\water_segment
├── configs/                         # 配置文件
│   ├── onnx_platform.yaml
│   └── onnx_platform_sam3_lora_frozen.yaml
│
├── onnx/                            # ONNX 模型
│   ├── floodnet_binary_aug_yolov8m_1024.onnx
│   ├── waterlogging_yolov8m_base_640.onnx
│   └── waterlogging_yolov8m_hard_finetune_640.onnx
│
├── scripts/                         # 命令行脚本
│   ├── 11_run_onnx_platform.py
│   ├── 16_cache_sam3_proposals.py
│   ├── 17_train_sam3_selector.py
│   ├── 18_train_sam3_lora.py
│   └── 19_eval_frozen_lora_test.py
│
├── waterseg_platform/               # 平台预测核心代码
│   ├── cli.py
│   ├── config.py
│   ├── engine.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── postprocessing.py
│   ├── tiling.py
│   ├── sam3_refinement.py
│   ├── sam3_worker.py
│   ├── sam3_selector.py
│   └── visualization.py
│
├── src/                             # 通用工具
│   └── mask_utils.py
│
├── tests/                           # 单元测试
│   ├── test_tiling.py
│   ├── test_postprocessing.py
│   ├── test_platform_config.py
│   └── test_platform_cascade.py
│
├── runs/                            # 训练、评估、预测输出
│   ├── sam3_finetune/
│   ├── final_test/
│   ├── platform_deploy_lora_test60/
│   ├── user_image_test_tuned_platform/
│   └── user_image_chatgpt_flood_gt_tuned_platform/
│
└── docs/                            # 文档
    ├── sam3_lora_project_workflow.html
    └── platform_prediction_project_structure.md
```


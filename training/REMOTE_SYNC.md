# Remote 同步准备说明

更新时间：2026-08-17  
当前同步分支：`codex/water-segment-sync`  
Remote：`origin` → `https://gitee.com/git-flood_enterprise/training.git`

## 1. 当前结论

当前工作区不能直接执行 `git add .`：整理前约有 501,592 个未跟踪文件，主要是数据集、训练结果、缓存、模型和导出副本。本地 `.git` 目录约 27.76 GiB，历史中已经包含大量数据及非 LFS 权重。

本次整理采用以下边界：

- 默认同步：Python/C# 源码、配置、测试、技术文档、项目结构说明。
- 默认不同步：`data/`、`runs/`、`outputs/`、`training_set/`、缓存、构建目录、压缩包和历史模型。
- 部署模型：允许最新 DualContext ONNX 与默认 YOLO 主模型/cascade 模型进入候选提交，并强制使用 Git LFS。
- 最新 PyTorch checkpoint `runs/dual_context_water/v2_large_1024_finetune/best.pt` 保留在本机；如确实需要远程保存，应使用模型制品库或经过确认后单独 `git add -f`，不要随源码批量提交。

## 2. 本次应同步的核心内容

```text
PROJECT_STRUCTURE.md
REMOTE_SYNC.md
.gitignore
.gitattributes
README.md
requirements.txt
run_pipeline_example.sh

configs/
dual_context_water/
training/
waterseg_platform/
src/
scripts/
tests/
docs/

platform_prediction_yolo_dotnet_export/
  configs/
  docs/
  dotnet/src/
  dotnet/samples/*/源码与项目文件

onnx/floodnet_binary_aug_yolov8m_1024.onnx            # YOLO 主模型，Git LFS
onnx/waterlogging_yolov8m_base_640.onnx               # YOLO cascade 基线，Git LFS
onnx/waterlogging_yolov8m_hard_finetune_640.onnx      # YOLO cascade hard-sample，Git LFS
onnx/dual_context_water_v2_large_1024_candidate.onnx  # 最新非 YOLO 模型，Git LFS
```

其中最新非 YOLO 模型为：

```text
runs/dual_context_water/v2_large_1024_finetune/best.pt       # 本地训练 checkpoint
onnx/dual_context_water_v2_large_1024_candidate.onnx         # 建议同步的部署模型
configs/dual_context_water_v2_large_1024_finetune.yaml       # 训练配置
dual_context_water/                                          # 模型和推理源码
```

## 3. 当前不能由 `.gitignore` 自动解决的问题

`.gitignore` 只影响未跟踪文件。当前 Git 历史已经跟踪：

- `data/`：23,572 个文件；
- `training_set/`：11,168 个文件；
- `runs/`：1,333 个文件；
- 多个 `src/__pycache__/*.pyc`；
- `training_set.zip`（约 541 MiB）；
- 多个 50～90 MiB 的历史 `.pt` 权重。

因此新增忽略规则后，这些文件的既有修改仍会出现在 `git status` 中。是否把它们从远端当前分支移除属于仓库策略变更，本次没有自动执行。

如果团队确认远端只保留源码，可在独立清理提交中执行以下命令。命令只从 Git 索引移除，工作区文件仍保留，但提交后远端对应分支会显示这些文件被删除：

```powershell
git rm -r --cached --ignore-unmatch data runs training_set
git rm --cached --ignore-unmatch training_set.zip
git rm --cached --ignore-unmatch yolo26n.pt yolov8n-seg.pt yolov8s-seg.pt yolov8m-seg.pt
git rm -r --cached --ignore-unmatch ':(glob)**/__pycache__/**'
```

执行前必须和远端协作者确认。即使这样做，旧提交中的大文件仍保留在 Git 历史里，不会让历史仓库立即变小。

## 4. 推荐的安全提交步骤

### 4.1 先运行只读审计

```powershell
powershell -ExecutionPolicy Bypass -File scripts/audit_remote_sync.ps1
```

### 4.2 创建同步分支

当前远端没有 `google-earth-downloader` 同名分支。建议使用 Codex 分支前缀：

```powershell
git switch -c codex/water-segment-sync
```

### 4.3 分组暂存，避免 `git add .`

```powershell
git add .gitignore .gitattributes PROJECT_STRUCTURE.md REMOTE_SYNC.md
git add README.md requirements.txt run_pipeline_example.sh
git add configs/*.yaml
git add dual_context_water/*.py
git add training/*.py training/sam3_lora/*.py
git add waterseg_platform/*.py waterseg_platform/*.md
git add src/*.py
git add scripts/*.py scripts/*.ps1 scripts/*.md
git add tests/*.py
git add docs
git add platform_prediction_yolo_dotnet_export/configs
git add platform_prediction_yolo_dotnet_export/docs
git add platform_prediction_yolo_dotnet_export/dotnet/src
git add platform_prediction_yolo_dotnet_export/dotnet/samples
git add platform_prediction_yolo_dotnet_export/onnx/floodnet_binary_aug_yolov8m_1024.onnx
git add onnx/floodnet_binary_aug_yolov8m_1024.onnx
git add onnx/waterlogging_yolov8m_base_640.onnx
git add onnx/waterlogging_yolov8m_hard_finetune_640.onnx
git add onnx/dual_context_water_v2_large_1024_candidate.onnx
```

然后检查：

```powershell
git status --short
git diff --cached --stat
git diff --cached --name-only
git lfs ls-files
```

最新 ONNX 必须出现在 `git lfs ls-files` 中。若没有出现，不要提交。

### 4.4 提交和推送

确认暂存清单后再执行：

```powershell
git commit -m "Organize water segmentation source and add dual-context model"
git push -u origin codex/water-segment-sync
```

本说明没有自动执行建分支、暂存、提交或推送。

## 5. Remote 与历史体量注意事项

远端当前可见分支包括：

```text
codex/dual-context-dotnet
codex/eval-results-no-full-gt
dotnet-yolo-interface
onnx-base-model
```

若目标是在现有 remote 上新增增量分支，可按上面的分组暂存方式操作。若目标是建立一个全新的轻量源码仓库，不建议从当前 27.76 GiB 的历史直接首次推送；应在新的空目录中初始化仓库，只复制本说明第 2 节列出的源码和 LFS 模型，避免携带旧数据历史。

## 6. 凭据检查

源码扫描未发现硬编码 token、密码或私钥。CDSE 下载脚本通过环境变量读取 `CDSE_CLIENT_ID` 和 `CDSE_CLIENT_SECRET`，符合远程同步要求。提交前仍应检查本机新增的 `.env`、临时配置和命令日志；这些文件不应进入仓库。

# Update Platform Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the latest binary FloodNet YOLOv8m segmentation checkpoint to a versioned ONNX file and make it the platform default without deleting the previous model.

**Architecture:** Keep the existing ONNX Runtime inference pipeline unchanged. Export a fixed-shape 1024x1024 ONNX model, then update both the YAML configuration and `PlatformConfig` defaults to use the new path and binary class count (`nc=1`).

**Tech Stack:** Python 3.10, Ultralytics YOLOv8, ONNX, ONNX Runtime, pytest.

---

### Task 1: Lock the new platform defaults with a test

**Files:**
- Modify: `tests/test_platform_config.py`
- Modify: `waterseg_platform/config.py`
- Modify: `configs/onnx_platform.yaml`

- [ ] **Step 1: Write a failing test**

Add assertions that both `PlatformConfig()` and `load_config("configs/onnx_platform.yaml")` use:

```python
assert cfg.model_path == "onnx/floodnet_binary_aug_yolov8m_1024.onnx"
assert cfg.imgsz == 1024
assert cfg.tile_size == 1024
assert cfg.nc == 1
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
pytest tests/test_platform_config.py -q
```

Expected: failure because the current defaults point to `onnx/best.onnx` with `nc=3`.

- [ ] **Step 3: Update the minimal configuration**

Change the model path and class count in `waterseg_platform/config.py` and `configs/onnx_platform.yaml`. Update nearby comments to describe the single `waterlogging` class.

- [ ] **Step 4: Run the test and verify it passes**

Run:

```powershell
pytest tests/test_platform_config.py -q
```

Expected: all configuration tests pass.

### Task 2: Export and inspect the versioned ONNX model

**Files:**
- Create: `onnx/floodnet_binary_aug_yolov8m_1024.onnx`

- [ ] **Step 1: Export the best checkpoint**

Run Ultralytics export with fixed `imgsz=1024`, ONNX format, opset 13, batch 1, and no dynamic axes.

- [ ] **Step 2: Move the exported artifact to the versioned path**

Keep `onnx/best.onnx` untouched and place the new artifact at:

```text
onnx/floodnet_binary_aug_yolov8m_1024.onnx
```

- [ ] **Step 3: Inspect model inputs and outputs**

Create an ONNX Runtime session and assert:

```python
input_shape == [1, 3, 1024, 1024]
output0_channels == 4 + 1 + 32
```

Expected: fixed input shape and 37 detection channels for one class plus 32 mask coefficients.

### Task 3: Verify platform inference end to end

**Files:**
- Test: `tests/test_pipeline_parity.py`
- Read: `data/floodnet_binary_aug_1024/processed/images/test`

- [ ] **Step 1: Run focused platform tests**

Run:

```powershell
pytest tests/test_platform_config.py tests/test_pipeline_parity.py -q
```

Expected: tests pass with model-dependent tests enabled when dependencies are present.

- [ ] **Step 2: Run one real test image through the platform**

Load `configs/onnx_platform.yaml`, construct `SegmentationService`, and infer one test image without tiling.

Expected: inference completes, output mask matches image dimensions, and the binary mask contains only `0` and `1`.

- [ ] **Step 3: Confirm the old model remains**

Verify both files exist:

```text
onnx/best.onnx
onnx/floodnet_binary_aug_yolov8m_1024.onnx
```


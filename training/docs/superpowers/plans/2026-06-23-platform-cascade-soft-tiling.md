# Platform Cascade And Soft Tiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add conservative multi-model fallback and replace tiled hard gating with soft probability fusion.

**Architecture:** `SegmentationService` owns one primary engine and lazily
creates two fallback engines. Pure helper functions decide fallback acceptance
and combine coarse/tiled predictions so behavior can be tested without loading
models.

**Tech Stack:** Python, NumPy, OpenCV, ONNX Runtime, pytest, Ultralytics export.

---

### Task 1: Configuration And Cascade Decision

**Files:**
- Modify: `waterseg_platform/config.py`
- Modify: `configs/onnx_platform.yaml`
- Modify: `tests/test_platform_config.py`

- [ ] Add failing tests for fallback defaults and conservative intersection.
- [ ] Add fallback model paths, fallback input size, trigger ratio, minimum
  accepted ratio, enable switch, and tile weight.
- [ ] Run focused tests until green.

### Task 2: Export Fallback Models

**Files:**
- Create: `onnx/waterlogging_yolov8m_base_640.onnx`
- Create: `onnx/waterlogging_yolov8m_hard_finetune_640.onnx`

- [ ] Export both local checkpoints as fixed 640 ONNX models.
- [ ] Verify input `[1,3,640,640]` and output channel count `37`.

### Task 3: Cascade Inference

**Files:**
- Modify: `waterseg_platform/pipeline.py`
- Create: `tests/test_platform_cascade.py`

- [ ] Add failing tests for no trigger, accepted consensus, rejected consensus,
  and disabled cascade.
- [ ] Implement lazy fallback engines and intersection-based replacement.
- [ ] Record cascade diagnostics in result info.
- [ ] Run focused tests until green.

### Task 4: Soft Tiled Fusion

**Files:**
- Modify: `waterseg_platform/pipeline.py`
- Modify: `tests/test_tiling.py`

- [ ] Add a failing pure test proving a strong tile probability survives when
  the coarse mask is empty.
- [ ] Replace hard gating and early exit with weighted max fusion.
- [ ] Run tiled tests until green.

### Task 5: End-To-End Verification

**Files:**
- Read: `C:/Users/17473/Desktop/test_image.png`
- Write: `runs/platform_test/test_image_cascade_soft_tiling/`

- [ ] Run all platform tests.
- [ ] Run single and tiled platform inference on the supplied image.
- [ ] Save overlays, masks, and metrics.
- [ ] Verify old and new ONNX files remain present.

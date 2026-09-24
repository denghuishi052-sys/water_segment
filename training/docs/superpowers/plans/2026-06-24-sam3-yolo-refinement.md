# SAM 3 Guided YOLO Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional, runtime-switchable conservative and open SAM 3 refinement after the platform's existing YOLO/cascade result.

**Architecture:** Keep candidate extraction and mask fusion as pure NumPy/OpenCV functions in a focused module. Run the installed official `sam3` package in a persistent `torch_env` child process to isolate its Windows native DLLs from ONNX Runtime/OpenCV, and call it once after direct or tiled YOLO inference. Configuration, CLI, and Gradio only select behavior; all SAM failures return the unchanged YOLO mask with diagnostics.

**Tech Stack:** Python 3.10, NumPy, OpenCV, PyTorch 2.5, official `sam3` package, ONNX Runtime, pytest, Gradio.

---

## File Structure

- Create `waterseg_platform/sam3_refinement.py`: candidates, fusion policy, lazy SAM 3 adapter, diagnostics.
- Create `waterseg_platform/sam3_worker.py`: process-isolated PyTorch/SAM 3 model and prompt execution.
- Modify `waterseg_platform/config.py`: SAM 3 configuration fields.
- Modify `waterseg_platform/pipeline.py`: shared post-YOLO refinement dispatch.
- Modify `waterseg_platform/cli.py`: command-line switches and per-call forwarding.
- Modify `waterseg_platform/ui_gradio.py`: refinement and conservative-mode controls.
- Modify `configs/onnx_platform.yaml`: disabled-by-default production settings.
- Create `configs/onnx_platform_sam3_trial.yaml`: enabled trial settings.
- Create `tests/test_sam3_refinement.py`: pure policy and adapter failure tests.
- Modify `tests/test_platform_config.py`: config and pipeline dispatch tests.
- Modify `waterseg_platform/README.md`: usage and fallback behavior.

### Task 1: Configuration Contract

**Files:**
- Modify: `tests/test_platform_config.py`
- Modify: `waterseg_platform/config.py`
- Modify: `configs/onnx_platform.yaml`
- Create: `configs/onnx_platform_sam3_trial.yaml`

- [ ] **Step 1: Write failing configuration tests**

Add assertions for `sam3_enabled=False`, `sam3_conservative=True`, the local
checkpoint path, CUDA device, candidate/fusion thresholds, and text prompts.
Also load the trial YAML and assert that only `sam3_enabled` changes to true.

- [ ] **Step 2: Run the focused test and verify RED**

Run:
`C:/Users/17473/miniforge3/envs/torch_env/python.exe -m pytest tests/test_platform_config.py -q`

Expected: failure because `PlatformConfig` has no SAM 3 fields.

- [ ] **Step 3: Add the dataclass and YAML fields**

Use these exact defaults:

```python
sam3_enabled: bool = False
sam3_conservative: bool = True
sam3_checkpoint: str = "D:/BaiduNetdiskDownload/课程- 权重(1)/sam3.pt"
sam3_device: str = "cuda"
sam3_confidence: float = 0.5
sam3_box_margin_ratio: float = 0.10
sam3_min_component_area_ratio: float = 0.0005
sam3_max_candidates: int = 8
sam3_min_yolo_overlap: float = 0.20
sam3_max_area_growth: float = 2.0
sam3_max_image_area_ratio: float = 0.65
sam3_text_prompts: List[str] = field(
    default_factory=lambda: ["flooded road", "standing flood water"]
)
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the same pytest command. Expected: all config tests pass.

### Task 2: Candidate Extraction and Fusion Policy

**Files:**
- Create: `tests/test_sam3_refinement.py`
- Create: `waterseg_platform/sam3_refinement.py`

- [ ] **Step 1: Write failing candidate tests**

Cover connected-component sorting, minimum-area removal, box-margin clipping,
and maximum-candidate limiting. Candidate objects contain `component_mask`,
`box_xyxy`, and normalized `box_cxcywh`.

- [ ] **Step 2: Verify candidate tests fail**

Run:
`C:/Users/17473/miniforge3/envs/torch_env/python.exe -m pytest tests/test_sam3_refinement.py -q`

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement candidate extraction**

Use `cv2.connectedComponentsWithStats`, stable descending area ordering, and
image-bound clipping. Return binary `uint8` masks.

- [ ] **Step 4: Write failing fusion tests**

Cover:

```python
conservative valid mask -> accepted inside expanded box
conservative low overlap -> original component retained
conservative excessive growth -> original component retained
open mask -> allowed outside candidate box
open mask over max_image_area_ratio -> rejected
wrong-shaped or non-finite mask -> rejected
```

- [ ] **Step 5: Implement and verify pure fusion**

Implement `fuse_sam3_masks(...) -> (mask, diagnostics)` without importing
PyTorch or SAM 3. Run the focused tests and expect all to pass.

### Task 3: Lazy Official SAM 3 Adapter

**Files:**
- Modify: `tests/test_sam3_refinement.py`
- Modify: `waterseg_platform/sam3_refinement.py`

- [ ] **Step 1: Write failing adapter tests**

Inject a fake backend factory and processor. Assert:

- disabled refinement never constructs the backend;
- the backend is loaded once and reused;
- BGR input is converted to RGB/PIL;
- image encoding runs once per request;
- each candidate receives normalized geometry;
- open mode runs both configured global text prompts;
- exceptions return the original YOLO mask and `fallback=True`.

- [ ] **Step 2: Verify tests fail for missing adapter**

Run the focused test file and confirm failures mention `Sam3Refiner`.

- [ ] **Step 3: Implement the lazy adapter**

The default backend factory starts one worker with `sys.executable` and an
authenticated local `multiprocessing.connection` channel. The worker imports:

```python
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
```

Build in the worker with `load_from_HF=False`, `compile=False`,
`enable_inst_interactivity=False`, and the configured checkpoint/device.
Use one `processor.set_image()` call per image under CUDA BF16 autocast. For
each candidate, reset prompt state and add its normalized positive box. Open
mode additionally resets prompts and calls `set_text_prompt()` for each global
concept. Convert tensors to NumPy in the worker before returning them for
fusion, and release request-local state in `finally`.

- [ ] **Step 4: Add CUDA failure cleanup**

Catch `torch.cuda.OutOfMemoryError` and general exceptions, clear CUDA cache
when available, and return YOLO with a concise diagnostic error. Do not retry.

- [ ] **Step 5: Run adapter tests and verify GREEN**

Expected: all pure and fake-adapter tests pass without loading the 3.45 GB
checkpoint.

### Task 4: Pipeline Integration

**Files:**
- Modify: `tests/test_platform_config.py`
- Modify: `waterseg_platform/pipeline.py`

- [ ] **Step 1: Write failing direct/tiled dispatch tests**

Construct `SegmentationService` via `__new__`, inject fake YOLO and refiner
objects, and assert:

- direct inference refines after cascade;
- tiled inference refines after tiling and cascade;
- disabled per-call mode skips the refiner;
- `sam3_conservative` per-call override reaches the refiner;
- returned `pred_area`, `pred_area_ratio`, and `info["sam3"]` match the final
  mask.

- [ ] **Step 2: Verify pipeline tests fail**

Run `tests/test_platform_config.py` and confirm missing SAM dispatch.

- [ ] **Step 3: Implement one shared `_apply_sam3()` method**

Initialize `self._sam3_refiner = None`. Create it lazily only when enabled.
Call `_apply_sam3()` after `_apply_cascade()` in both `segment_array()` and
`segment_array_tiled()`. Add optional `sam3_enabled` and `sam3_conservative`
arguments through `segment_file()` and `segment_directory()`.

- [ ] **Step 4: Run pipeline and existing cascade/tiling tests**

Run:

`C:/Users/17473/miniforge3/envs/torch_env/python.exe -m pytest tests/test_platform_config.py tests/test_platform_cascade.py tests/test_tiling.py -q`

Expected: all pass.

### Task 5: CLI and Gradio Controls

**Files:**
- Modify: `tests/test_sam3_refinement.py`
- Modify: `waterseg_platform/cli.py`
- Modify: `waterseg_platform/ui_gradio.py`

- [ ] **Step 1: Write failing CLI parser/config tests**

Assert `--sam3` enables refinement, `--no_sam3` disables it, and
`--sam3_open` switches conservative mode off. The config value remains the
default when no override flag is supplied.

- [ ] **Step 2: Implement CLI flags and forwarding**

Add mutually exclusive enable/disable flags plus `--sam3_open`. Forward values
through single-image, directory, and UI configuration.

- [ ] **Step 3: Add Gradio controls**

Add a `SAM 3` accordion to both tabs with:

```python
gr.Checkbox(label="SAM 3 refinement", value=cfg.sam3_enabled)
gr.Checkbox(label="Conservative mode", value=cfg.sam3_conservative)
```

Forward both values without mutating the singleton config. Append SAM 3
diagnostics to the result textbox.

- [ ] **Step 4: Run focused tests**

Run config, refinement, and platform tests. Expected: all pass.

### Task 6: Real Checkpoint Smoke Test and Comparison

**Files:**
- Create: `scripts/13_compare_sam3_refinement.py`
- Modify: `waterseg_platform/README.md`

- [ ] **Step 1: Write the comparison script**

Load the trial config once, run YOLO-only, conservative, and open modes on
`C:/Users/17473/Desktop/test_image.png`, and save masks, overlays, timing, area
ratio, and JSON diagnostics under `runs/platform_compare/sam3_refinement/`.

- [ ] **Step 2: Run unit regression before the large model**

Run:

`C:/Users/17473/miniforge3/envs/torch_env/python.exe -m pytest tests/test_platform_config.py tests/test_platform_cascade.py tests/test_preprocessing.py tests/test_postprocessing.py tests/test_tiling.py tests/test_sam3_refinement.py -q`

Expected: zero failures.

- [ ] **Step 3: Run the real checkpoint smoke test**

Run:

`C:/Users/17473/miniforge3/envs/torch_env/python.exe scripts/13_compare_sam3_refinement.py`

Expected: local checkpoint load without network access, three output variants,
or a recorded fallback diagnostic if the current SAM 3/PyTorch combination is
incompatible.

- [ ] **Step 4: Resolve only evidence-backed compatibility failures**

For each real error, add a failing regression test where practical, make the
smallest adapter change, and rerun the smoke test. Do not upgrade PyTorch or
replace the existing environment unless the user explicitly approves it.

- [ ] **Step 5: Document operation**

Document config fields, CLI examples, UI controls, first-load cost, 12 GB VRAM
expectations, open-mode risk, and guaranteed YOLO fallback.

### Task 7: Final Verification

**Files:**
- All changed files.

- [ ] **Step 1: Run syntax and focused test verification**

Run:

```powershell
C:/Users/17473/miniforge3/envs/torch_env/python.exe -m compileall waterseg_platform scripts/13_compare_sam3_refinement.py
C:/Users/17473/miniforge3/envs/torch_env/python.exe -m pytest tests/test_platform_config.py tests/test_platform_cascade.py tests/test_preprocessing.py tests/test_postprocessing.py tests/test_tiling.py tests/test_sam3_refinement.py -q
```

- [ ] **Step 2: Inspect comparison artifacts**

Verify all expected files exist, masks match source dimensions, conservative
pixels stay inside expanded candidate boxes, and diagnostics report the actual
mode and fallback state.

- [ ] **Step 3: Review the diff**

Run `git diff --check` and inspect only the files listed in this plan. Preserve
all unrelated user changes.

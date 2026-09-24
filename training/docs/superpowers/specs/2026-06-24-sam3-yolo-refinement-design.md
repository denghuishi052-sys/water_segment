# SAM 3 Guided YOLO Waterlogging Refinement Design

## Goal

Add optional SAM 3 refinement to the existing binary waterlogging platform.
YOLOv8-seg remains the primary detector and supplies spatial candidates. SAM 3
refines those candidates with geometry and text prompts. The original YOLO
result remains available as a reliable fallback.

The feature supports two runtime-selectable modes:

- Conservative mode, enabled by default, prioritizes precision.
- Open mode allows SAM 3 to expand or add nearby water regions and prioritizes
  recall.

## Constraints

- Use the existing `torch_env` environment.
- Use the local checkpoint:
  `D:/BaiduNetdiskDownload/课程- 权重(1)/sam3.pt`.
- Use the installed official `sam3` package and its image-model API. Do not
  require Transformers.
- Preserve the current ONNX YOLO model, cascade, direct inference, and tiled
  inference behavior when SAM 3 is disabled.
- Target an NVIDIA RTX 3060 with 12 GB VRAM.
- A SAM 3 failure must never prevent the platform from returning a YOLO mask.

## Architecture

Introduce a small SAM 3 boundary behind an interface instead of importing SAM 3
throughout the platform:

1. `Sam3Refiner` lazily loads the checkpoint on first use.
2. The existing `SegmentationService` produces the final YOLO/cascade mask as
   it does today.
3. Connected components from that mask become candidate prompts.
4. Each candidate is converted to an expanded bounding box. The original mask
   component is also retained as a fusion constraint.
5. SAM 3 receives the source image, geometry prompt, and configured English
   water concept prompts.
6. A pure fusion function validates and combines SAM 3 masks with YOLO masks
   according to the selected mode.
7. Invalid, empty, implausible, timed-out, or out-of-memory SAM 3 output falls
   back to the unchanged YOLO result.

SAM 3 runs in a persistent child process launched with the current
`torch_env` Python executable. Process isolation is required because the
installed ONNX Runtime/OpenCV and PyTorch builds load incompatible Windows
native DLLs when combined in one process. The child imports PyTorch before any
other large native runtime. Its model and GPU allocation remain lazy, and the
model stays resident after the first successful load to avoid paying the
roughly 3.45 GB checkpoint load cost for every image.

## Configuration

Extend `PlatformConfig` and `configs/onnx_platform.yaml` with:

```yaml
sam3_enabled: false
sam3_conservative: true
sam3_checkpoint: "D:/BaiduNetdiskDownload/课程- 权重(1)/sam3.pt"
sam3_device: cuda
sam3_box_margin_ratio: 0.10
sam3_min_component_area_ratio: 0.0005
sam3_max_candidates: 8
sam3_min_yolo_overlap: 0.20
sam3_max_area_growth: 2.00
sam3_text_prompts:
  - flooded road
  - standing flood water
```

`sam3_enabled` defaults to false initially so installing this change does not
silently change production predictions. The Gradio interface exposes:

- `SAM 3 refinement`
- `Conservative mode`

The CLI exposes equivalent enable/disable and mode flags. Per-call values
override configuration without mutating the shared service configuration.

## Candidate Generation

Candidates are generated only after the current primary, tiling, and cascade
pipeline has produced its selected binary mask.

1. Find connected components in the selected YOLO mask.
2. Remove components below `sam3_min_component_area_ratio`.
3. Sort remaining components by area and keep at most
   `sam3_max_candidates`.
4. Compute an axis-aligned bounding box for each component.
5. Expand each box by `sam3_box_margin_ratio`, clipped to image bounds.

If no candidate remains, conservative mode immediately returns the YOLO mask.
Open mode always runs one global text-prompt pass, including when the YOLO mask
is empty. This avoids invoking a large model for empty conservative
predictions while giving open mode a defined recall-oriented behavior.

## SAM 3 Inference

Build the official image model with:

```python
build_sam3_image_model(
    checkpoint_path=checkpoint_path,
    load_from_HF=False,
    device="cuda",
    eval_mode=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
)
```

Inference runs under `torch.inference_mode()` and CUDA autocast using
float16/bfloat16 where supported. The first implementation does not use
`torch.compile`, Flash Attention 3, video tracking, or model fine-tuning.

The refiner processes one source image at a time. It computes image features
once and reuses them for every candidate and text prompt through the installed
SAM 3 image processor API. Candidate prompts use geometry only, which avoids a
redundant text-grounding pass and reduces first-request memory pressure. Open
mode additionally evaluates the configured global text prompts. Candidate
prompts are evaluated sequentially to limit peak VRAM on the 12 GB GPU.

## Fusion Modes

### Conservative Mode

For each candidate:

1. Clip the SAM mask to the expanded YOLO candidate box.
2. Require overlap with the corresponding YOLO component:
   `intersection / YOLO component area >= sam3_min_yolo_overlap`.
3. Reject a SAM mask whose area exceeds
   `YOLO component area * sam3_max_area_growth`.
4. If valid, use the SAM mask as the refined component.
5. If invalid or empty, keep the original YOLO component.

The final mask is the union of all accepted refined components and all
fallback YOLO components. SAM 3 cannot create a remote component.

### Open Mode

SAM masks are selected using the same candidate prompts and text concepts, but
they are not clipped to the expanded candidate boxes. Open mode also runs one
global text-prompt pass, and its accepted masks are merged with candidate-based
masks. Basic safeguards still reject non-finite output, wrong shapes, empty
output when a candidate was expected, and masks covering an implausibly large
fraction of the image.

Open mode is explicitly experimental and must be labeled as such in the UI.

## Tiled Images

SAM 3 refinement occurs once on the original full-resolution image after the
existing coarse/tiled fusion and cascade selection. It does not run separately
on every tile. This preserves global context, avoids seams, and prevents the
SAM 3 cost from multiplying by tile count.

YOLO tiling remains independently switchable. Therefore the supported paths
are:

- direct YOLO
- tiled YOLO
- direct YOLO plus SAM 3 refinement
- tiled YOLO plus SAM 3 refinement

## Failure Handling

Catch and record:

- checkpoint not found
- unsupported SAM 3 package/API
- CUDA unavailable
- checkpoint load failure
- CUDA out of memory
- inference exception
- malformed SAM output
- candidate rejection by fusion safeguards

On any model-level failure, return the original YOLO mask and add diagnostic
information under `info["sam3"]`. CUDA out-of-memory handling clears cached
memory after releasing temporary tensors. The platform must not retry
automatically in the same request.

Diagnostics include enabled state, selected mode, load state, candidate count,
accepted count, rejected count, elapsed time, fallback status, and a concise
error message when applicable.

## User Interface

Add two controls to both single-image and directory workflows:

- Checkbox: `SAM 3 refinement`
- Checkbox: `Conservative mode`, default checked

The result summary reports whether SAM 3 ran, how many candidates it accepted,
and whether fallback occurred. Existing result panels and saved mask/overlay
formats remain unchanged.

## Testing

Unit tests cover:

- configuration defaults and YAML loading
- component-to-box candidate extraction and clipping
- conservative acceptance, rejection, and YOLO fallback
- open-mode expansion behavior
- disabled-mode behavior
- malformed output and simulated CUDA failure
- pipeline dispatch after direct and tiled YOLO inference
- lazy loading so SAM 3 is not imported or allocated when disabled

Integration verification uses:

- `C:/Users/17473/Desktop/test_image.png`
- the existing 60-image FloodNet test split

For each, save YOLO-only, conservative SAM 3, and open SAM 3 masks and overlays.
Report pixel area ratio, runtime, and test-set precision, recall, IoU, and F1
where labels are available. Production configuration is switched on only if
conservative mode does not introduce a material precision regression.

## Acceptance Criteria

- Existing platform tests continue to pass with SAM 3 disabled.
- Both modes are selectable from config, CLI, and Gradio.
- Missing or failing SAM 3 always returns the original YOLO mask.
- Conservative output never contains pixels outside the union of expanded
  candidate boxes.
- Direct and tiled YOLO paths can both feed the same refiner.
- The supplied checkpoint loads locally without a Hugging Face download.
- Comparison artifacts and metrics are produced before changing the production
  default.

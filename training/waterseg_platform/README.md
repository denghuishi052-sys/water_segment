# waterseg_platform — ONNX Water-Segmentation Platform

A standalone, **pure-Python** (no PyTorch, no Ultralytics at runtime) inference
platform for the binary YOLOv8-seg water model at
`onnx/floodnet_binary_aug_yolov8m_1024.onnx`. The previous
`onnx/best.onnx` model is retained for historical parity checks. It is
designed so the entire pipeline can be ported 1:1 to .NET (WPF desktop app
on .NET 8 LTS) by following the translation table at the bottom of this
document.

## Why a separate platform

The training/eval pipeline in `scripts/` (`ultralytics.YOLO(...).predict()`)
hides all of the YOLOv8-seg math (letterbox, sigmoid, NMS, mask decode)
inside the `ultralytics` package. We need a re-implementation in NumPy +
OpenCV so the inference path has no `ultralytics` dependency — and so a
C# engineer can translate each module without learning PyTorch.

The package is named **`waterseg_platform`** (not `platform`) because the
shorter name shadows the stdlib `platform` module when running from the
project root.

## Module map

```
waterseg_platform/
├── __init__.py          # public surface: PlatformConfig, OnnxSegmenter, SegmentationService
├── config.py            # PlatformConfig dataclass + YAML loader
├── engine.py            # OnnxSegmenter: ORT session, provider auto-fallback
├── preprocessing.py     # letterbox + NCHW float conversion
├── postprocessing.py    # sigmoid, NMS, mask decode, paste-and-OR (shared helper)
├── tiling.py            # compute_tiles + stitch_probability_maps (tiled inference)
├── pipeline.py          # SegmentationService: end-to-end Image -> Mask (single + tiled)
├── image_io.py          # thin re-exports of src/dataset_utils
├── visualization.py     # re-exports of src/visualization
├── metrics.py           # re-exports of src/metrics
├── cli.py               # argparse CLI (single / directory / ui)
├── ui_gradio.py         # Gradio Blocks web UI
├── README.md            # this file
└── PARITY.md            # parity report vs Ultralytics baseline
```

`image_io.py`, `visualization.py`, `metrics.py` are *thin re-exports* of
existing `src/` code — they exist so the platform has a single import path
and so the .NET port can ignore `src/`.

## Public API

```python
from waterseg_platform import PlatformConfig, OnnxSegmenter, SegmentationService

cfg = PlatformConfig(
    model_path="onnx/floodnet_binary_aug_yolov8m_1024.onnx",
    imgsz=1024,
    nc=1,
)
svc = SegmentationService(cfg)

# Single-tile — letterboxes the whole image to 1024×1024 in one call.
mask, info = svc.segment_array(image_bgr)

# Tiled — splits large/rectangular images into 1024×1024 patches and
# stitches per-tile probability maps. Always tile; for a 1-tile image
# the result is mathematically equivalent to segment_array().
mask, info = svc.segment_array_tiled(image_bgr, tile_size=1024, tile_overlap_px=256)

# File / directory dispatchers accept use_tiling=... to pick the path.
result = svc.segment_file("path/to.jpg", use_tiling=True, ...)
df = svc.segment_directory("path/to/dir", use_tiling=True, ...)
```

The default configuration also enables a conservative road-scene cascade.
When the primary model predicts less than 0.5% foreground on a landscape
image (aspect ratio at least 1.15), two local 640px road-flood models run and
their pixel intersection is accepted only if it covers at least 0.5% of the
image. This recovers cross-domain urban flooding while avoiding the fallback
on square aerial imagery.

### Optional SAM 3 refinement

SAM 3 refinement is disabled in the production config and enabled in
`configs/onnx_platform_sam3_trial.yaml`. YOLO remains the detector. Each
connected component produces an expanded box, distance-transform foreground
points, and exterior-ring background points.

- The PCS branch produces `text + box` proposals for the configured prompt
  ensemble.
- The instance-interactive branch independently produces
  `box + positive/negative points` proposals.
- Conservative mode can replace only the corresponding YOLO component.
- Balanced mode may accept high-scoring nearby global proposals.
- Open mode may recover remote components and remains experimental.
- Fusion records score, overlap, YOLO coverage, area growth, distance, and a
  rejection reason for every proposal.
- If interactive prompting fails, the request falls back to text+box. If all
  SAM 3 branches fail, the unchanged YOLO mask is returned.

On Windows, ONNX Runtime/OpenCV and PyTorch load incompatible native DLL sets
when combined in one process. The platform therefore launches one persistent
SAM 3 worker using the same `torch_env` Python executable. The 3.45 GB
checkpoint is loaded once on first use and reused for later requests.

The 60-image FloodNet test split favored YOLO-only as the production default:

| mode | precision | recall | F1 | IoU |
|---|---:|---:|---:|---:|
| YOLO-only | 0.8124 | 0.7849 | 0.7984 | 0.6645 |
| SAM 3 conservative | 0.7944 | 0.7708 | 0.7824 | 0.6426 |
| SAM 3 open | 0.6920 | 0.8558 | 0.7652 | 0.6197 |

Open mode improves recall while increasing false positives. Conservative mode
produced cleaner boundaries on the supplied urban flood image, but did not
improve aggregate test-set metrics.

Tiled inference uses soft fusion rather than a hard coarse-mask gate:

```text
combined = max(coarse_binary, tile_weight * tiled_probability)
```

The default `tile_weight` is `0.7`, so strong local detections may recover
regions missed by the whole-image pass.

## End-to-end pipeline

```
image_bgr (H, W, 3)
  │
  ├──[use_tiling=False]──▶  segment_array() ──▶ engine.predict_mask()
  │                                                  │
  │                                                  ▼
  │                                       decode_segmentation()
  │                                                  │ uses helper
  │                                                  ▼
  │                                       _place_masks_on_canvas()  [shared, returns float]
  │                                                  │
  │                                                  ▼
  │                                       threshold + OR-merge + postprocess_mask
  │                                                  │
  │                                                  ▼
  │                                       (H, W) uint8 mask
  │
  └──[use_tiling=True]───▶  segment_array_tiled()
                                │
                                ▼
                       compute_tiles(H, W, tile_size, tile_overlap_px)
                                │
                                ▼
                       for each tile:
                         crop (tile_h, tile_w) ─▶ letterbox(704) ─▶ to_nchw ─▶ engine.run()
                                │
                                ▼
                       decode_segmentation_probs()
                                │
                                ▼
                       _place_masks_on_canvas()  [shared, returns float, no threshold]
                                │
                                ▼
                       accumulate prob_full[y0:y0+tile_h, x0:x0+tile_w] += prob_tile
                       accumulate count_full[y0:y0+tile_h, x0:x0+tile_w] += 1
                                │
                                ▼
                       stitch = prob_full / count_full   [uniform, edge-safe]
                                │
                                ▼
                       (threshold at mask_thres, then postprocess_mask)
                                │
                                ▼
                       (H, W) uint8 mask
```

The key insight: `_place_masks_on_canvas` is the **only** piece of geometry
both paths share, and it returns a **float probability map** so thresholding
and postprocessing move *above* the stitch in the tiled path.

### Tile-seam caveat

The tiled path's combine rule is "average sigmoid probabilities across
overlapping tiles, then threshold" — this is the standard SAHI / YOLOv5-tile
convention and produces smooth boundaries in the interior. At tile seams
with `tile_overlap_px = 0` you may see faint discontinuities if a detection
sits right on the seam. Increase `tile_overlap_px` (e.g. 100–200) to smooth
these at the cost of more inference time. For most production use cases
the default `tile_overlap_px = 0` is fine.

### Tile size constraint

`tile_size` defaults to `config.imgsz` (704) because the model has a fixed
input shape. To use a different `tile_size` the user must re-export the
ONNX with the matching `imgsz`. The service raises a clear `ValueError`
if `tile_size != config.imgsz`.

## YOLOv8-seg decoder algorithm

The decoder in `postprocessing.py` is the part that has to be reproduced
exactly in C#. The contract:

```text
Inputs:  output0 (1, 4+nc+nm, A) ; output1 (1, nm, mh, mw)
         nc=1, nm=32, A=10164, mh=mw=176, imgsz=704

1. Split:
     boxes  = output0[0, 0:4, :]                # (4, A)   cx,cy,w,h
     scores = output0[0, 4:4+nc, :]             # (nc, A)  ALREADY SIGMOID'D
     coeffs = output0[0, 4+nc:4+nc+nm, :]       # (nm, A)  raw

2. scores are already in [0, 1] from the ONNX Sigmoid node — use as-is.
   For nc=1, scores is (1, A); flatten to (A,).

3. keep = scores > conf  ->  K candidates
   xywh -> xyxy in LETTERBOX coords

4. Class-agnostic NMS, iou_thr, top max_det=300  ->  M survivors

5. Mask decode:
     proto_flat = output1[0].reshape(nm, mh*mw)   # (32, 30976)
     masks_low  = sigmoid(coeffs_kept @ proto_flat)
                 .reshape(M, mh, mw)              # at 1/4 input res

6. Per detection, in proto coords:
     xyxy_proto = xyxy_letter / 4
     crop m_low[i] to box in proto grid
     cv2.resize(...) to (W_orig, H_orig), INTER_LINEAR
     paste at (x1_orig, y1_orig), threshold at mask_thres
     OR-merge into the final HxW mask

7. postprocess_mask(...)  (remove small components + MORPH_CLOSE)

Letterbox inverse:  (coord - pad) / r  gives original-image coord
Letterbox forward:  r = imgsz / max(H, W);  new_w = round(W*r);  new_h = round(H*r)
                    pad_w = (imgsz - new_w) / 2;  pad_h = (imgsz - new_h) / 2
                    canvas[y:y+new_h, x:x+new_w] = cv2.resize(img, (new_w, new_h))
                    canvas filled with 114
```

The full C#-port reference is the body of `decode_segmentation()` in
`postprocessing.py:121` — it is line-by-line translatable to C#.

## CLI

```bash
# single image (tiled by default; --no_tiling forces the legacy single-tile
# letterbox path that the parity test pins)
python -m waterseg_platform.cli \
    --image data/gf_floodnet/processed/images/test/gf_Australia_010_10_12.jpg \
    --output_dir runs/onnx_cli_smoke \
    --save_overlay --providers CPUExecutionProvider

# directory of images (200 images, ~80s on CPU)
python -m waterseg_platform.cli dir \
    --image_dir data/gf_floodnet/processed/images/test \
    --output_dir runs/onnx_platform_200 \
    --num 200 \
    --gt_mask_dir data/gf_floodnet/processed/masks/test

# large/rectangular image with overlap
python -m waterseg_platform.cli \
    --image path/to/1500x1000.jpg \
    --output_dir runs/tiled_smoke \
    --save_overlay --tile_size 704 --tile_overlap 100

# force the single-tile (legacy) path on a single image
python -m waterseg_platform.cli --image path/to/image.jpg --no_tiling

# enable conservative SAM 3 refinement
python -m waterseg_platform.cli \
    --config configs/onnx_platform_sam3_trial.yaml \
    --image path/to/image.jpg --no_tiling --save_overlay

# enable balanced fusion
python -m waterseg_platform.cli \
    --image path/to/image.jpg --sam3 --sam3_mode balanced --save_overlay

# enable experimental open fusion
python -m waterseg_platform.cli \
    --image path/to/image.jpg --sam3 --sam3_mode open --save_overlay

# launch the Gradio web UI (http://127.0.0.1:7860)
python -m waterseg_platform.cli ui

# dump the default config
python -m waterseg_platform.cli --dump-config configs/onnx_platform.yaml
```

CLI flags override values from `--config configs/onnx_platform.yaml`.
`--tile_size` / `--tile_overlap` set the config defaults; `--no_tiling`
forces the legacy single-tile path (default is tiled). `--sam3` and
`--no_sam3` override refinement, while `--sam3_mode` selects conservative,
balanced, or open fusion. `--sam3_open` remains as a deprecated alias.

## Gradio UI

The UI is launched by the `ui` subcommand. Two tabs:

* **Single image** — file upload, sliders for `conf`, `iou`, `mask_thres`,
  `min_area_ratio`, a MORPH_CLOSE checkbox, and a 3-panel output
  (image | mask | overlay with white contour). A **Tiling** accordion
  exposes a `Use tiling` checkbox (default on), a `tile size` number
  (0 = use imgsz), and a `tile overlap (px)` slider. A **SAM 3** accordion
  exposes `SAM 3 refinement` and a three-value fusion-mode selector.
* **Directory** — text inputs for `image_dir` and `output_dir`, a `num`
  cap, sliders for the same hyperparameters, and the same **Tiling**
  and **SAM 3** controls as the single-image tab. The metrics CSV is written to
  `<output_dir>/metrics.csv`.

The service is constructed once (lazy ORT session) and reused across
requests. Toggling **Use tiling** on/off produces visibly different
outputs for large/rectangular images; for 256×256 test tiles the two
paths are bit-equivalent.

## Parity vs Ultralytics baseline

See [`PARITY.md`](./PARITY.md) for the full 200-image parity report.
**Headline numbers** (200 test images, imgsz=704, ONNX vs .pt at the same
imgsz):

| metric                  | value    |
|-------------------------|----------|
| mean \|Δ IoU\|          | 0.0035   |
| 99th pct \|Δ IoU\|      | 0.047    |
| max \|Δ IoU\|           | 0.066    |
| mean \|Δ Dice\|         | 0.0028   |
| micro IoU (ONNX)        | 0.6957   |
| micro F1  (ONNX)        | 0.8205   |

The decoder is bit-exact on the median image. The tail is sub-pixel
letterbox INTER_LINEAR noise on tiny masks (< 1% of pixels).

## .NET port translation table

| Python module           | .NET class                | NuGet package                          |
|-------------------------|---------------------------|----------------------------------------|
| `config.py`             | `Config.cs` (record)      | `YamlDotNet`                           |
| `image_io.py`           | `ImageIo.cs`              | `OpenCvSharp4`                         |
| `preprocessing.py`      | `Preprocessor.cs`         | `OpenCvSharp4`                         |
| `engine.py`             | `OnnxSegmenter.cs`        | `Microsoft.ML.OnnxRuntime` (`.Gpu` for CUDA) |
| `postprocessing.py`     | `Postprocessor.cs`        | `OpenCvSharp4`                         |
| `tiling.py`             | `Tiling.cs`               | `OpenCvSharp4`                         |
| `pipeline.py`           | `SegmentationService.cs`  | (none)                                 |
| `visualization.py`      | `OverlayRenderer.cs`      | `OpenCvSharp4`                         |
| `metrics.py`            | `Metrics.cs`              | (none — pure math)                     |
| `ui_gradio.py`          | `MainWindow.xaml` + `MainViewModel.cs` | `OpenCvSharp4.Windows`, WPF |
| `cli.py`                | `Program.cs`              | `System.CommandLine`                   |
| `onnx/floodnet_binary_aug_yolov8m_1024.onnx` | same file (consumed by `InferenceSession`) | — |

Target framework: **.NET 8 (LTS)** with `<TargetFramework>net8.0-windows</TargetFramework>`
for the WPF app.

```text
D:/project/water_segment_cs/         (parallel to Python project)
├── WaterSegment.sln
├── src/
│   ├── WaterSegment.Core/           # class library — pure inference
│   │   ├── Config.cs
│   │   ├── OnnxSegmenter.cs
│   │   ├── Preprocessor.cs
│   │   ├── Postprocessor.cs
│   │   ├── SegmentationService.cs
│   │   └── ...
│   └── WaterSegment.App/            # WPF desktop app
│       ├── MainWindow.xaml
│       ├── MainViewModel.cs
│       └── ...
└── README.md
```

## Testing

```bash
cd D:/project/water_segment
C:/Users/17473/miniforge3/envs/torch_env/python.exe -m pytest tests/test_preprocessing.py tests/test_postprocessing.py tests/test_tiling.py -v
# ~5s — unit tests, no model load (test_tiling includes two integration
# tests gated on onnx/best.onnx being present)

# Slow parity test (loads the ONNX, runs 200 images, ~2 min on CPU)
C:/Users/17473/miniforge3/envs/torch_env/python.exe -m pytest tests/test_pipeline_parity.py -v
```

The parity test is opt-in (skipped by default if the baseline CSV or test
images are missing).

## Environment

Tested with: `onnxruntime-gpu 1.23.2`, `onnx 1.21.0`, `opencv-python 4.11`,
`numpy 1.26.4`, `pyyaml 6.0.3`, `tqdm 4.67.3`, `pillow 10.4.0`, `pandas`,
`gradio 5.x`.

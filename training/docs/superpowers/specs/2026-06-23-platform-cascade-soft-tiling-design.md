# Platform Cascade And Soft Tiling Design

## Goal

Improve cross-domain urban flood recall without requiring new manual labels,
while preserving the latest FloodNet model as the primary platform model.

## Cascade

The primary 1024 model runs first. If its foreground area ratio is below
`0.005` and the image aspect ratio is at least `1.15`, the platform runs two
existing road-flood models:

- `waterlogging_yolov8m_640_b8-3`
- `waterlogging_yolov8m_hard_finetune`

The fallback result is the pixel intersection of both masks. It replaces the
primary result only when its area ratio is at least `0.005`; otherwise the
primary result is retained. This makes the cascade conservative on true
negative images.

The fallback ONNX models use fixed 640 inputs and `conf=0.15`. Their final
intersection is still subject to the minimum consensus area, which limits the
extra false-positive risk from the lower per-model threshold.

## Soft Tiling

The current hard gate prevents tiles from adding detections outside the
coarse mask. Replace it with:

```text
combined_probability = max(coarse_binary, tile_weight * tiled_probability)
```

The default `tile_weight` is `0.7`. Every tile is evaluated; no tile is
discarded because of the coarse mask. The combined map is thresholded once
and passed through the existing connected-component and morphology cleanup.

## Platform Behavior

Single-image calls use the cascade by default. Tiled calls first produce the
soft-fused primary result, then apply the same low-response cascade. Diagnostic
info records whether fallback ran, whether it was accepted, and each candidate
area ratio.

Cascade can be disabled in configuration. Existing CLI model overrides remain
valid.

## Verification

- Unit tests cover cascade trigger, acceptance, rejection, and disabled mode.
- Unit tests prove tiled detections can survive outside an empty coarse mask.
- ONNX shape checks cover both fallback models.
- The supplied `test_image.png` must recover a substantial road-flood mask
  through fallback consensus while retaining a binary output.
- Existing platform unit tests remain green.

# FloodNet Binary Multiscale Crop Dataset Design

## Goal

Build a separate binary segmentation dataset whose training samples contain a
fixed multiscale crop pyramid. Images, binary masks, and YOLO segmentation
labels must remain synchronized after every geometric transformation.

## Dataset Policy

- Keep the source dataset and the existing augmented dataset unchanged.
- Generate six samples for every training image:
  - `global`: full image
  - `large_01`: crop covering 75%-90% of each source dimension
  - `medium_01`, `medium_02`: crops covering 50%-70%
  - `local_01`, `local_02`: crops covering 30%-50%
- Validation and test splits contain only the full-image `global` sample.
- Positive images use foreground-, boundary-, and context-aware crop centers.
- Negative images use deterministic random crop centers.

## Geometry And Labels

1. Apply one crop window to both the image and binary mask.
2. Resize the pair with the same scale while preserving aspect ratio.
3. Center-pad both outputs to 1024 x 1024.
4. Use bilinear interpolation for images and nearest-neighbor interpolation
   for masks.
5. Regenerate YOLO polygons from the final padded binary mask.

This avoids accumulated coordinate errors and guarantees that labels describe
the actual mask written to disk.

## Outputs

- Dataset: `data/floodnet_binary_multiscale_1024/processed`
- Dataset YAML:
  `data/floodnet_binary_multiscale_1024/processed/waterlogging_binary_multiscale.yaml`
- Crop report:
  `data/floodnet_binary_multiscale_1024/split_report/multiscale_crop_report.csv`
- Training config:
  `configs/train_floodnet_binary_multiscale_yolov8m.yaml`

## Verification

- Every image has a matching mask and label file.
- Every mask contains only values 0 and 1.
- Every YOLO label uses class 0 and normalized coordinates.
- Rasterized YOLO polygons overlap the saved masks at high IoU.
- Train/validation/test counts match the fixed split policy.


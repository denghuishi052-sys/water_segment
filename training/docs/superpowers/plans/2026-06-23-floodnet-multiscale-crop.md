# FloodNet Multiscale Crop Implementation Plan

1. Add focused tests for paired crop/letterbox geometry, fixed scale plans, and
   mask-derived YOLO labels.
2. Implement `scripts/12_build_floodnet_multiscale_crop.py` with deterministic
   crop selection and separate output directories.
3. Add a YOLOv8m training configuration pointing at the generated dataset.
4. Generate the complete dataset.
5. Validate file counts, binary masks, normalized class-0 labels, and sampled
   mask-to-polygon IoU.


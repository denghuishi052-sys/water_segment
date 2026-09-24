#!/usr/bin/env bash
set -e

python scripts/01_check_dataset.py \
  --image_dir data/raw/images \
  --mask_dir data/raw/masks \
  --output_dir data/split_report \
  --mask_mode red \
  --foreground_rgb 128,0,0 \
  --tolerance 10

python scripts/02_split_dataset.py \
  --image_dir data/raw/images \
  --mask_dir data/raw/masks \
  --output_dir data/processed \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42 \
  --group_by_prefix true \
  --mask_mode red \
  --foreground_rgb 128,0,0 \
  --tolerance 10



python scripts/03_convert_mask_to_yolo_seg.py \
  --processed_dir data/processed \
  --class_id 0 \
  --class_name waterlogging \
  --min_area 20 \
  --epsilon_ratio 0.002

python scripts/04_train_yolov8.py --config configs/train_yolov8s.yaml

python scripts/05_eval_test.py \
  --model runs/segment/waterlogging_yolov8s_640/weights/best.pt \
  --data data/processed/waterlogging.yaml \
  --split test \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.5 \
  --output_dir runs/eval


python scripts/01_check_dataset.py --image_dir .\training_set\JPEGImages\ --mask_dir .\training_set\SegmentationClass\ --output_dir data/split_report --mask_mode red --foreground_rgb 128,0,0 --tolerance 10
python scripts/02_split_dataset.py --image_dir .\training_set\JPEGImages\ --mask_dir .\training_set\SegmentationClass\ --output_dir data/processed --train_ratio 0.70 --val_ratio 0.15 --test_ratio 0.15 --seed 42 --group_by_prefix true --mask_mode red --foreground_rgb 128,0,0 --tolerance 10     
python scripts/03_convert_mask_to_yolo_seg.py --processed_dir data/processed --class_id 0 --class_name waterlogging --min_area 20 --epsilon_ratio 0.002
python scripts/04_train_yolov8.py --config configs/train_yolov8s.yaml
python scripts/05_eval_test.py --model runs/segment/waterlogging_yolov8s_640/weights/best.pt --data data/processed/waterlogging.yaml --split test --imgsz 640 --conf 0.25 --iou 0.5 --output_dir runs/eval
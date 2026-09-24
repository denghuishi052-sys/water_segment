$ErrorActionPreference = "Stop"

$Python = "C:\Users\17473\miniforge3\envs\torch_env\python.exe"
$MetricsCsv = ".\runs\eval_yolov8m_b8_3_test\metrics_test.csv"
$Data = ".\data\processed\waterlogging.yaml"
$HardDir = ".\data\hard_recall"
$Config = ".\configs\train_hard_recall_freeze.yaml"

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"
$env:PYTHONIOENCODING = "utf-8"

& $Python .\scripts\09_create_balanced_finetune_dataset.py `
  --metrics_csv $MetricsCsv `
  --data $Data `
  --output_dir $HardDir `
  --yaml_name waterlogging_hard_recall.yaml `
  --total_samples 160 `
  --normal_positive_ratio 0.45 `
  --normal_negative_ratio 0.10 `
  --hard_false_positive_negative_ratio 0.15 `
  --hard_false_negative_or_low_iou_positive_ratio 0.30 `
  --iou_threshold 0.4 `
  --val_ratio 0.15 `
  --seed 42 `
  --source_split test

& $Python .\scripts\08_finetune_hard_samples.py --config $Config

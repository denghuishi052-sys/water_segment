$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\project\water_segment'

$python = 'C:\Users\17473\miniforge3\envs\torch_env\python.exe'
$pipelineOut = 'runs\sam3_finetune\lora_pipeline.out.log'
$pipelineErr = 'runs\sam3_finetune\lora_pipeline.err.log'

"[$(Get-Date -Format o)] Starting LoRA proposal cache" | Out-File -FilePath $pipelineOut -Encoding utf8 -Append

if (!(Test-Path -LiteralPath 'runs\sam3_finetune\lora\best\adapter.pt')) {
  "[$(Get-Date -Format o)] Missing best adapter" | Out-File -FilePath $pipelineErr -Encoding utf8 -Append
  exit 2
}

& $python 'scripts\16_cache_sam3_proposals.py' `
  '--manifest' 'data\sam3_finetune\manifest_train_val.jsonl' `
  '--config' 'configs\onnx_platform_sam3_lora_trial.yaml' `
  '--output' 'data\sam3_finetune\lora_proposals.jsonl' `
  '--cache-dir' 'data\sam3_finetune\lora_cache' `
  1>> $pipelineOut 2>> $pipelineErr
if ($LASTEXITCODE -ne 0) {
  "[$(Get-Date -Format o)] LoRA proposal cache failed with exit code $LASTEXITCODE" | Out-File -FilePath $pipelineErr -Encoding utf8 -Append
  exit $LASTEXITCODE
}

"[$(Get-Date -Format o)] Starting LoRA selector training" | Out-File -FilePath $pipelineOut -Encoding utf8 -Append

& $python 'scripts\17_train_sam3_selector.py' `
  '--proposals' 'data\sam3_finetune\lora_proposals.jsonl' `
  '--output' 'runs\sam3_finetune\lora_selector.json' `
  1>> $pipelineOut 2>> $pipelineErr
if ($LASTEXITCODE -ne 0) {
  "[$(Get-Date -Format o)] LoRA selector training failed with exit code $LASTEXITCODE" | Out-File -FilePath $pipelineErr -Encoding utf8 -Append
  exit $LASTEXITCODE
}

"[$(Get-Date -Format o)] LoRA proposal + selector pipeline finished" | Out-File -FilePath $pipelineOut -Encoding utf8 -Append

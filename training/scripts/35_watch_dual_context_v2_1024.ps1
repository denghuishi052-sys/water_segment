param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingPid
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\17473\miniforge3\envs\torch_env\python.exe"
$run = Join-Path $root "runs\dual_context_water\v2_large_1024_finetune"
$checkpoint = Join-Path $run "best.pt"
$results = Join-Path $run "results.csv"
$onnx = Join-Path $root "onnx\dual_context_water_v2_large_1024_candidate.onnx"
$log = Join-Path $run "finalize.log"

$process = Get-Process -Id $TrainingPid -ErrorAction Stop
$process.WaitForExit()
$process.Refresh()
$exitCode = $process.ExitCode
if ($null -ne $exitCode -and $exitCode -ne 0) {
    "Training process $TrainingPid failed with exit code $exitCode." |
        Out-File -FilePath $log -Encoding utf8
    exit $exitCode
}

if (-not (Test-Path $checkpoint) -or -not (Test-Path $results)) {
    "Training ended without the expected checkpoint or results file." |
        Out-File -FilePath $log -Encoding utf8
    exit 1
}
$lastEpoch = [int](Import-Csv $results | Select-Object -Last 1).epoch
if ($lastEpoch -lt 100) {
    "Training ended at epoch $lastEpoch, before the required minimum of 100." |
        Out-File -FilePath $log -Encoding utf8
    exit 1
}

Push-Location $root
try {
    & $python "scripts\31_evaluate_dual_context.py" `
        --checkpoint $checkpoint `
        --manifest "data\satellite_adaptation_1024_v1\satellite_val.csv" *>&1 |
        Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    & $python "scripts\28_export_dual_context_onnx.py" `
        --checkpoint $checkpoint `
        --output $onnx *>&1 |
        Tee-Object -FilePath $log -Append
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

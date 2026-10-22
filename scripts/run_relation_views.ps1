param(
    [string]$Python = 'D:\App\Code\Anaconda\envs\MFP-Nova\python.exe',
    [string]$Device = 'cuda:0',
    [int]$CpuThreads = 4,
    [string[]]$Tasks = @('major15', 'fine39')
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:OMP_NUM_THREADS = "$CpuThreads"
$env:MKL_NUM_THREADS = "$CpuThreads"
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'
$runs = @()
foreach ($task in $Tasks) {
    $run = "trirelay_${task}_esm650m_ema_relationview_seed42_rtx5070_$tag"
    & $Python -u -m mfp_nova.cli train `
        --model trirelay --task $task --esm 650m --seed 42 `
        --device $Device --precision fp32 --batch-size 64 `
        --ema-decay 0.999 --ema-start-epoch 6 --shared-private `
        --branch-supervision shared_private --evidence-fusion normalized_competitive `
        --relation-view-weight 0.1 --relation-dropout 0.2 `
        --consistency-weight 0 --function-aux-weight 0 `
        --reconstruction-gradient sum --name $run
    if ($LASTEXITCODE -ne 0) { throw "$task training failed; remaining tasks were not started." }
    $runs += "runs/$run"
}
$out = "runs/trirelay_relationview_seed42_rtx5070_${tag}_val"
& $Python -u -m mfp_nova.branch_evidence --runs $runs `
    --device $Device --batch-size 64 --cpu-threads $CpuThreads --output $out
if ($LASTEXITCODE -ne 0) { throw 'Validation export failed.' }
Write-Host "Transfer: ${out}.zip"

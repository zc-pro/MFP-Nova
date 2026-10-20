[CmdletBinding()]
param(
    [string]$Python = 'D:\App\Code\Anaconda\envs\MFP-Nova\python.exe',
    [string]$Device = 'cuda:0',
    [int]$BatchSize = 64
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'
$output = Join-Path $projectRoot "runs\trirelay_val_comparison_seed42_$tag"
$jobs = @(
    @{ Task = 'major15'; Variant = 'ema_only'; Run = 'trirelay_major15_ema_seed42_20261018_215416' },
    @{ Task = 'major15'; Variant = 'consistency'; Run = 'trirelay_major15_esm650m_ema_consistency_seed42_20261020_093530' },
    @{ Task = 'fine39'; Variant = 'ema_only'; Run = 'trirelay_fine39_ema_seed42_20261019_090117' },
    @{ Task = 'fine39'; Variant = 'consistency'; Run = 'trirelay_fine39_esm650m_ema_consistency_seed42_20261020_125726' }
)

Push-Location -LiteralPath $projectRoot
try {
    New-Item -ItemType Directory -Path $output | Out-Null
    foreach ($job in $jobs) {
        $stem = "$($job.Task)_$($job.Variant)"
        $checkpoint = Join-Path $projectRoot "runs\$($job.Run)\best.pt"
        Write-Host "Exporting Validation: $stem | $checkpoint"
        & $Python -u -m mfp_nova.cli evaluate --checkpoint $checkpoint --split val --device $Device --batch-size $BatchSize --output (Join-Path $output "$stem.json") --predictions (Join-Path $output "$stem.npz")
        if ($LASTEXITCODE -ne 0) { throw "Validation export failed: $stem" }
    }
    $zip = Join-Path ([Environment]::GetFolderPath('Desktop')) "trirelay_val_comparison_seed42_$tag.zip"
    Compress-Archive -Path (Join-Path $output '*.json'), (Join-Path $output '*.npz') -DestinationPath $zip
    Write-Host "Completed four Validation exports. No training or Test evaluation."
    Write-Host "Transfer this file: $zip"
}
finally {
    Pop-Location
}

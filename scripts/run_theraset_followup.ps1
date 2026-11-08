param(
    [ValidateSet('plan', 'run', 'pause', 'pack', 'diagnose')]
    [string]$Action = 'run',
    [string]$Python = 'python',
    [int]$GpuId = 0,
    [int]$CpuThreads = 4,
    [string]$RunRoot = 'runs/theraset_followup_seed42_rtx5070',
    [string]$ReuseRoot = 'runs/theraset_ablation_seed42_rtx5070',
    [string[]]$Tasks = @('fine39'),
    [string[]]$Suites = @('traditional5', 'micro_f1'),
    [string[]]$Variants = @('no_label_context_single_prototype'),
    [switch]$Background
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = Join-Path $PWD.Path 'src'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:CUDA_VISIBLE_DEVICES = "$GpuId"
$env:OMP_NUM_THREADS = "$CpuThreads"
$env:MKL_NUM_THREADS = "$CpuThreads"
$env:OPENBLAS_NUM_THREADS = "$CpuThreads"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

$pythonPath = (Get-Command $Python -ErrorAction Stop).Source
$cliArgs = @('-u', '-m', 'mfp_nova.theraset_ablation', $Action,
    '--output', $RunRoot, '--reuse-root', $ReuseRoot, '--device', 'cuda:0',
    '--cpu-threads', "$CpuThreads", '--platform', 'rtx5070', '--tasks') + $Tasks +
    @('--suites') + $Suites + @('--variants') + $Variants
if ($Action -eq 'run') { $cliArgs += '--diagnostics' }

if ($Background -and $Action -in @('run', 'diagnose')) {
    New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
    $tag = Get-Date -Format 'yyyyMMdd_HHmmss'
    $log = Join-Path $RunRoot "${Action}_${tag}.log"
    $errorLog = Join-Path $RunRoot "${Action}_${tag}.err.log"
    # File paths cannot contain a double quote; quote each native argument for spaces.
    $nativeArguments = ($cliArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $process = Start-Process -FilePath $pythonPath -ArgumentList $nativeArguments `
        -WorkingDirectory $PWD.Path -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError $errorLog
    Write-Host "PID=$($process.Id)"
    Write-Host "Log: $log"
    Write-Host "Stderr: $errorLog"
} else {
    & $pythonPath @cliArgs
    if ($LASTEXITCODE -ne 0) { throw "TheraSet $Action failed (exit $LASTEXITCODE)." }
}

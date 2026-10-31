param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('prepare', 'train', 'evaluate', 'pack')]
    [string]$Action,
    [string]$Output = 'runs/helixset_therapep3_seed42',
    [ValidateSet('therapep49', 'therapep49_positive', 'therapep48')]
    [string[]]$Tasks = @('therapep49', 'therapep49_positive', 'therapep48'),
    [ValidateSet('traditional5', 'micro_f1')]
    [string[]]$Suites = @('traditional5', 'micro_f1'),
    [string]$Python = 'D:\App\Code\Anaconda\envs\MFP-Nova\python.exe',
    [string]$Gpu = '0',
    [int]$Threads = 4
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$Output = [IO.Path]::GetFullPath($Output)
$env:PYTHONPATH = Join-Path $PWD.Path 'src'
$env:CUDA_VISIBLE_DEVICES = $Gpu
$env:OMP_NUM_THREADS = "$Threads"
$env:MKL_NUM_THREADS = "$Threads"
$FeatureRoot = 'features/therapep_ai/esm2_650m'
$PeptideRoot = 'features/therapep_ai/esm2_peptide_150m'
$Roots = @{
    therapep49 = 'data/therapep_ai'
    therapep49_positive = 'data/therapep49_positive'
    therapep48 = 'data/therapep48'
}

function Invoke-Mfp {
    param([string[]]$Arguments, [string]$LogPath)
    # Windows PowerShell turns native stderr warnings into error records.
    # Judge the subprocess by its exit code, preserving stderr in the log.
    $ErrorActionPreference = 'Continue'
    if ($LogPath) {
        & $Python -u -m mfp_nova.cli @Arguments 2>&1 |
            Tee-Object -FilePath $LogPath -Append
    } else {
        & $Python -u -m mfp_nova.cli @Arguments
    }
    if ($LASTEXITCODE -ne 0) {
        throw "mfp_nova exited with code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Prepare-Caches {
    param([string[]]$Splits)
    foreach ($View in @('primary', 'peptide')) {
        $Cache = if ($View -eq 'primary') { $FeatureRoot } else { $PeptideRoot }
        foreach ($Split in $Splits) {
            if (Test-Path -LiteralPath "$Cache/$Split/metadata.json") {
                Write-Host "Reuse cache: $Cache/$Split"
                continue
            }
            Invoke-Mfp -Arguments @('features', '--config', 'configs/helixset.yaml',
                '--track', 'therapep49', '--encoder-view', $View, '--splits', $Split,
                '--device', 'cuda:0', '--batch-size', '8')
        }
    }
}

switch ($Action) {
    'prepare' {
        Invoke-Mfp -Arguments @('download-therapep')
        Invoke-Mfp -Arguments @('prepare-therapep-variants')
        Prepare-Caches -Splits @('train', 'val')
    }
    'train' {
        New-Item -ItemType Directory -Path $Output -Force | Out-Null
        foreach ($Task in $Tasks) {
            foreach ($Suite in $Suites) {
                $RunDir = Join-Path (Join-Path $Output $Task) $Suite
                if (Test-Path -LiteralPath "$RunDir/results.json") {
                    Write-Host "Completed: $Task/$Suite"
                    continue
                }
                $Log = Join-Path $Output "${Task}_${Suite}.log"
                $Arguments = @('train', '--model', 'helixset', '--config', 'configs/helixset.yaml',
                    '--task', $Task, '--data-root', $Roots[$Task],
                    '--feature-root', $FeatureRoot, '--peptide-feature-root', $PeptideRoot,
                    '--seed', '42', '--device', 'cuda:0', '--precision', 'fp32',
                    '--batch-size', '16', '--accumulation-steps', '4', '--num-workers', '0',
                    '--rdrop-weight', '0.1', '--ema-decay', '0.999', '--ema-start-epoch', '6',
                    '--metric-suite', $Suite, '--threshold-strategy', 'global',
                    '--epochs', '100', '--name', $RunDir)
                if (Test-Path -LiteralPath "$RunDir/last.pt") {
                    $Arguments += @('--resume', "$RunDir/last.pt", '--resume-in-place')
                } elseif (Test-Path -LiteralPath $RunDir) {
                    # Ctrl+C during epoch one leaves a directory but no committed checkpoint.
                    $Arguments += @('--resume-in-place')
                }
                Write-Host "Training: $Task/$Suite -> $RunDir"
                Invoke-Mfp -Arguments $Arguments -LogPath $Log
                if (-not (Test-Path -LiteralPath "$RunDir/results.json")) {
                    throw "Training paused before completion: $Task/$Suite"
                }
            }
        }
    }
    'evaluate' {
        Prepare-Caches -Splits @('test')
        foreach ($Task in $Tasks) {
            foreach ($Suite in $Suites) {
                $RunDir = Join-Path (Join-Path $Output $Task) $Suite
                foreach ($Split in @('val', 'test')) {
                    $ReportPath = "$RunDir/${Split}.json"
                    $Predictions = "$RunDir/${Split}_predictions.npz"
                    if (-not ((Test-Path -LiteralPath $ReportPath) -and
                              (Test-Path -LiteralPath $Predictions))) {
                        Invoke-Mfp -Arguments @('evaluate', '--checkpoint', "$RunDir/best.pt",
                            '--data-root', $Roots[$Task], '--feature-root', $FeatureRoot,
                            '--peptide-feature-root', $PeptideRoot, '--split', $Split,
                            '--device', 'cuda:0', '--batch-size', '16',
                            '--output', $ReportPath, '--predictions', $Predictions)
                    }
                    $Report = Get-Content -LiteralPath $ReportPath -Raw | ConvertFrom-Json
                    Write-Host "$Task/$Suite/$Split - all outputs"
                    $Report.metrics | Select-Object accuracy, precision, coverage,
                        absolute_true, absolute_false, micro_f1, macro_f1, micro_ap, macro_ap | Format-List
                    Write-Host "$Task/$Suite/$Split - positive functional48"
                    $Report.therapep.positive_functions48 | Select-Object accuracy, precision, coverage,
                        absolute_true, absolute_false, micro_f1, macro_f1, micro_ap, macro_ap | Format-List
                }
            }
        }
    }
    'pack' {
        $ZipPath = Join-Path ([Environment]::GetFolderPath('Desktop')) `
            "helixset_therapep3_seed42_$(Split-Path $Output -Leaf)_results_$(Get-Date -Format 'yyyyMMdd_HHmmss').zip"
        @'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import csv, io, json, sys

root, archive = Path(sys.argv[1]), Path(sys.argv[2])
tasks, suites = sys.argv[3].split(','), sys.argv[4].split(',')
metrics = ('accuracy', 'precision', 'coverage', 'absolute_true', 'absolute_false',
           'micro_f1', 'macro_f1', 'micro_ap', 'macro_ap')
rows, paths = [], []
for task in tasks:
    for suite in suites:
        directory = root/task/suite
        result = json.loads((directory/'results.json').read_text(encoding='utf-8'))
        for split in ('val', 'test'):
            report = json.loads((directory/f'{split}.json').read_text(encoding='utf-8'))
            for scope, values in (('all_outputs', report['metrics']),
                                  ('positive_functions48', report['therapep']['positive_functions48'])):
                rows.append(dict(task=task, suite=suite, split=split, scope=scope,
                    rows=report['rows'] if scope == 'all_outputs' else report['groups']['therapeutic_positive']['rows'],
                    best_epoch=result['best_epoch'], threshold=json.dumps(report['threshold']),
                    **{k: values[k] for k in metrics}))
        paths.extend(p for p in directory.iterdir()
                     if p.is_file() and p.suffix in ('.json', '.jsonl', '.npz', '.log'))
        log = root/f'{task}_{suite}.log'
        if log.is_file():
            paths.append(log)
with ZipFile(archive, 'x', ZIP_DEFLATED) as out:
    for p in paths:
        out.write(p, p.relative_to(root).as_posix())
    summary = io.StringIO(newline='')
    writer = csv.DictWriter(summary, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    out.writestr('summary.csv', summary.getvalue().encode('utf-8-sig'))
print('RESULT ZIP:', archive.resolve())
'@ | & $Python - $Output $ZipPath ($Tasks -join ',') ($Suites -join ',')
        if ($LASTEXITCODE -ne 0) { throw 'Result packaging failed.' }
    }
}

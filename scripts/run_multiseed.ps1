[CmdletBinding()]
param(
    [ValidateSet('all', 'trirelay', 'geofactor', 'protoroute', 'helixset', 'theraset')][string]$Model = 'all',
    [ValidateSet('all', 'major15', 'fine39', 'therapep49')][string]$Task = 'all',
    [string]$Seeds = '42,101,202,303,404',
    [string]$Device = 'cuda:0',
    [ValidateRange(1, 50)][int]$MaxRuns = 1,
    [switch]$All,
    [switch]$DryRun,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
if ($All -and $PSBoundParameters.ContainsKey('MaxRuns')) {
    throw '-All and -MaxRuns cannot be combined.'
}
$pythonCommand = Get-Command $Python -ErrorAction Stop
$projectRoot = Split-Path $PSScriptRoot -Parent
Push-Location -LiteralPath $projectRoot
try {
    $selection = @('--model', $Model, '--task', $Task, '--seeds', $Seeds)
    $planText = & $pythonCommand.Source -m mfp_nova.experiments plan @selection --json
    if ($LASTEXITCODE -ne 0) { throw 'Could not read the experiment plan; see the error above.' }
    $plan = ($planText -join "`n") | ConvertFrom-Json
    Write-Host ('Completed {0}/{1}; fixed seeds: 42,101,202,303,404.' -f $plan.counts.completed, $plan.total)
    foreach ($job in $plan.jobs) {
        Write-Host ('{0,-25} {1,-10} {2,-7} seed={3}' -f $job.status, $job.model, $job.task, $job.seed)
    }
    $pending = @($plan.jobs | Where-Object { $_.status -ne 'completed' })
    $limit = if ($All) { $pending.Count } else { $MaxRuns }
    Write-Host ('This invocation will run at most {0} experiment(s), serially.' -f $limit)
    if ($DryRun) {
        Write-Host 'DryRun: no training, checkpoint loading, or files written.'
        return
    }
    $executed = 0
    foreach ($job in $pending) {
        if ($executed -ge $limit) { break }
        if ($job.status -eq 'running') {
            throw ('Already running: {0}. No duplicate experiment was started.' -f $job.run_name)
        }
        if ($job.status -eq 'incompatible') {
            throw ('Incompatible experiment: {0}. {1}' -f $job.run_name, ($job.issues -join '; '))
        }
        # Refresh the live plan before launch, so a second terminal does not
        # knowingly resume an experiment that is already training.
        $freshText = & $pythonCommand.Source -m mfp_nova.experiments plan --json
        if ($LASTEXITCODE -ne 0) { throw 'Could not refresh live experiment status.' }
        $fresh = ($freshText -join "`n") | ConvertFrom-Json
        if (@($fresh.jobs | Where-Object { $_.running -and @($_.running.PSObject.Properties).Count -gt 0 }).Count -gt 0) {
            throw 'An MFP five-seed experiment is already running. No parallel training was started.'
        }
        $job = $fresh.jobs | Where-Object { $_.model -eq $job.model -and $_.task -eq $job.task -and $_.seed -eq $job.seed }
        if ($job.status -eq 'completed') { continue }
        if ($job.status -eq 'incompatible') { throw ($job.issues -join '; ') }
        $settings = $job.train_settings
        $arguments = @('-m', 'mfp_nova.cli', 'train', '--model', $job.model, '--task', $job.task,
                       '--esm', '650m', '--seed', [string]$job.seed, '--device', $Device,
                       '--precision', $settings.precision, '--batch-size', [string]$settings.batch_size,
                       '--accumulation-steps', [string]$settings.accumulation_steps,
                       '--metric-suite', $settings.metric_suite, '--name', $job.run_name)
        if ($null -ne $settings.ema_decay) {
            $arguments += @('--ema-decay', [string]$settings.ema_decay,
                            '--ema-start-epoch', [string]$settings.ema_start_epoch)
        }
        if ($job.resume) { $arguments += @('--resume', $job.resume) }
        Write-Host ('[{0}/{1}] {2}' -f ($executed+1), $limit, $job.run_name)
        & $pythonCommand.Source @arguments
        if ($LASTEXITCODE -ne 0) { throw ('Training failed: {0}; subsequent jobs were not started.' -f $job.run_name) }
        $executed++
    }
    Write-Host ('Finished {0} experiment(s). Run this command again when convenient.' -f $executed)
    & $pythonCommand.Source -m mfp_nova.experiments plan @selection
    if ($LASTEXITCODE -ne 0) { throw 'Could not display the final experiment status.' }
}
finally {
    Pop-Location
}

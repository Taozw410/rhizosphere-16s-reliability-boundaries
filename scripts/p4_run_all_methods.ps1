param(
    [Parameter(Mandatory = $true)][string]$InputDir,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$Scenario = "unspecified",
    [int]$Seed = 4000,
    [string[]]$Methods = @("aldex2", "ancombc2", "linda", "maaslin2"),
    [string]$Root = (Get-Location).Path,
    [string]$Renv = "D:\bioenv\envs\riz16s",
    [string]$Conda = "D:\ProgramData\miniconda3\Scripts\conda.exe",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$runId = "{0}_{1}" -f (Get-Date -Format "yyyyMMdd_HHmmss"), $Seed
$work = Join-Path $Root "output_work\p4_method_runs\$runId"
$stageInput = Join-Path $work "input"
$results = Join-Path $work "results"
New-Item -ItemType Directory -Force -Path $stageInput,$results,$OutputDir | Out-Null
Copy-Item -Path (Join-Path $InputDir "*") -Destination $stageInput -Recurse -Force

$runnerSource = Join-Path $Root "scripts\p4_unified_da_runner.R"
$runner = Join-Path $work "p4_unified_da_runner.R"
Copy-Item -LiteralPath $runnerSource -Destination $runner -Force

$env:R_LIBS_USER = (Join-Path $Root "Rlib")
$wrapperRows = @()
foreach ($method in $Methods) {
    $csv = Join-Path $results "$method.csv"
    $json = Join-Path $results "$method.json"
    $log = Join-Path $results "$method.log"
    Write-Output "Running $method ..."
    & $Conda run --no-capture-output -p $Renv Rscript `
        $runner $method $stageInput $csv $json $Seed $Scenario *> $log
    $exit = $LASTEXITCODE
    $status = "unknown"
    if (Test-Path $json) {
        $status = (Get-Content -LiteralPath $json -Raw -Encoding UTF8 | ConvertFrom-Json).status
    }
    $wrapperRows += [pscustomobject]@{
        method = $method
        exit_code = $exit
        status = $status
        csv = $csv
        json = $json
        log = $log
    }
}

$prefix = Join-Path $work "method_unified"
& $Python (Join-Path $Root "scripts\p4_collect_method_results.py") `
    --results-dir $results `
    --truth (Join-Path $stageInput "truth.csv") `
    --scenario $Scenario `
    --out-prefix $prefix
if ($LASTEXITCODE -ne 0) {
    throw "Result collection failed"
}

Copy-Item -Path (Join-Path $results "*") -Destination $OutputDir -Force
foreach ($suffix in @("_long.csv", "_metrics.csv", "_manifest.csv")) {
    Copy-Item -LiteralPath "$prefix$suffix" -Destination $OutputDir -Force
}
$wrapperRows | Export-Csv -LiteralPath (Join-Path $OutputDir "method_wrapper_status.csv") `
    -NoTypeInformation -Encoding UTF8
Write-Output "Finished. Results copied to $OutputDir"

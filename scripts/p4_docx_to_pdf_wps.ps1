param(
    [Parameter(Mandatory = $true)][string]$InPath,
    [Parameter(Mandatory = $true)][string]$OutPath
)

$ErrorActionPreference = "Stop"
$resolvedInput = (Resolve-Path -LiteralPath $InPath).Path
$outDirectory = Split-Path -Parent $OutPath
if (-not (Test-Path -LiteralPath $outDirectory)) {
    New-Item -ItemType Directory -Path $outDirectory | Out-Null
}
$resolvedOutput = [System.IO.Path]::GetFullPath($OutPath)

$app = $null
$doc = $null
try {
    $app = New-Object -ComObject 'KWPS.Application'
    $app.Visible = $false
    $app.DisplayAlerts = 0
    $doc = $app.Documents.Open($resolvedInput, $false, $true)
    try {
        $doc.SaveAs([ref]$resolvedOutput, [ref]17)
    }
    catch {
        $doc.ExportAsFixedFormat($resolvedOutput, 17)
    }
}
finally {
    if ($null -ne $doc) {
        $doc.Close($false)
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) | Out-Null
    }
    if ($null -ne $app) {
        $app.Quit()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

if (-not (Test-Path -LiteralPath $resolvedOutput)) {
    throw "WPS did not produce the requested PDF."
}

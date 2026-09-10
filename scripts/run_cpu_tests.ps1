param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [string[]]$TestPaths = @('tests')
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$evidenceDir = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
$started = [DateTime]::UtcNow.ToString('o')
$arguments = @('-m', 'pytest') + $TestPaths + @('-q', '--tb=short', "--junitxml=$evidenceDir/results.xml", "--basetemp=$evidenceDir/tmp")
$PSNativeCommandUseErrorActionPreference = $false
& $Python @arguments *> (Join-Path $evidenceDir 'pytest.log')
$result = $LASTEXITCODE
[ordered]@{
    pid = $PID
    python = $Python
    arguments = $arguments
    cwd = $repoRoot
    started_utc = $started
    finished_utc = [DateTime]::UtcNow.ToString('o')
    exit_code = $result
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $evidenceDir 'result.json') -Encoding UTF8
exit $result

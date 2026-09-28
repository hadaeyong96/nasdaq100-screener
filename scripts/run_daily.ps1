<#
.SYNOPSIS
    Daily automation entry point for the Nasdaq 100 screener (Windows Task Scheduler).

.DESCRIPTION
    Moves to the project root, runs engine.daily in live mode then paper mode
    (--no-send), and appends a log to logs\daily_YYYY-MM-DD.log (that folder is
    in .gitignore). If either step exits non-zero, sends one short Telegram
    error alert via notify.telegram.notify_ops_error (through
    scripts\send_ops_alert.py) instead of building a new sender. .env values
    are never printed here or anywhere downstream -- only presence/length is
    ever logged (CLAUDE.md security rule).

    NOTE ON LANGUAGE: every literal string in *this file* is English-only on
    purpose, even though the rest of the project uses Korean comments/output.
    Windows PowerShell 5.1 parses a .ps1 file using the system codepage
    unless the file has a UTF-8 BOM, so non-ASCII literals written straight
    into a script can get corrupted before they're ever logged or sent as a
    Telegram alert (confirmed: the first version of this script did exactly
    that). Python's own stdout/stderr is unaffected by this -- engine.daily
    already reconfigures its streams to UTF-8, and that output is captured
    and logged correctly (see the Console/PYTHONUTF8 settings below) -- so the
    Korean text you see in the log from the python steps is fine. Only text
    defined directly in this .ps1 file must stay ASCII.

    This project has no dedicated virtualenv -- it uses the system Python
    install. The Get-Command fallback below only guards against a scheduled
    task's PATH not matching an interactive session's PATH.
#>

$ErrorActionPreference = "Stop"

# Always run from the project root, regardless of the scheduled task's start-in directory.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# Make sure UTF-8 text from the python child processes (Korean, arrows, etc.)
# round-trips correctly through PowerShell's output capture and into the log,
# regardless of the console's codepage (e.g. cp949 on a Korean-locale machine).
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    $PythonExe = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"
}
if (-not (Test-Path $PythonExe)) {
    throw "Could not find a python executable: $PythonExe"
}

$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir ("daily_{0}.log" -f (Get-Date -Format "yyyy-MM-dd"))

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $LogPath -Value $line -Encoding utf8
}

function Invoke-Step {
    <# Runs $PythonExe with the given arguments, appends all stdout/stderr to
       the log, and returns whether it exited 0. #>
    param([string]$Name, [string[]]$Arguments)

    Write-Log "START: $Name ($PythonExe $($Arguments -join ' '))"
    $output = & $PythonExe @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    foreach ($line in $output) {
        Add-Content -Path $LogPath -Value $line.ToString() -Encoding utf8
    }
    if ($exitCode -ne 0) {
        Write-Log "FAILED: $Name (exit code $exitCode)"
        return $false
    }
    Write-Log "DONE: $Name"
    return $true
}

Write-Log "=== daily run start ==="

$liveOk = Invoke-Step -Name "live" -Arguments @("-m", "engine.daily", "--mode", "live")
$paperOk = Invoke-Step -Name "paper" -Arguments @("-m", "engine.daily", "--mode", "paper", "--no-send")

if (-not $liveOk -or -not $paperOk) {
    $failed = @()
    if (-not $liveOk) { $failed += "live" }
    if (-not $paperOk) { $failed += "paper" }
    $alertMessage = "Nasdaq100 screener daily run FAILED: $($failed -join ', '). Log: $LogPath"

    $alertScript = Join-Path $ProjectRoot "scripts\send_ops_alert.py"
    $alertOk = Invoke-Step -Name "error alert" -Arguments @($alertScript, $alertMessage)
    if (-not $alertOk) {
        Write-Log "WARNING: failed to send the error alert too (missing token or network issue -- values are never logged)."
    }

    Write-Log "=== daily run end (FAILED) ==="
    exit 1
}

Write-Log "=== daily run end (ok) ==="
exit 0

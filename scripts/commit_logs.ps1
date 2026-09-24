# commit_logs.ps1
# Commits and pushes ONLY the logs/ folder. Run hourly by Task Scheduler.
# Safe to run any time: exits quietly when there is nothing new.
# Continue: git writes progress to stderr, which PS 5.1 would treat as fatal.
$ErrorActionPreference = "Continue"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$runtimeLogs = Join-Path $repo "runtime_logs"
New-Item -ItemType Directory -Force $runtimeLogs | Out-Null
$logFile = Join-Path $runtimeLogs "commit_logs.log"

function Write-Log($message) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message" | Out-File -Append -Encoding utf8 $logFile
}

if (-not (Test-Path (Join-Path $repo "logs"))) {
    Write-Log "No logs/ folder yet"
    exit 0
}

git add -- logs
git diff --cached --quiet -- logs
if ($LASTEXITCODE -eq 0) {
    Write-Log "Nothing new to commit"
    exit 0
}

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
# The pathspec limits the commit to logs/, even if other files are staged.
git commit -m "logs: $stamp" -- logs
if ($LASTEXITCODE -ne 0) { Write-Log "Commit failed"; exit 1 }

git pull --rebase --autostash
if ($LASTEXITCODE -ne 0) {
    git rebase --abort 2>$null
    Write-Log "Pull failed; commit kept locally, will push next hour"
    exit 1
}

git push
if ($LASTEXITCODE -ne 0) { Write-Log "Push failed; will retry next hour"; exit 1 }

Write-Log "Pushed logs: $stamp"

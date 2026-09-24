# register_tasks.ps1
# Creates two Windows Task Scheduler jobs for the current user:
#   TradingBotV1-Live        weekdays 13:15 local (+ at logon): runs one session
#   TradingBotV1-CommitLogs  weekdays hourly 14:00-22:00 local: pushes logs/
# Times assume UK time; the bot waits for Alpaca's market clock, so starting
# a bit early is fine. Re-run this script to update the tasks.
# Remove with: Unregister-ScheduledTask TradingBotV1-* -Confirm:$false
$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo "venv\Scripts\python.exe"
$weekdays = "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"

$settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 10)

# --- Live trading session ---
$liveAction = New-ScheduledTaskAction -Execute $python -Argument "-m tradingbot live" -WorkingDirectory $repo
$liveTriggers = @(
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At "13:15"),
    # Resume after a reboot mid-session (exits immediately if the market is far from open).
    (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME)
)
Register-ScheduledTask -TaskName "TradingBotV1-Live" -Action $liveAction `
    -Trigger $liveTriggers -Settings $settings -Force | Out-Null

# --- Hourly log commits ---
$commitAction = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$repo\scripts\commit_logs.ps1`"" `
    -WorkingDirectory $repo
$commitTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At "14:00"
$commitTrigger.Repetition = (New-ScheduledTaskTrigger -Once -At "14:00" `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Hours 8)).Repetition
Register-ScheduledTask -TaskName "TradingBotV1-CommitLogs" -Action $commitAction `
    -Trigger $commitTrigger -Settings $settings -Force | Out-Null

Get-ScheduledTask -TaskName "TradingBotV1-*" | Format-Table TaskName, State
Write-Host "Registered. Make sure the PC doesn't sleep during 13:15-22:00 on weekdays."

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Environnement absent. Lancez d’abord .\scripts\install_windows.ps1"
}

$ServerCommand = "Set-Location '$ProjectRoot'; & '$PythonExe' manage.py runserver 127.0.0.1:8000"
$WorkerCommand = "Set-Location '$ProjectRoot'; & '$PythonExe' manage.py run_analysis_worker"

Start-Process powershell -ArgumentList "-NoExit", "-Command", $WorkerCommand
Start-Process powershell -ArgumentList "-NoExit", "-Command", $ServerCommand
Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:8000"

Write-Host "Serveur et worker démarrés. Fermez leurs deux fenêtres pour arrêter l’application." -ForegroundColor Green

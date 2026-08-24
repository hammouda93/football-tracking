param(
    [switch]$WithML
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Football Tracking - installation locale" -ForegroundColor Green

$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $PythonLauncher) {
    throw "Python Launcher est introuvable. Installez Python 3.12 depuis python.org, cochez Add Python to PATH, puis redemarrez VS Code."
}

& py -3.12 --version 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12 est requis. Installez-le, fermez VS Code, puis relancez ce script."
}

if (-not (Test-Path ".venv")) {
    & py -3.12 -m venv .venv
}

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$EnvironmentVersion = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($EnvironmentVersion -ne "3.12") {
    throw "Le dossier .venv utilise Python $EnvironmentVersion. Supprimez uniquement .venv puis relancez ce script avec Python 3.12."
}

& $PythonExe -m pip install --upgrade pip wheel setuptools
& $PythonExe -m pip install -r requirements.txt

if ($WithML) {
    Write-Host "Installation des dependances ML..." -ForegroundColor Cyan
    & $PythonExe -m pip install -r requirements-ml.txt
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

& $PythonExe manage.py migrate
& $PythonExe manage.py check
& $PythonExe manage.py diagnose

Write-Host "Installation terminee." -ForegroundColor Green
Write-Host "Demarrez avec: .\scripts\start_local.ps1"

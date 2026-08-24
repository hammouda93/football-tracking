param(
    [switch]$WithML
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Football Tracking - installation locale" -ForegroundColor Green

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python est introuvable. Installez Python 3.12 et ajoutez-le au PATH."
}

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
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

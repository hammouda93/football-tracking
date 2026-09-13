param(
    [switch]$AllowCpu,
    [string]$Distro = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$EnvPath = Join-Path $ProjectRoot ".env"
$LinuxRoot = ""

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "WSL2 est requis. Installez Ubuntu avec: wsl --install -d Ubuntu-22.04"
}

$WslArgs = @()
if ($Distro) {
    $WslArgs += @("-d", $Distro)
}

function Invoke-WslText {
    param([string[]]$Arguments)
    $result = & wsl.exe @WslArgs @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "La commande WSL a échoué: $($Arguments -join ' ')"
    }
    return ($result | Out-String).Trim()
}

$WslProjectRoot = Invoke-WslText @("wslpath", "-a", $ProjectRoot)
$LinuxHome = Invoke-WslText @("sh", "-lc", 'printf %s "$HOME"')
$LinuxRoot = "$LinuxHome/.football-tracking/gsr/sn-gamestate"
$InstallScript = "$WslProjectRoot/scripts/gsr/install_sn_gamestate_wsl.sh"

Write-Host "Installation isolée de TrackLab + sn-gamestate dans WSL2..."
& wsl.exe @WslArgs bash $InstallScript $LinuxRoot
if ($LASTEXITCODE -ne 0) {
    throw "L'installation sn-gamestate a échoué. L'environnement Django n'a pas été modifié."
}

$Runner = "$WslProjectRoot/scripts/gsr/sn_gamestate_runner.py"
$PreflightArgs = @(
    "$LinuxRoot/.venv/bin/python",
    $Runner,
    "--preflight",
    "--sn-gamestate-root",
    $LinuxRoot
)
if ($AllowCpu) {
    $PreflightArgs += "--allow-cpu"
}

Write-Host "Vérification des modèles, bibliothèques et du GPU..."
& wsl.exe @WslArgs @PreflightArgs
if ($LASTEXITCODE -ne 0) {
    throw "Préflight refusé. Rien n'a été activé dans .env. Vérifiez CUDA/WSL ou relancez avec -AllowCpu pour un test CPU explicitement lent."
}

if (-not (Test-Path $EnvPath)) {
    Copy-Item (Join-Path $ProjectRoot ".env.example") $EnvPath
}
$BackupPath = "$EnvPath.gsr-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
Copy-Item $EnvPath $BackupPath

function Set-EnvValue {
    param([string]$Name, [string]$Value)
    $lines = [System.Collections.Generic.List[string]](Get-Content $EnvPath)
    $replacement = "$Name=$Value"
    $found = $false
    for ($index = 0; $index -lt $lines.Count; $index++) {
        if ($lines[$index] -match "^$([regex]::Escape($Name))=") {
            $lines[$index] = $replacement
            $found = $true
        }
    }
    if (-not $found) {
        $lines.Add($replacement)
    }
    [System.IO.File]::WriteAllLines(
        $EnvPath,
        $lines,
        (New-Object System.Text.UTF8Encoding($false))
    )
}

$RunnerCommand = @(
    "wsl.exe"
)
if ($Distro) {
    $RunnerCommand += @("-d", $Distro)
}
$RunnerCommand += @(
    "$LinuxRoot/.venv/bin/python",
    $Runner,
    "--sn-gamestate-root",
    $LinuxRoot
)
if ($AllowCpu) {
    $RunnerCommand += "--allow-cpu"
}
$RunnerJson = ConvertTo-Json -Compress -InputObject $RunnerCommand

Set-EnvValue "ANALYSIS_ATHLETE_ENGINE" "tracklab"
Set-EnvValue "GSR_RUNNER_COMMAND_JSON" $RunnerJson
Set-EnvValue "GSR_TRACKING_FPS" "5.0"
Set-EnvValue "GSR_BALL_BACKEND" "yolo"

Write-Host "TrackLab + sn-gamestate activé."
Write-Host "Sauvegarde de votre ancien .env: $BackupPath"
Write-Host "Redémarrez ensuite l'application avec .\scripts\start_local.ps1"
Write-Host "Commencez uniquement par le test court de 40 s."

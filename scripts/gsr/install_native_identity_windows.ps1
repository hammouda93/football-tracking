param(
    [switch]$SkipEasyOcr
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$EnvPath = Join-Path $ProjectRoot ".env"
$ModelRoot = Join-Path $ProjectRoot "models\native-gsr"
$ReIdFileName = "osnet_x0_25_msmt17.pth"
$ReIdPath = Join-Path $ModelRoot $ReIdFileName
$ReIdUrl = "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth?download=true"
$ReIdSha256 = "cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18"
$TorchReIdCommit = "f8cd150fdf77e8d9e1ed143b7f308c2c609ded50"

if (-not (Test-Path $Python)) {
    throw "Environnement absent: lancez d'abord .\scripts\install_windows.ps1 -WithML"
}

New-Item -ItemType Directory -Path $ModelRoot -Force | Out-Null

Write-Host "Installation de Torchreid OSNet (revision MIT f8cd150)..."
& $Python -m pip install "git+https://github.com/KaiyangZhou/deep-person-reid.git@$TorchReIdCommit"
if ($LASTEXITCODE -ne 0) {
    throw "Installation Torchreid impossible. Le .env n'a pas ete modifie."
}

if (-not $SkipEasyOcr) {
    Write-Host "Installation du moteur OCR chiffres..."
    & $Python -m pip install "easyocr>=1.7,<2"
    if ($LASTEXITCODE -ne 0) {
        throw "Installation EasyOCR impossible. Le .env n'a pas ete modifie."
    }
}

if (-not (Test-Path $ReIdPath)) {
    Write-Host "Telechargement du checkpoint officiel OSNet x0.25 MSMT17..."
    Invoke-WebRequest -Uri $ReIdUrl -OutFile $ReIdPath -UseBasicParsing
}
$ActualHash = (Get-FileHash -Path $ReIdPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualHash -ne $ReIdSha256) {
    throw "SHA256 OSNet incorrect. Supprimez $ReIdPath puis relancez."
}

if (-not (Test-Path $EnvPath)) {
    Copy-Item (Join-Path $ProjectRoot ".env.example") $EnvPath
}
$BackupPath = "$EnvPath.native-gsr-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
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

Set-EnvValue "ANALYSIS_BACKEND" "yolo"
Set-EnvValue "ANALYSIS_ATHLETE_ENGINE" "legacy"
Set-EnvValue "YOLO_PROFILE" "native_gsr"
Set-EnvValue "NATIVE_GSR_REID_BACKEND" "torchreid"
Set-EnvValue "NATIVE_GSR_REID_MODEL_NAME" "osnet_x0_25"
Set-EnvValue "NATIVE_GSR_REID_MODEL_PATH" "models/native-gsr/$ReIdFileName"
Set-EnvValue "NATIVE_GSR_JERSEY_ENGINE" $(if ($SkipEasyOcr) { "auto" } else { "easyocr" })
Set-EnvValue "NATIVE_GSR_JERSEY_DEVICE" "cpu"
Set-EnvValue "NATIVE_GSR_JERSEY_INTERVAL_FRAMES" "12"
Set-EnvValue "NATIVE_GSR_JERSEY_MAX_CROPS_PER_FRAME" "4"
Set-EnvValue "YOLO_BALL_TILED_RECOVERY" "1"
Set-EnvValue "YOLO_BALL_RECOVERY_INTERVAL_FRAMES" "12"
Set-EnvValue "YOLO_BALL_RECOVERY_IMAGE_SIZE" "960"
Set-EnvValue "YOLO_BALL_RECOVERY_OVERLAP" "0.15"
Set-EnvValue "NATIVE_GSR_STRICT_VALIDATION" "1"

Write-Host "Re-ID OSNet installe et verifie."
Write-Host "Sauvegarde de l'ancien .env: $BackupPath"
Write-Host "Le terrain 97 points exige encore un checkpoint ONNX + son schema correspondant."
Write-Host "Redemarrez avec .\scripts\start_local.ps1 puis lancez uniquement le test 40 s."

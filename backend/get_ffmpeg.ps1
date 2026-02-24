# Download FFmpeg for Windows into backend/tools/ffmpeg (no PATH needed)
$tools = Join-Path $PSScriptRoot "tools"
$ffmpegDir = Join-Path $tools "ffmpeg"
$zipPath = Join-Path $tools "ffmpeg.zip"

# BtbN static build (no install, extract and run)
$url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"

Write-Host "Downloading FFmpeg..."
New-Item -ItemType Directory -Force -Path $tools | Out-Null
Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing

Write-Host "Extracting..."
Expand-Archive -Path $zipPath -DestinationPath $tools -Force
Remove-Item $zipPath -Force

# BtbN zip has folder ffmpeg-master-latest-win64-gpl; rename to ffmpeg
$extracted = Get-ChildItem $tools -Directory | Where-Object { $_.Name -like "ffmpeg*" } | Select-Object -First 1
if ($extracted) {
    if (Test-Path $ffmpegDir) { Remove-Item $ffmpegDir -Recurse -Force }
    Rename-Item $extracted.FullName "ffmpeg"
}

$exe = Join-Path $ffmpegDir "bin\ffmpeg.exe"
if (Test-Path $exe) {
    Write-Host "Done. FFmpeg at: $exe"
} else {
    Write-Host "Check that bin\ffmpeg.exe exists in: $ffmpegDir"
}

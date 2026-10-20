# build.ps1 -- build (and optionally run) experiments on Windows
#
#   .\build.ps1                         build all *.cu into bin\
#   .\build.ps1 01_decode_attention.cu  build one file and run it
#   .\build.ps1 -Run                    build all and run all
#
# nvcc needs cl.exe as host compiler, so this pulls in vcvars64.bat first.
# Keep this file ASCII-only (Windows PowerShell 5.1 reads BOM-less .ps1 as ANSI).

param(
    [Parameter(Position = 0)] [string]$Source = "",
    [switch]$Run,
    [string]$Arch = "native"
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

$cands = @(
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
    "D:\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
$vcvars = $cands | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $vcvars) { Write-Host "vcvars64.bat not found (install MSVC Build Tools)" -ForegroundColor Red; exit 1 }

$outDir = Join-Path $root "bin"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$flags = "-O3 -std=c++17 -arch=$Arch -lineinfo -I`"$root`" -Xcompiler /utf-8 -Xcompiler /Zc:preprocessor -lcublas"

if ($Source) { $files = @(Get-Item (Join-Path $root $Source)) ; $Run = $true }
else         { $files = Get-ChildItem (Join-Path $root "*.cu") | Sort-Object Name }

foreach ($f in $files) {
    $exe = Join-Path $outDir ($f.BaseName + ".exe")
    Write-Host "[build] $($f.Name)" -ForegroundColor Cyan
    & cmd.exe /c "call `"$vcvars`" >nul 2>&1 && nvcc $flags `"$($f.FullName)`" -o `"$exe`" 2>&1"
    if ($LASTEXITCODE -ne 0) { Write-Host "[build] FAILED: $($f.Name)" -ForegroundColor Red; exit $LASTEXITCODE }
}

if ($Run) {
    foreach ($f in $files) {
        $exe = Join-Path $outDir ($f.BaseName + ".exe")
        Write-Host ""
        Write-Host "########## $($f.BaseName) ##########" -ForegroundColor Yellow
        & $exe
        if ($LASTEXITCODE -ne 0) { Write-Host "[run] $($f.BaseName) returned $LASTEXITCODE" -ForegroundColor Red }
    }
}

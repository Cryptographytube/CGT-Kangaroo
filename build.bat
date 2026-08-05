@echo off
REM ============================================================================
REM  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
REM  Author: sisujhon
REM
REM  One-click Windows build. Produces build\cryptographytube.exe as a fat
REM  binary that runs on every CUDA GPU from Maxwell through Blackwell
REM  (GTX 750 ... RTX 3050 / 3090 / 4050 / 4090 / 5060 / 5090), with embedded
REM  PTX so newer cards JIT instead of failing.
REM ============================================================================
setlocal enabledelayedexpansion

cd /d "%~dp0"

REM ---- locate the MSVC host compiler ---------------------------------------
if not defined VCINSTALLDIR (
  for %%E in (Enterprise Professional Community BuildTools) do (
    for %%V in (2022 2019) do (
      if exist "C:\Program Files\Microsoft Visual Studio\%%V\%%E\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\%%V\%%E\VC\Auxiliary\Build\vcvars64.bat" >nul
        goto :have_msvc
      )
      if exist "C:\Program Files (x86)\Microsoft Visual Studio\%%V\%%E\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\%%V\%%E\VC\Auxiliary\Build\vcvars64.bat" >nul
        goto :have_msvc
      )
    )
  )
)
:have_msvc

REM ---- locate nvcc ---------------------------------------------------------
set "NVCC=nvcc"
where nvcc >nul 2>&1
if errorlevel 1 (
  for %%V in (v13.1 v13.0 v12.9 v12.8 v12.6 v12.4 v12.3 v12.2 v12.1 v12.0 v11.8) do (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%V\bin\nvcc.exe" (
      set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%V\bin\nvcc.exe"
      goto :have_nvcc
    )
  )
  echo ERROR: nvcc not found. Install the CUDA Toolkit and retry.
  exit /b 1
)
:have_nvcc
echo Using NVCC: %NVCC%

REM ---- multi-architecture fat binary --------------------------------------
REM  Each architecture is probed against this toolkit first, because CUDA
REM  versions differ in what they still accept (CUDA 13 dropped Maxwell,
REM  Pascal and Volta; CUDA 11 does not know Blackwell). Whatever this nvcc
REM  supports goes into the fat binary, so the same script produces a working
REM  all-GPU build on CUDA 11, 12 and 13 alike.
REM   52/61  Maxwell / Pascal  (GTX 750, 10xx)
REM   70/75  Volta / Turing    (RTX 20xx, GTX 16xx)
REM   80/86  Ampere            (RTX 3050 ... 3090)
REM   89     Ada               (RTX 4050 ... 4090)
REM   90     Hopper
REM   100/120 Blackwell        (RTX 5060 ... 5090)
if not exist build mkdir build
echo __global__ void k(){} > build\_probe.cu

set "GENCODE="
set "LASTARCH="
for %%A in (52 61 70 75 80 86 89 90 100 120) do (
  "%NVCC%" -arch=compute_%%A -ptx build\_probe.cu -o build\_probe.ptx >nul 2>&1
  if not errorlevel 1 (
    set "GENCODE=!GENCODE! -gencode arch=compute_%%A,code=sm_%%A"
    set "LASTARCH=%%A"
    echo   arch sm_%%A ... supported
  ) else (
    echo   arch sm_%%A ... not in this toolkit, skipped
  )
)
REM PTX of the newest supported arch = forward compatibility for future GPUs
set "GENCODE=!GENCODE! -gencode arch=compute_!LASTARCH!,code=compute_!LASTARCH!"
del build\_probe.cu build\_probe.ptx >nul 2>&1

if "!GENCODE!"=="" (
  echo ERROR: no usable GPU architecture found for this nvcc.
  exit /b 1
)

taskkill /F /IM cryptographytube.exe >nul 2>&1

echo.
echo Building (this takes a few minutes)...
"%NVCC%" -O3 --use_fast_math !GENCODE! -I . ^
  cgt_field.cpp cgt_ec.cpp cgt_kangaroo.cpp cgt_main.cpp ^
  cgt_gpu.cu ^
  -o build\cryptographytube.exe

if errorlevel 1 (
  echo BUILD FAILED
  exit /b 1
)

echo.
echo BUILD OK  -^>  build\cryptographytube.exe
echo.
echo Run:  build\cryptographytube.exe -gpu 0 -range START:END -pubkey ^<hex^>
endlocal

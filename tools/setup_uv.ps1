param(
    [int]$BuildJobs = 2
)

$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoPath

function Invoke-Uv {
    & uv @args
    if ($LASTEXITCODE -ne 0) { throw "uv failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    Invoke-Uv venv --python 3.10
}
# Source metadata imports torch; bootstrap it before uv can resolve PyTorch3D.
Invoke-Uv pip install --python .venv/Scripts/python.exe 'setuptools>=68,<81' pip wheel 'numpy==1.23.5'
& .venv/Scripts/python.exe tools/fetch_torch_wheel.py
if ($LASTEXITCODE -ne 0) { throw 'Downloading the verified upstream torch wheel failed.' }
Invoke-Uv pip install --python .venv/Scripts/python.exe 'torch==2.3.0+cu121' --no-index --find-links .build-tools/wheels --no-deps
Invoke-Uv pip install --python .venv/Scripts/python.exe 'torchvision==0.18.0+cu121' --index https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match

$vswherePath = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Test-Path -LiteralPath $vswherePath)) { throw 'Visual Studio C++ Build Tools are required.' }
$vsPath = & $vswherePath -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw 'Install the Visual Studio Desktop development with C++ workload.' }
& (Join-Path $vsPath 'Common7/Tools/Launch-VsDevShell.ps1') -SkipAutomaticLocation -Arch amd64 -HostArch amd64
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) { throw 'MSVC cl.exe is unavailable.' }
# VS 2026's STL requires CUDA 13.2. Use a project-local VS 2022 compiler
# instead of overriding unsupported-compiler checks or changing the system.
& .venv/Scripts/python.exe tools/bootstrap_msvc.py
if ($LASTEXITCODE -ne 0) { throw 'Fetching the local CUDA-compatible MSVC toolset failed.' }
$localMsvcPath = (Get-ChildItem -LiteralPath '.build-tools/msvc/VC/Tools/MSVC' -Directory | Where-Object Name -Like '14.38.*' | Select-Object -First 1).FullName
# NVCC checks this VS layout even with --use-local-env. The environment
# itself is already loaded by Launch-VsDevShell above.
$localAuxiliaryPath = Join-Path $repoPath '.build-tools/msvc/VC/Auxiliary/Build'
New-Item -ItemType Directory -Path $localAuxiliaryPath -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $vsPath 'VC/Auxiliary/Build/vcvarsall.bat') -Destination (Join-Path $localAuxiliaryPath 'vcvarsall.bat')
$env:PATH = "$localMsvcPath\bin\Hostx64\x64;$env:PATH"
$env:INCLUDE = "$localMsvcPath\include;$env:INCLUDE"
$env:LIB = "$localMsvcPath\lib\x64;$env:LIB"
$env:VCToolsInstallDir = "$localMsvcPath\"
if (-not $env:CUDA_PATH) { throw 'Install a CUDA 12.x toolkit to compile the GPU extension.' }
$env:CUDA_HOME = $env:CUDA_PATH
$env:CUB_HOME = Join-Path $env:CUDA_PATH 'include'
$env:TORCH_CUDA_ARCH_LIST = (& .venv/Scripts/python.exe -c "import torch; print('.'.join(map(str, torch.cuda.get_device_capability())))").Trim()
if ($LASTEXITCODE -ne 0) { throw 'PyTorch cannot access the CUDA GPU.' }
$env:MAX_JOBS = "$BuildJobs"
$env:DISTUTILS_USE_SDK = '1'
$env:NVCC_FLAGS = ''
# Torch's MSVC build repeats the absolute source path in object filenames.
# Keep this checkout out of uv's long user-cache path to avoid MAX_PATH.
if (-not (Test-Path -LiteralPath '.build-tools/pytorch3d/setup.py')) {
    & git clone --depth 1 --branch v0.7.6 https://github.com/facebookresearch/pytorch3d.git .build-tools/pytorch3d
    if ($LASTEXITCODE -ne 0) { throw 'Fetching the pinned PyTorch3D source failed.' }
}
$pytorch3dHead = (& git -C .build-tools/pytorch3d rev-parse HEAD).Trim()
if ($pytorch3dHead -ne 'f34104cf6ebefacd7b7e07955ee7aaa823e616ac') { throw 'Unexpected PyTorch3D source revision.' }
# CUDA 12.8's tuple headers concatenate uint/ushort into vector type names.
# Pulsar's old Windows macros expand those tokens; equivalent type aliases
# preserve the types without rewriting CUDA header tokens.
$pulsarHeaderPath = Join-Path $repoPath '.build-tools/pytorch3d/pytorch3d/csrc/pulsar/global.h'
$pulsarHeader = [IO.File]::ReadAllText($pulsarHeaderPath)
$macroPattern = '#define uint unsigned int\r?\n#define ushort unsigned short'
$typeAliases = "using uint = unsigned int;`nusing ushort = unsigned short;"
if ([regex]::IsMatch($pulsarHeader, $macroPattern)) {
    $pulsarHeader = [regex]::Replace($pulsarHeader, $macroPattern, $typeAliases)
    [IO.File]::WriteAllText($pulsarHeaderPath, $pulsarHeader, [Text.UTF8Encoding]::new($false))
    Write-Output 'Applied the PyTorch3D Windows/CUDA type-alias compatibility fix.'
} elseif (-not $pulsarHeader.Contains($typeAliases)) {
    throw 'Unexpected PyTorch3D Pulsar header; compatibility patch was not applied.'
}
Invoke-Uv sync --extra compat --no-install-package pytorch3d --no-install-package chumpy
Invoke-Uv sync --extra compat --locked
Invoke-Uv run --no-sync python tools/check_env.py

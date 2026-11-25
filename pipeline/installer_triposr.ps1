param(
    [Parameter(Mandatory)][string]$Uv,
    [Parameter(Mandatory)][string]$Output,
    [Parameter(Mandatory)][string]$Cache
)

$ErrorActionPreference = 'Stop'
$uvExe = (Resolve-Path -LiteralPath $Uv).Path
$runtime = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
$downloads = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Cache)
$sourceRevision = '107cefdc244c39106fa830359024f6a2f1c78871'
$weightRevision = '5b521936b01fbe1890f6f9baed0254ab6351c04a'
$weightUrl = "https://huggingface.co/stabilityai/TripoSR/resolve/$weightRevision"
$pythonVersion = '3.13.15'
$torchVersion = '2.8.0+cpu'
$runtimeSpec = "TripoSR $sourceRevision $weightRevision CPython $pythonVersion torch $torchVersion"
$artifacts = @(
    @{ file = 'triposr.zip'; url = "https://codeload.github.com/VAST-AI-Research/TripoSR/zip/$sourceRevision"
       sha256 = 'a7b3871a1b4377bef96cbfb99bdc952bdd4f5df9f65fea4208d978a343fd1162' }
    @{ file = 'weights/model.ckpt'; url = "$weightUrl/model.ckpt?download=true"
       sha256 = '429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee' }
    @{ file = 'weights/config.yaml'; url = "$weightUrl/config.yaml"
       sha256 = '74ca708ce086bf68e97709ea6b3d91f14717921c04691e84043f0eb8fcc68e62' }
    @{ file = 'weights/dino-config.json'
       url = 'https://huggingface.co/facebook/dino-vitb16/resolve/f205d5d8e640a89a2b8ef0369670dfc37cc07fc2/config.json'
       sha256 = 'b87c0270b97db085fd82cf114a761fd0f62ae7914fbd407c752a2260646b689c' }
)

function Test-Artifact([string]$Path, [string]$Digest) {
    return (Test-Path -LiteralPath $Path -PathType Leaf) -and
        (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $Digest
}

function Get-Artifact($Spec) {
    $file = Join-Path $downloads $Spec.file
    if (Test-Artifact $file $Spec.sha256) { return $file }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $file) | Out-Null
    $partial = "$file.$PID.part"
    try {
        & curl.exe --fail --silent --show-error --location --retry 2 --connect-timeout 30 --output $partial $Spec.url
        if ($LASTEXITCODE -ne 0 -or -not (Test-Artifact $partial $Spec.sha256)) {
            throw "TripoSR artifact verification failed for $($Spec.file)."
        }
        Move-Item -LiteralPath $partial -Destination $file -Force
    } finally {
        if (Test-Path -LiteralPath $partial -PathType Leaf) { Remove-Item -LiteralPath $partial -Force }
    }
    return $file
}

$sourceDirectory = Join-Path $runtime 'source'
$sourceEntry = Join-Path $sourceDirectory "TripoSR-$sourceRevision/tsr/system.py"
$venv = Join-Path $runtime 'venv'
$python = Join-Path $venv 'Scripts/python.exe'
$managedDirectory = Join-Path $runtime 'python'
$managedHome = Join-Path $managedDirectory "cpython-$pythonVersion-windows-x86_64-none"
$venvSettings = Join-Path $venv 'pyvenv.cfg'
$marker = Join-Path $runtime '.open-video-triposr'
$weights = $artifacts | Where-Object { $_.file.StartsWith('weights/') }

if ((Test-Path -LiteralPath $marker -PathType Leaf) -and
    (Get-Content -LiteralPath $marker -Raw).Trim() -eq $runtimeSpec -and
    (Test-Path -LiteralPath $sourceEntry -PathType Leaf) -and
    (Test-Path -LiteralPath (Join-Path $managedHome 'python.exe') -PathType Leaf) -and
    (Test-Path -LiteralPath $python -PathType Leaf) -and
    (Test-Path -LiteralPath $venvSettings -PathType Leaf)) {
    $homeLine = Get-Content -LiteralPath $venvSettings | Where-Object { $_.StartsWith('home = ') }
    $invalidWeights = $weights | Where-Object { -not (Test-Artifact (Join-Path $runtime $_.file) $_.sha256) }
    if ($homeLine -eq "home = $managedHome" -and -not $invalidWeights) {
        Write-Output "TripoSR CPU ready: $runtime"
        return
    }
}

if (Test-Path -LiteralPath $marker -PathType Leaf) { Remove-Item -LiteralPath $marker -Force }
New-Item -ItemType Directory -Force -Path $sourceDirectory,(Join-Path $runtime 'weights') | Out-Null
$archive = Get-Artifact $artifacts[0]
& tar.exe -xf $archive -C $sourceDirectory
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $sourceEntry -PathType Leaf)) {
    throw 'Could not extract the pinned TripoSR source.'
}
foreach ($spec in $weights) {
    $destination = Join-Path $runtime $spec.file
    if (-not (Test-Artifact $destination $spec.sha256)) {
        Copy-Item -LiteralPath (Get-Artifact $spec) -Destination $destination -Force
    }
}

$requirements = Join-Path $runtime 'requirements-cpu.txt'
[IO.File]::WriteAllText($requirements, @"
certifi==2026.11.20
charset-normalizer==3.5.3
colorama==0.4.6
einops==0.8.2
filelock==4.2.4
fsspec==2026.9.0
huggingface-hub==0.36.2
idna==3.20
imageio==2.38.2
jinja2==3.1.6
lazy-loader==0.6
markupsafe==3.0.3
mpmath==1.3.0
networkx==3.7
numpy==2.5.5
omegaconf==2.4.0
packaging==26.4
pillow==13.0.0
pyyaml==6.0.4
regex==2026.11.23
requests==2.34.2
safetensors==0.8.0
scikit-image==0.26.0
scipy==1.18.1
setuptools==78.1.0
sympy==1.14.0
tifffile==2026.11.16
tokenizers==0.21.4
torch==$torchVersion
tqdm==4.70.1
transformers==4.51.3
trimesh==5.2.2
typing-extensions==4.16.0
urllib3==2.8.0
"@)

$previousInstallDirectory = $env:UV_PYTHON_INSTALL_DIR
$previousCacheDirectory = $env:UV_CACHE_DIR
try {
    $env:UV_PYTHON_INSTALL_DIR = $managedDirectory
    $env:UV_CACHE_DIR = Join-Path $downloads 'uv-cache'
    & $uvExe python install $pythonVersion --no-bin --no-config
    if ($LASTEXITCODE -ne 0) { throw 'Could not install portable Python for TripoSR.' }
    & $uvExe venv $venv --python $pythonVersion --managed-python --allow-existing --no-config
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the TripoSR environment.' }
    # PyPI supplies the pinned dependencies, with the CPU PyTorch index supplying the +cpu build.
    & $uvExe pip sync $requirements --python $python --index 'https://pypi.org/simple' `
        --default-index 'https://download.pytorch.org/whl/cpu' --index-strategy unsafe-first-match --no-config
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the pinned TripoSR dependencies.' }
} finally {
    $env:UV_PYTHON_INSTALL_DIR = $previousInstallDirectory
    $env:UV_CACHE_DIR = $previousCacheDirectory
}

& $python -I -c "import imageio, omegaconf, scipy, skimage, torch, transformers, trimesh; assert torch.__version__ == '$torchVersion'"
if ($LASTEXITCODE -ne 0) { throw 'TripoSR runtime dependency check failed.' }
[IO.File]::WriteAllText($marker, $runtimeSpec + [Environment]::NewLine)
Write-Output "TripoSR CPU ready: $runtime"

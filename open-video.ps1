param(
    [ValidateSet('initialiser', 'produire', 'auditer', 'materiau', 'preparer-pov', 'preparer-studio',
        'initialiser-reconstruction', 'reconstruire-decor', 'importer-objet', 'initialiser-photo', 'reconstruire-objet',
        'texturer-objet')]
    [string]$Action = 'produire',
    [string]$Scene,
    [string]$Stone,
    [string]$Assets,
    [string]$Recette,
    [ValidateSet('reference', 'rebonds', 'rapide')][string]$Profil = 'reference',
    [ValidateSet('reference', 'fond-doux')][string]$Lumiere = 'reference',
    [string]$Sortie,
    [string]$Source,
    [string]$Captures,
    [ValidateSet('capture', 'studio')][string]$Finition = 'capture',
    [string]$Runtime,
    [string]$Collection,
    [double]$DimensionM,
    [ValidateSet('apercu', 'standard', 'detail')][string]$Niveau = 'standard',
    [ValidateSet('exhaustif', 'sequentiel')][string]$Correspondance = 'exhaustif',
    [switch]$CameraUnique,
    [switch]$Reprendre,
    [int[]]$Images
)

$ErrorActionPreference = 'Stop'
if ($Reprendre -and $Action -ne 'produire') { throw '-Reprendre requires -Action produire.' }
$pilotRequested = $PSBoundParameters.ContainsKey('Images')
if ($pilotRequested -and ($Action -ne 'produire' -or $Images.Count -eq 0)) {
    throw '-Images requires -Action produire and at least one Blender frame number.'
}
$root = $PSScriptRoot
$tools = Join-Path $root 'outils'
$cache = Join-Path $root 'cache'
$lockedTools = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $root 'outils.lock.json') | ConvertFrom-Json

function Test-Tool($Spec) {
    $folder = Join-Path $tools $Spec.folder
    $marker = Join-Path $folder '.open-video-sha256'
    if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) { return $false }
    if ((Get-Content -Raw -LiteralPath $marker).Trim() -ne $Spec.sha256) { return $false }
    foreach ($file in $Spec.files) {
        if (-not (Test-Path -LiteralPath (Join-Path $folder $file) -PathType Leaf)) { return $false }
    }
    return $true
}

function Install-LocalTools([string[]]$Names) {
    $archives = Join-Path $cache 'archives'
    New-Item -ItemType Directory -Force -Path $tools, $archives | Out-Null
    foreach ($name in $Names) {
        $spec = $lockedTools.$name
        if (Test-Tool $spec) { continue }
        $archive = Join-Path $archives $spec.archive
        if (-not (Test-Path -LiteralPath $archive -PathType Leaf) -or
            (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $spec.sha256) {
            $partial = "$archive.part"
            curl.exe --fail --silent --show-error --location --retry 2 --max-time 240 --output $partial $spec.url
            if ($LASTEXITCODE -ne 0 -or (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $spec.sha256) {
                throw "Archive verification failed for $name."
            }
            Move-Item -LiteralPath $partial -Destination $archive -Force
        }
        $destination = $tools
        if ($spec.extract_into_folder) {
            $destination = Join-Path $tools $spec.folder
            New-Item -ItemType Directory -Force -Path $destination | Out-Null
        }
        tar.exe -xf $archive -C $destination
        if ($LASTEXITCODE -ne 0) { throw "Extraction failed for $name." }
        $marker = Join-Path (Join-Path $tools $spec.folder) '.open-video-sha256'
        [IO.File]::WriteAllText($marker, $spec.sha256)
        if (-not (Test-Tool $spec)) { throw "Incomplete runtime for $name." }
        Write-Output "$name $($spec.version) ready."
    }
}

if ($Action -eq 'initialiser') {
    Install-LocalTools @('blender', 'materialcraft', 'ffmpeg')
    return
}
if ($Action -eq 'initialiser-reconstruction') {
    Install-LocalTools @('blender', 'colmap', 'brush')
    return
}
if (-not $Runtime) { $Runtime = Join-Path $tools 'triposr' }
if ($Action -eq 'initialiser-photo') {
    Install-LocalTools @('uv')
    $uv = Join-Path (Join-Path $tools $lockedTools.uv.folder) $lockedTools.uv.files[0]
    & (Join-Path $root 'pipeline/installer_triposr.ps1') -Uv $uv -Output $Runtime -Cache (Join-Path $cache 'archives/triposr')
    return
}

if ($Action -in 'produire', 'auditer', 'preparer-studio' -and -not $Scene) {
    throw "-Scene is required for $Action. Provide a local Blender file."
}
if ($Action -eq 'preparer-pov' -and (-not $Assets -or -not $Stone)) {
    throw '-Assets and -Stone are required for preparer-pov.'
}

$requiredTools = switch ($Action) {
    'reconstruire-decor' { 'blender'; 'colmap'; 'brush' }
    'reconstruire-objet' {}
    'texturer-objet' {}
    'materiau' { 'blender'; 'materialcraft' }
    { $_ -in 'produire', 'auditer' } { 'blender'; 'ffmpeg' }
    default { 'blender' }
}
foreach ($name in $requiredTools) {
    if (-not (Test-Tool $lockedTools.$name)) {
        $setup = if ($Action -eq 'reconstruire-decor') { 'initialiser-reconstruction' } else { 'initialiser' }
        throw "Missing $name. Run .\open-video.ps1 -Action $setup."
    }
}

$blenderPath = Join-Path $tools $lockedTools.blender.folder
$blender = Join-Path $blenderPath $lockedTools.blender.files[0]
$python = Join-Path $blenderPath $lockedTools.blender.files[1]
$ffmpeg = Join-Path (Join-Path $tools $lockedTools.ffmpeg.folder) $lockedTools.ffmpeg.files[0]
$materialcraft = Join-Path (Join-Path $tools $lockedTools.materialcraft.folder) $lockedTools.materialcraft.files[0]
$label = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$setting = if ($Action -in 'reconstruire-decor', 'reconstruire-objet') { $Niveau } else { $Profil }
if (-not $Sortie) { $Sortie = Join-Path $cache "rendus/$Action-$setting-$label-$PID" }
$output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Sortie)
New-Item -ItemType Directory -Force -Path $output | Out-Null
function Invoke-Logged([string]$Executable, [string[]]$Arguments, [string]$Log) {
    $previousConfig = $env:BLENDER_USER_CONFIG
    try {
        $env:BLENDER_USER_CONFIG = Join-Path $output 'blender-config'
        & $Executable @Arguments > $Log 2>&1
        $code = $LASTEXITCODE
    } finally {
        $env:BLENDER_USER_CONFIG = $previousConfig
    }
    if ($code -ne 0) {
        Get-Content -LiteralPath $Log -Tail 25
        throw "Command failed. See $Log"
    }
    Get-Content -LiteralPath $Log | Where-Object { $_ -match '^Open-Video |^\d+ frames checked' } | Select-Object -Last 4
}

$blenderArgs = @('--background', '--factory-startup', '--disable-autoexec')
if ($Action -eq 'reconstruire-objet') {
    if (-not $Source) { throw '-Source must be a PNG with a transparent background.' }
    $sourcePath = (Resolve-Path -LiteralPath $Source).Path
    $photoPython = Join-Path $Runtime 'venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $photoPython -PathType Leaf)) { throw 'Run .\open-video.ps1 -Action initialiser-photo first.' }
    & $photoPython -B (Join-Path $root 'pipeline/reconstruire_objet.py') --photo $sourcePath --output $output --runtime $Runtime --profile $Niveau
    if ($LASTEXITCODE -ne 0) { throw "Object reconstruction failed. Check $output." }
    Write-Output "Object: $(Join-Path $output 'objet.glb')"
    return
}

if ($Action -eq 'texturer-objet') {
    if (-not $Source -or -not $Captures) { throw '-Source (GLB) and -Captures (manifest JSON) are required.' }
    $sourcePath = (Resolve-Path -LiteralPath $Source).Path
    $manifestPath = (Resolve-Path -LiteralPath $Captures).Path
    $photoPython = Join-Path $Runtime 'venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $photoPython -PathType Leaf)) { throw 'Run .\open-video.ps1 -Action initialiser-photo first.' }
    & $photoPython -B (Join-Path $root 'pipeline/texturer_multivue.py') --asset $sourcePath --captures $manifestPath --output $output --lighting $Finition
    if ($LASTEXITCODE -ne 0) { throw "Texture projection failed. Check $output." }
    Write-Output "Textured object: $(Join-Path $output 'objet.glb')"
    return
}

if ($Action -in 'importer-objet', 'reconstruire-decor') {
    if (-not $Source) { throw "-Source is required for $Action." }
    $sourcePath = (Resolve-Path -LiteralPath $Source).Path
    if ($Action -eq 'importer-objet') {
        if (-not $Collection) { $Collection = 'OpenVideo_Objet' }
        $asset = Join-Path $output 'objet.blend'
        $importArgs = $blenderArgs + @('--python-exit-code', '1', '--python',
            (Join-Path $root 'pipeline/importer_objet.py'), '--', '--asset', $sourcePath, '--collection', $Collection,
            '--output', $asset, '--report', (Join-Path $output 'import.json'))
        if ($PSBoundParameters.ContainsKey('DimensionM')) {
            $importArgs += @('--dimension-m', $DimensionM.ToString([Globalization.CultureInfo]::InvariantCulture))
        }
        Invoke-Logged $blender $importArgs (Join-Path $output 'import.log')
        Write-Output "Asset: $asset"
        return
    }
    $colmap = Join-Path (Join-Path $tools $lockedTools.colmap.folder) $lockedTools.colmap.files[0]
    $brush = Join-Path (Join-Path $tools $lockedTools.brush.folder) $lockedTools.brush.files[0]
    $reconstructionArgs = @('-B', (Join-Path $root 'pipeline/reconstruire_decor.py'), '--photos', $sourcePath,
        '--colmap', $colmap, '--brush', $brush, '--output', $output, '--profile', $Niveau,
        '--matching', $Correspondance)
    if ($CameraUnique) { $reconstructionArgs += '--single-camera' }
    & $python @reconstructionArgs
    if ($LASTEXITCODE -ne 0) { throw "Reconstruction failed. Check $output." }
    Write-Output "Decor: $(Join-Path $output 'scene.ply')"
    return
}

if ($Action -eq 'materiau') {
    Invoke-Logged $python @('-B', (Join-Path $root 'essais/materiaux/tester_transfert.py'), '--materialcraft', $materialcraft,
        '--blender', $blender, '--output', $output) (Join-Path $output 'materiau.log')
    Write-Output "Material library: $(Join-Path $output 'stone.blend')"
    return
}

if (-not $Recette) {
    $defaultRecipe = if ($Action -eq 'preparer-studio') { 'essais/studio/recette.json' } else { 'essais/pov/recette.json' }
    $Recette = Join-Path $root $defaultRecipe
}
$recipePath = (Resolve-Path -LiteralPath $Recette).Path
if ($Action -eq 'preparer-pov') {
    Invoke-Logged $blender ($blenderArgs + @('--python-exit-code', '1', '--python', (Join-Path $root 'essais/pov/rendre.py'), '--',
        'prepare', '--recipe', $recipePath, '--assets', (Resolve-Path -LiteralPath $Assets).Path, '--stone', (Resolve-Path -LiteralPath $Stone).Path,
        '--scene', (Join-Path $output 'scene.blend'), '--output', $output, '--profile', $Profil, '--lighting', $Lumiere)) (Join-Path $output 'preparation.log')
    Write-Output "Prepared scene: $(Join-Path $output 'scene.blend')"
    return
}

$scenePath = (Resolve-Path -LiteralPath $Scene).Path
if ($Action -eq 'preparer-studio') {
    $studioArgs = $blenderArgs + @('--python-exit-code', '1', '--python', (Join-Path $root 'pipeline/studio.py'), '--',
        '--asset', $scenePath, '--recipe', $recipePath, '--output', $output,
        '--profile', $Profil, '--lighting', $Lumiere)
    if ($Stone) { $studioArgs += @('--stone', (Resolve-Path -LiteralPath $Stone).Path) }
    if ($Collection) { $studioArgs += @('--collection', $Collection) }
    if ($Captures) { $studioArgs += @('--captures', (Resolve-Path -LiteralPath $Captures).Path) }
    Invoke-Logged $blender $studioArgs (Join-Path $output 'preparation.log')
    Write-Output "Prepared scene: $(Join-Path $output 'scene.blend')"
    return
}
$mode = if ($Action -eq 'auditer') { 'audit' } else { 'frames' }
$renderArgs = $blenderArgs + @($scenePath, '--python-exit-code', '1', '--python', (Join-Path $root 'pipeline/rendu.py'), '--',
    $mode, '--recipe', $recipePath, '--output', $output, '--profile', $Profil, '--lighting', $Lumiere)
if ($Reprendre) { $renderArgs += '--resume' }
if ($pilotRequested) { $renderArgs += @('--images') + @($Images | ForEach-Object { $_.ToString() }) }
$productionLock = $null
try {
    if ($Action -eq 'produire') {
        $productionLock = [IO.File]::Open((Join-Path $output '.open-video-production.lock'),
            [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    }
    Invoke-Logged $blender $renderArgs (Join-Path $output 'rendu.log')
    if ($pilotRequested) {
        Write-Output "Pilot frames: $output"
        return
    }
    if ($Action -eq 'auditer') {
        & $python -B (Join-Path $root 'pipeline/analyser.py') --ffmpeg $ffmpeg --directory $output
    } else {
        & $python -B (Join-Path $root 'pipeline/assembler.py') --ffmpeg $ffmpeg --frames $output --recipe $recipePath --output $output
    }
    if ($LASTEXITCODE -ne 0) { throw 'Output validation failed.' }
    Write-Output "Results: $output"
} finally {
    if ($productionLock) { $productionLock.Dispose() }
}

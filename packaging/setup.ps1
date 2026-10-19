param(
    [string]$InstallDirectory = (Join-Path $PSScriptRoot 'ConsultASR'),
    [switch]$SkipModelDownload
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Add-Type -AssemblyName System.IO.Compression.FileSystem
if (-not [Environment]::Is64BitOperatingSystem) { throw 'Windows x64 is required.' }
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'release-manifest.json') -Raw | ConvertFrom-Json
$target = [IO.Path]::GetFullPath($InstallDirectory)
$cache = Join-Path $PSScriptRoot '.downloads'
$staging = $null
function Get-Sha256($Path) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try { return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
function Test-Asset($Path, $Asset) {
    return ((Test-Path -LiteralPath $Path -PathType Leaf) -and
        (Get-Item -LiteralPath $Path).Length -eq [long]$Asset.size -and
        (Get-Sha256 $Path) -eq $Asset.sha256)
}
function Get-Asset($Asset) {
    if ($Asset.name -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid asset name.' }
    $local = Join-Path $PSScriptRoot $Asset.name
    if (Test-Asset $local $Asset) { return $local }
    $destination = Join-Path $cache $Asset.name
    if (Test-Asset $destination $Asset) { return $destination }
    $partial = "$destination.partial"
    if (Test-Asset $partial $Asset) {
        Move-Item -LiteralPath $partial -Destination $destination -Force
        return $destination
    }
    if ((Test-Path -LiteralPath $partial) -and (Get-Item -LiteralPath $partial).Length -ge [long]$Asset.size) {
        Remove-Item -LiteralPath $partial
    }
    $url = "$($manifest.base_url)/$($Asset.name)"
    Write-Host "Downloading $($Asset.name) ... (rerun setup.bat to resume)"
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        & curl.exe --silent --show-error --fail --location --retry 4 --retry-delay 3 --connect-timeout 30 --continue-at - --output $partial $url
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $($Asset.name). Rerun setup.bat to resume." }
    } else {
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $partial
    }
    if (-not (Test-Asset $partial $Asset)) {
        Remove-Item -LiteralPath $partial -ErrorAction SilentlyContinue
        throw "SHA-256 verification failed: $($Asset.name). Rerun setup to download again."
    }
    Move-Item -LiteralPath $partial -Destination $destination -Force
    return $destination
}
function Expand-VerifiedArchive($Archive, $Destination) {
    $parts = @($Archive.parts | ForEach-Object { Get-Asset $_ })
    $joined = Join-Path $cache $Archive.name
    if ($parts.Count -eq 1) {
        $joined = $parts[0]
    } else {
        $output = [IO.File]::Create($joined)
        try {
            foreach ($part in $parts) {
                $inputFile = [IO.File]::OpenRead($part)
                try { $inputFile.CopyTo($output) } finally { $inputFile.Dispose() }
            }
        } finally { $output.Dispose() }
    }
    if ((Get-Sha256 $joined) -ne $Archive.sha256) {
        throw "Archive checksum mismatch: $($Archive.name)"
    }
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    Write-Host "Extracting $($Archive.name) ..."
    [IO.Compression.ZipFile]::ExtractToDirectory($joined, $Destination)
    if ($parts.Count -gt 1) { Remove-Item -LiteralPath $joined }
}
try {
    $marker = Join-Path $target '.installed-version'
    if (Test-Path -LiteralPath $target) {
        if (-not (Test-Path -LiteralPath $marker) -or (Get-Content -LiteralPath $marker -Raw).Trim() -ne $manifest.version) {
            throw 'The destination already exists. Choose a NEW folder with -InstallDirectory. Existing installations are preserved.'
        }
        Write-Host 'Installation already present; checking resources.'
    } else {
        New-Item -ItemType Directory -Path $cache -Force | Out-Null
        $parent = Split-Path -Parent $target
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
        $drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($target))
        if ($drive.AvailableFreeSpace -lt 30GB) { throw 'At least 30 GiB free space is required on the installation drive.' }
        $staging = Join-Path $parent ('.consult-asr-install-' + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $staging | Out-Null
        foreach ($archive in $manifest.archives) {
            $destination = [IO.Path]::GetFullPath((Join-Path $staging $archive.destination))
            if (-not $destination.StartsWith($staging + '\', [StringComparison]::OrdinalIgnoreCase) -and $destination -ne $staging) { throw 'Invalid archive destination.' }
            Expand-VerifiedArchive $archive $destination
        }
        # conda-unpack must run at the FINAL path; a failed run remains retryable.
        Set-Content -LiteralPath (Join-Path $staging '.needs-unpack') -Value $manifest.version -Encoding ASCII
        Set-Content -LiteralPath (Join-Path $staging '.installed-version') -Value $manifest.version -Encoding ASCII
        Move-Item -LiteralPath $staging -Destination $target
        $staging = $null
    }
    $python = Join-Path $target '.runtime\python\python.exe'
    $env:PYTHONNOUSERSITE = '1'
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    $env:CONDA_PREFIX = Split-Path -Parent $python
    $env:CONSULT_ASR_FFMPEG = Join-Path $target '.runtime\ffmpeg\ffmpeg.exe'
    $env:PATH = "$target\.runtime\ffmpeg;$env:CONDA_PREFIX;$env:CONDA_PREFIX\Library\bin;$env:CONDA_PREFIX\Lib\site-packages\torch\lib;$env:PATH"
    $needsUnpack = Join-Path $target '.needs-unpack'
    if (Test-Path -LiteralPath $needsUnpack) {
        Write-Host 'Relocating the bundled Python runtime ...'
        & $python -X utf8 -B (Join-Path $target '.runtime\python\Scripts\conda-unpack-script.py')
        if ($LASTEXITCODE -ne 0) { throw 'Runtime relocation failed. Rerun setup.bat.' }
        Remove-Item -LiteralPath $needsUnpack
    }
    Push-Location -LiteralPath $target
    try {
        & $python -X utf8 -B tools\check_deployment.py
        if ($LASTEXITCODE -ne 0) { throw 'Runtime check failed. Run diagnose.bat for details.' }
        & $python -X utf8 -B tools\prepare_models.py --verify
        if ($LASTEXITCODE -ne 0 -and -not $SkipModelDownload) {
            Write-Host 'First use: allow access on BOTH Hugging Face pages using your own account:'
            Write-Host 'https://huggingface.co/pyannote/speaker-diarization-3.1'
            Write-Host 'https://huggingface.co/pyannote/segmentation-3.0'
            Write-Host 'Create a read token at https://huggingface.co/settings/tokens . Input below is hidden.'
            Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
            Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
            & $python -X utf8 -B tools\prepare_models.py --diarization --prompt-token
            if ($LASTEXITCODE -ne 0) { throw 'Model preparation incomplete. After obtaining access, rerun setup.bat.' }
            & $python -X utf8 -B tools\prepare_models.py --verify
            if ($LASTEXITCODE -ne 0) { throw 'Model verification failed.' }
        }
        if ($SkipModelDownload) { Write-Host 'Model download was skipped; transcription requires all mandatory models.' }
    } finally { Pop-Location }
    Write-Host "Deployment finished. Start: $target\咨询录音转写.exe"
    Write-Host 'After checking the app, you may delete the installer .downloads folder to reclaim space.'
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    if ($staging -and (Test-Path -LiteralPath $staging)) {
        # Only remove this run's verified, freshly-created staging directory.
        $resolved = (Resolve-Path -LiteralPath $staging).Path
        if ((Split-Path -Parent $resolved) -eq (Split-Path -Parent $target) -and (Split-Path -Leaf $resolved) -like '.consult-asr-install-*') {
            Remove-Item -LiteralPath $resolved -Recurse -Force
        }
    }
    exit 1
}

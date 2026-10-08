param(
    [string]$OutputDirectory = 'F:\桌面\程序\APP',
    [string]$PythonExe = 'G:\conda\.conda\envs\consult_asr_fast\python.exe',
    [switch]$Replace
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$pythonPath = (Resolve-Path -LiteralPath $PythonExe).Path
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) { throw 'Windows .NET Framework C# compiler is missing.' }
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$outputRoot = (Resolve-Path -LiteralPath $OutputDirectory).Path
$target = Join-Path $outputRoot '咨询录音转写.exe'
if ((Test-Path -LiteralPath $target) -and -not $Replace) { throw 'The output EXE already exists. Use -Replace for a validated update.' }
$stagedExe = Join-Path $outputRoot ('.consult-asr-stage-' + [guid]::NewGuid().ToString('N') + '.exe')
$backupExe = Join-Path $outputRoot ('.consult-asr-previous-' + [guid]::NewGuid().ToString('N') + '.bak')
$published = $false
$temporaryConfig = Join-Path $outputRoot ('.launcher-config-' + [guid]::NewGuid().ToString('N') + '.txt')
try {
    Push-Location -LiteralPath $projectRoot
    try {
        $ffmpegPath = & $pythonPath -X utf8 -B -c 'import config; print(config.FFMPEG_EXECUTABLE or "")'
        if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve the current runtime configuration.' }
    } finally { Pop-Location }
    [IO.File]::WriteAllText($temporaryConfig, "$projectRoot`n$pythonPath`n$ffmpegPath`n", [Text.UTF8Encoding]::new($false))
    & $compiler /nologo /target:winexe /platform:x64 /optimize+ /codepage:65001 `
        /reference:System.Windows.Forms.dll `
        "/win32icon:$projectRoot\assets\consult_asr.ico" `
        "/win32manifest:$PSScriptRoot\launcher.manifest" `
        "/resource:$temporaryConfig,ConsultASR.Config" `
        "/resource:$PSScriptRoot\launcher_probe.py,ConsultASR.Probe" `
        "/out:$stagedExe" "$PSScriptRoot\Launcher.cs"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $stagedExe -PathType Leaf)) { throw 'EXE build failed.' }
    foreach ($checkMode in @('--check', '--check-native')) {
        $process = Start-Process -FilePath $stagedExe -ArgumentList $checkMode -PassThru -WindowStyle Hidden
        if (-not $process.WaitForExit(60000)) { $process.Kill(); throw 'Launcher check timed out.' }
        if ($process.ExitCode -ne 0) { throw "Launcher $checkMode failed with exit code $($process.ExitCode)." }
    }
    if (Test-Path -LiteralPath $target) {
        [IO.File]::Replace($stagedExe, $target, $backupExe)
    } else {
        Move-Item -LiteralPath $stagedExe -Destination $target
    }
    $published = $true
    Write-Output "Built and checked: $target"
} finally {
    if (Test-Path -LiteralPath $temporaryConfig) { Remove-Item -LiteralPath $temporaryConfig }
    if (Test-Path -LiteralPath $stagedExe) { Remove-Item -LiteralPath $stagedExe }
    if ($published -and (Test-Path -LiteralPath $backupExe)) { Remove-Item -LiteralPath $backupExe }
}

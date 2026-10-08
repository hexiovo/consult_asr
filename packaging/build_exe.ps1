param(
    [string]$OutputDirectory = 'F:\桌面\程序\APP',
    [string]$PythonExe = 'G:\conda\.conda\envs\consult_asr_fast\python.exe'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$pythonPath = (Resolve-Path -LiteralPath $PythonExe).Path
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) { throw 'Windows .NET Framework C# compiler is missing.' }
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$outputRoot = (Resolve-Path -LiteralPath $OutputDirectory).Path
$target = Join-Path $outputRoot '咨询录音转写.exe'
if (Test-Path -LiteralPath $target) { throw 'The output EXE already exists. Choose a new output directory or move the previous EXE first.' }
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
        "/out:$target" "$PSScriptRoot\Launcher.cs"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $target -PathType Leaf)) { throw 'EXE build failed.' }
    $process = Start-Process -FilePath $target -ArgumentList '--check' -PassThru -WindowStyle Hidden
    if (-not $process.WaitForExit(60000)) { $process.Kill(); throw 'Launcher check timed out.' }
    if ($process.ExitCode -ne 0) { throw "Launcher check failed with exit code $($process.ExitCode)." }
    Write-Output "Built and checked: $target"
} finally {
    if (Test-Path -LiteralPath $temporaryConfig) { Remove-Item -LiteralPath $temporaryConfig }
}

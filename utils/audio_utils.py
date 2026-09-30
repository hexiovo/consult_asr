from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
from threading import Event

from config import FFMPEG_EXECUTABLE, SUPPORTED_AUDIO_EXTENSIONS


class ProcessingCancelled(Exception):
    """Raised when cancellation is requested at a safe processing boundary."""


def is_audio_file(path: str | os.PathLike[str]) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def collect_audio_files(input_path: str | os.PathLike[str]) -> list[Path]:
    path = Path(input_path).expanduser()
    if path.is_file():
        return [path.resolve()] if is_audio_file(path) else []
    if not path.is_dir():
        return []
    files = [
        Path(root) / name
        for root, dirs, names in os.walk(path, followlinks=False)
        for name in sorted(names)
        if is_audio_file(Path(root) / name)
    ]
    return sorted((item.resolve() for item in files), key=lambda item: str(item).casefold())


def normalize_audio(
    source_path: str | os.PathLike[str],
    task_dir: str | os.PathLike[str],
    cancel_event: Event | None = None,
) -> Path:
    source = Path(source_path).expanduser().resolve()
    target_dir = Path(task_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError("所选音频文件不存在。")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "normalized-16k-mono.wav"
    if target.exists():
        raise FileExistsError("本任务临时音频已存在，拒绝覆盖。")
    if not FFMPEG_EXECUTABLE:
        raise RuntimeError(
            "未找到 FFmpeg。请将其所在环境激活，或设置 CONSULT_ASR_FFMPEG 指向 ffmpeg 可执行文件。"
        )

    command = [
        FFMPEG_EXECUTABLE, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
        "-i", str(source), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(target),
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    child_env = os.environ.copy()
    ffmpeg_dir = str(Path(FFMPEG_EXECUTABLE).parent)
    child_env["PATH"] = ffmpeg_dir + os.pathsep + child_env.get("PATH", "")
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
            env=child_env,
        )
    except OSError as exc:
        raise RuntimeError("无法启动已配置的 FFmpeg，请检查 CONSULT_ASR_FFMPEG 路径。") from exc
    try:
        try:
            while process.poll() is None:
                if cancel_event and cancel_event.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise ProcessingCancelled("已在音频规范化安全边界取消。")
                time.sleep(0.1)
            stderr = process.stderr.read() if process.stderr else b""
            if process.returncode != 0:
                # Do not surface ffmpeg's command line or private input paths.
                detail = stderr.decode("utf-8", errors="replace").strip()
                reason = "；".join(detail.splitlines()[-2:]) if detail else "未提供详细错误"
                raise RuntimeError(f"FFmpeg音频规范化失败（退出码 {process.returncode}）：{reason[:300]}")
            if not target.is_file() or target.stat().st_size == 0:
                raise RuntimeError("FFmpeg未生成有效的规范化音频。")
            return target
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    finally:
        if process.stderr:
            process.stderr.close()

from __future__ import annotations

import os
from pathlib import Path
import json
import shutil
import subprocess
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(
    os.environ.get("CONSULT_ASR_MODEL_DIR", PROJECT_ROOT / "local_models")
).expanduser()
ASR_MODEL_DIR = MODEL_ROOT / "faster-whisper-large-v3"
DIARIZATION_MODEL_DIR = MODEL_ROOT / "pyannote" / "speaker-diarization-3.1"
OPENAI_WHISPER_MODEL_DIR = MODEL_ROOT / "openai-whisper"
TONE_MODEL_DIR = PROJECT_ROOT / "models" / "emotion2vec_plus_base"
TONE_RUNTIME_DIR = PROJECT_ROOT / ".runtime" / "tone"
LOG_DIR = PROJECT_ROOT / "log"
TONE_DEFAULT_ENABLED = os.environ.get("CONSULT_ASR_TONE_ENABLED", "1").strip().casefold() not in {
    "0",
    "false",
    "no",
    "off",
}
WHISPER_MODEL = "large-v3"
NUM_SPEAKERS = 2
OFFLINE_INFERENCE = True
TEMP_ROOT = Path(os.environ.get("CONSULT_ASR_TEMP_DIR", tempfile.gettempdir()))
OUTPUT_DIR = Path.home() / "Documents" / "咨询转写"
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma", ".mp4"}


def _resolve_ffmpeg() -> str | None:
    explicit = os.environ.get("CONSULT_ASR_FFMPEG")
    if explicit and Path(explicit).is_file():
        return str(Path(explicit).resolve())
    bundled = PROJECT_ROOT / ".runtime" / "ffmpeg" / "ffmpeg.exe"
    if bundled.is_file():
        return str(bundled.resolve())
    on_path = shutil.which("ffmpeg")
    if on_path:
        return str(Path(on_path).resolve())

    active_prefix = os.environ.get("CONDA_PREFIX")
    prefixes = [Path(active_prefix)] if active_prefix else []
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).is_file():
        try:
            listing = subprocess.run(
                [conda_exe, "env", "list", "--json"],
                check=False,
                capture_output=True,
                text=True,
                timeout=4,
                env={**os.environ, "CONDA_NO_PLUGINS": "true"},
            )
            if listing.returncode == 0:
                for prefix in json.loads(listing.stdout).get("envs", []):
                    if Path(prefix).name.casefold() in {"consult_asr", "consult_asr_fast"}:
                        prefixes.append(Path(prefix))
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass

    for prefix in prefixes:
        candidate = prefix / "Library" / "bin" / "ffmpeg.exe"
        if candidate.is_file():
            return str(candidate.resolve())
    return None


FFMPEG_EXECUTABLE = _resolve_ffmpeg()

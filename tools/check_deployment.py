"""Offline runtime diagnostic. Does not read consultation files or run inference."""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    failures = []
    if sys.version_info[:2] != (3, 10):
        failures.append("The packaged runtime requires Python 3.10.")
    for name in ("PySide6.QtWidgets", "PySide6.QtMultimedia", "torch", "torchaudio",
                 "faster_whisper", "pyannote.audio", "docx", "soundfile"):
        try:
            importlib.import_module(name)
            print(f"OK import {name}", flush=True)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    from config import FFMPEG_EXECUTABLE, TONE_MODEL_DIR, TONE_RUNTIME_DIR
    try:
        if not FFMPEG_EXECUTABLE:
            raise RuntimeError("Bundled FFmpeg is missing.")
        with tempfile.TemporaryDirectory(prefix="consult-asr-runtime-check-") as temporary:
            output = Path(temporary) / "silence.wav"
            subprocess.run([FFMPEG_EXECUTABLE, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                            "-i", "anullsrc=r=16000:cl=mono", "-t", "0.1", str(output)],
                           check=True, capture_output=True, timeout=30)
            with wave.open(str(output)) as audio:
                assert audio.getnchannels() == 1 and audio.getframerate() == 16000
        print("OK bundled FFmpeg audio conversion", flush=True)
    except Exception as exc:
        failures.append(f"FFmpeg: {type(exc).__name__}: {exc}")
    from core.emotion import check_emotion_assets
    tone = check_emotion_assets(TONE_MODEL_DIR, TONE_RUNTIME_DIR)
    if not tone.get("ready"):
        failures.append(f"Emotion assets: {tone.get('reason')}")
    if args.models:
        from tools.prepare_models import verify_models
        failures.extend(verify_models())
    for problem in failures:
        print(f"FAIL {problem}", file=sys.stderr)
    if not failures:
        print("Deployment checks passed. Inference accuracy is not evaluated by this check.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

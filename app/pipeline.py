from __future__ import annotations

import gc
import os
import re
import shutil
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping

from config import TEMP_ROOT
from core.aligner import align_words
from core.diarizer import Diarizer
from core.exporter import write_exports
from core.transcriber import Transcriber
from utils.audio_utils import ProcessingCancelled, normalize_audio


ProgressCallback = Callable[[str], None]


@dataclass
class FileAnalysis:
    input_path: Path
    output_dir: Path
    status: str
    words: list[dict[str, Any]] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float | None = None
    elapsed_seconds: float = 0.0
    error: str = ""
    word_path: Path | None = None
    json_path: Path | None = None
    roles_confirmed: bool = False
    counselor_speaker: str | None = None

    @property
    def display_name(self) -> str:
        return self.input_path.name


def _duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            return audio.getnframes() / rate if rate else None
    except (OSError, wave.Error):
        return None


def _safe_error(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    for key in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.environ.get(key)
        if value:
            message = message.replace(value, "[已隐藏凭据]")
    message = re.sub(r"\bhf_[A-Za-z0-9]{16,}\b", "[已隐藏凭据]", message)
    return message[:700]


def _release_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class ConsultationPipeline:
    """Analyze one audio file at a time, releasing ASR before diarization."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        compute_type: str | None = None,
        beam_size: int = 5,
        temp_root: str | os.PathLike[str] = TEMP_ROOT,
    ) -> None:
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.temp_root = Path(temp_root).expanduser().resolve()

    def analyze_file(
        self,
        input_path: str | os.PathLike[str],
        output_dir: str | os.PathLike[str],
        *,
        cancel_event: Event | None = None,
        progress: ProgressCallback | None = None,
    ) -> FileAnalysis:
        source = Path(input_path).expanduser().resolve()
        destination = Path(output_dir).expanduser().resolve()
        result = FileAnalysis(source, destination, status="processing")
        started = time.perf_counter()
        task_dir: Path | None = None
        diarizer: Diarizer | None = None
        try:
            if not source.is_file():
                raise FileNotFoundError("所选音频文件不存在。")
            if cancel_event and cancel_event.is_set():
                raise ProcessingCancelled("已在当前文件开始前取消。")
            self.temp_root.mkdir(parents=True, exist_ok=True)
            task_dir = Path(tempfile.mkdtemp(prefix="consult-asr-", dir=self.temp_root)).resolve()

            if progress:
                progress("正在规范化音频（16 kHz 单声道）…")
            normalized = normalize_audio(source, task_dir, cancel_event)
            result.duration_seconds = _duration_seconds(normalized)

            if progress:
                progress("正在识别中文语音…")
            transcriber = Transcriber(
                device=self.device,
                compute_type=self.compute_type,
                beam_size=self.beam_size,
            )
            result.words = transcriber.transcribe(normalized, cancel_event=cancel_event)
            del transcriber
            _release_memory()

            if cancel_event and cancel_event.is_set():
                raise ProcessingCancelled("已在语音识别安全边界取消。")
            if not result.words:
                result.status = "no_speech"
                return result

            if progress:
                progress("正在区分说话人…")
            diarizer = Diarizer(device=self.device)
            diarization = diarizer.diarize(normalized, cancel_event=cancel_event)
            result.words = align_words(diarization, result.words)
            result.segments = _merge_turns(result.words)
            if cancel_event and cancel_event.is_set():
                raise ProcessingCancelled("已在说话人分离安全边界取消。")
            result.status = "ready"
        except ProcessingCancelled as exc:
            result.status = "cancelled"
            result.error = _safe_error(exc)
        except Exception as exc:
            result.status = "failed"
            result.error = _safe_error(exc)
        finally:
            if diarizer is not None:
                diarizer.release()
                del diarizer
            _release_memory()
            if task_dir is not None:
                try:
                    shutil.rmtree(task_dir)
                except OSError:
                    pass
            result.elapsed_seconds = time.perf_counter() - started
        return result

    def export_file(
        self,
        result: FileAnalysis,
        *,
        counselor_speaker: str | None = None,
    ) -> tuple[Path, Path]:
        if result.status not in {"ready", "no_speech", "exported", "export_failed"}:
            raise ValueError("只有识别完成或无语音的文件可以导出。")
        role_labels: dict[str, str] = {}
        roles_confirmed = counselor_speaker in {"speaker_1", "speaker_2"} and bool(result.words)
        if roles_confirmed:
            other = "speaker_2" if counselor_speaker == "speaker_1" else "speaker_1"
            role_labels[counselor_speaker] = "咨询师"
            role_labels[other] = "来访者"
        try:
            word_path, json_path = write_exports(
                words=result.words,
                output_dir=result.output_dir,
                filename=result.input_path.stem,
                source_name=result.input_path.name,
                duration_seconds=result.duration_seconds,
                status="no_speech" if not result.words else "success",
                role_labels=role_labels,
                roles_confirmed=roles_confirmed,
            )
        except Exception as exc:
            result.status = "export_failed"
            result.error = _safe_error(exc)
            raise
        result.word_path = word_path
        result.json_path = json_path
        result.roles_confirmed = roles_confirmed
        result.counselor_speaker = counselor_speaker if roles_confirmed else None
        result.status = "exported"
        result.error = ""
        return word_path, json_path


def _merge_turns(words: list[dict[str, Any]], gap: float = 1.5) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for word in words:
        if (
            turns
            and turns[-1]["speaker_id"] == word.get("speaker_id")
            and float(word["start"]) - turns[-1]["end"] <= gap
        ):
            turns[-1]["end"] = max(turns[-1]["end"], float(word["end"]))
            turns[-1]["text"] += str(word.get("text", ""))
            turns[-1]["uncertain"] = turns[-1]["uncertain"] or bool(word.get("uncertain", False))
        else:
            turns.append({
                "start": float(word["start"]),
                "end": float(word["end"]),
                "text": str(word.get("text", "")),
                "speaker_id": word.get("speaker_id"),
                "uncertain": bool(word.get("uncertain", False)),
            })
    return turns

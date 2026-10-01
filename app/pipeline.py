from __future__ import annotations

import gc
import math
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

from config import TEMP_ROOT, TONE_MODEL_DIR, TONE_RUNTIME_DIR
from core.acoustics import AcousticsCancelled, analyze_acoustics
from core.aligner import align_words, speaker_tracks
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
    emotion_windows: list[dict[str, Any]] | None = None
    analysis_windows: list[dict[str, Any]] = field(default_factory=list)
    tone_enabled: bool | None = None
    tone_model: dict[str, Any] | None = None
    tone_status: str | None = None
    tone_error: str = ""

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
        tone_enabled: bool | None = None,
        tone_model_dir: str | os.PathLike[str] = TONE_MODEL_DIR,
        tone_runtime_dir: str | os.PathLike[str] = TONE_RUNTIME_DIR,
    ) -> None:
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.temp_root = Path(temp_root).expanduser().resolve()
        self.tone_enabled = None if tone_enabled is None else bool(tone_enabled)
        self.tone_model_dir = Path(tone_model_dir).expanduser().resolve()
        self.tone_runtime_dir = Path(tone_runtime_dir).expanduser().resolve()

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
        emotion_recognizer: Any | None = None
        result.tone_enabled = self.tone_enabled
        try:
            if not source.is_file():
                raise FileNotFoundError("所选音频文件不存在。")
            if cancel_event and cancel_event.is_set():
                raise ProcessingCancelled("已在当前文件开始前取消。")
            emotion_api: Any | None = None
            if self.tone_enabled is True:
                try:
                    from core.emotion import (
                        EmotionCancelled,
                        EmotionRuntimeError,
                        EmotionRecognizer,
                        check_emotion_assets,
                    )

                    assets = check_emotion_assets(self.tone_model_dir, self.tone_runtime_dir)
                    if not assets.get("ready"):
                        result.tone_status = "failed"
                        result.tone_error = str(assets.get("reason") or "情绪模型资源未就绪。")
                    else:
                        result.tone_model = _public_model_metadata(assets)
                    emotion_api = (EmotionCancelled, EmotionRuntimeError, EmotionRecognizer)
                except Exception as exc:
                    result.tone_status = "failed"
                    result.tone_error = _safe_error(exc)
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
                if self.tone_enabled is True:
                    result.emotion_windows = []
                    result.tone_status = "no_speech"
                elif self.tone_enabled is False:
                    result.tone_status = "disabled"
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

            if self.tone_enabled is not None:
                if progress:
                    progress("正在提取语速、音量、停顿和音高走势…")
                result.analysis_windows = _build_analysis_windows(
                    result.words,
                    speaker_tracks(diarization),
                    result.duration_seconds,
                )
                for window in result.analysis_windows:
                    if cancel_event and cancel_event.is_set():
                        raise ProcessingCancelled("已在声学特征分析安全边界取消。")
                    if not window.get("analysis_safe"):
                        continue
                    text = "".join(window.pop("_word_texts", []))
                    try:
                        acoustics = analyze_acoustics(
                            normalized,
                            window["start"],
                            window["end"],
                            text,
                            cancel_event=cancel_event,
                        )
                    except AcousticsCancelled as exc:
                        raise ProcessingCancelled("已在声学特征分析安全边界取消。") from exc
                    acoustics["speech_rate_method"] = "asr_word_midpoint_assigned_to_safe_window"
                    window["acoustics"] = acoustics
                    window["chinese_character_count"] = acoustics["chinese_character_count"]
                    window["denominator_seconds"] = acoustics["denominator_seconds"]
                if self.tone_enabled is False:
                    for window in result.analysis_windows:
                        if window.get("analysis_safe"):
                            window.update({
                                "status": "disabled",
                                "label": None,
                                "model_score": None,
                                "reason": None,
                            })

            if self.tone_enabled is True:
                if progress:
                    progress("正在分析局部语音情绪倾向（CPU）…")
                result.emotion_windows = result.analysis_windows
                if result.tone_status != "failed":
                    emotion_cancelled, emotion_runtime_error, recognizer_type = emotion_api
                    candidates = [
                        window for window in result.emotion_windows
                        if window.get("status") == "pending"
                    ]
                    try:
                        if candidates:
                            emotion_recognizer = recognizer_type(
                                model_dir=self.tone_model_dir,
                                runtime_dir=self.tone_runtime_dir,
                            )
                            for window in candidates:
                                if cancel_event and cancel_event.is_set():
                                    raise ProcessingCancelled("已在情绪分析窗口边界取消。")
                                try:
                                    prediction = emotion_recognizer.recognize(
                                        normalized,
                                        window["start"],
                                        window["end"],
                                        cancel_event=cancel_event,
                                    )
                                except emotion_cancelled as exc:
                                    raise ProcessingCancelled("已在情绪分析安全边界取消。") from exc
                                except emotion_runtime_error as exc:
                                    if getattr(exc, "error_code", None) != "short_audio":
                                        raise
                                    window.update({
                                        "status": "unknown",
                                        "label": "unknown",
                                        "reason": "window_too_short",
                                    })
                                    continue
                                _apply_emotion_prediction(window, prediction)
                                if prediction.get("model"):
                                    result.tone_model = _public_model_metadata(prediction["model"])
                    except ProcessingCancelled:
                        raise
                    except emotion_runtime_error as exc:
                        result.tone_error = _safe_error(exc)
                        succeeded = any(item.get("status") == "ok" for item in result.emotion_windows)
                        result.tone_status = "partial" if succeeded else "failed"
                        for window in candidates:
                            if window.get("status") == "pending":
                                window.update({
                                    "status": "unknown",
                                    "label": "unknown",
                                    "reason": "emotion_runtime_failed",
                                })
                    except Exception as exc:
                        result.tone_error = _safe_error(exc)
                        succeeded = any(item.get("status") == "ok" for item in result.emotion_windows)
                        result.tone_status = "partial" if succeeded else "failed"
                        for window in candidates:
                            if window.get("status") == "pending":
                                window.update({
                                    "status": "unknown",
                                    "label": "unknown",
                                    "reason": "emotion_runtime_failed",
                                })
                    else:
                        result.tone_status = "complete"
                else:
                    for window in result.emotion_windows:
                        if window.get("status") == "pending":
                            window.update({
                                "status": "unknown",
                                "label": "unknown",
                                "reason": "emotion_runtime_failed",
                            })
            if cancel_event and cancel_event.is_set():
                raise ProcessingCancelled("已在文件分析完成前取消。")
            result.status = "ready"
            if self.tone_enabled is False:
                result.tone_status = "disabled"
        except ProcessingCancelled as exc:
            result.status = "cancelled"
            result.error = _safe_error(exc)
        except Exception as exc:
            result.status = "failed"
            result.error = _safe_error(exc)
        finally:
            if emotion_recognizer is not None:
                try:
                    emotion_recognizer.close()
                except Exception:
                    pass
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
                segments=result.segments if result.tone_enabled is not None else None,
                emotion_windows=result.emotion_windows,
                analysis_windows=result.analysis_windows if result.tone_enabled is not None else None,
                tone_enabled=result.tone_enabled,
                tone_status=result.tone_status,
                tone_error=result.tone_error,
                tone_model=result.tone_model,
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


_EMOTION_MAX_WINDOW_SECONDS = 6.0
_EMOTION_MIN_WINDOW_SECONDS = 1.0

def _build_analysis_windows(
    words: list[dict[str, Any]],
    tracks: list[tuple[float, float, str]],
    duration_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Build ≤6 s windows from raw diarization coverage and word certainty."""
    valid_words: list[dict[str, Any]] = []
    for word_index, word in enumerate(words):
        try:
            start, end = float(word["start"]), float(word["end"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            continue
        if duration_seconds is not None and math.isfinite(duration_seconds):
            if start >= duration_seconds:
                continue
            end = min(end, duration_seconds)
        if end <= start:
            continue
        valid_words.append({
            "id": word_index,
            "start": start,
            "end": end,
            "speaker_id": word.get("speaker_id"),
            "uncertain": bool(word.get("uncertain", False)),
            "text": str(word.get("text", "")),
        })
    if not valid_words:
        return []

    envelope_start = min(word["start"] for word in valid_words)
    envelope_end = max(word["end"] for word in valid_words)
    valid_tracks: dict[int, dict[str, Any]] = {}
    for track_index, (raw_start, raw_end, speaker_id) in enumerate(tracks):
        try:
            start, end = float(raw_start), float(raw_end)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            continue
        start = max(start, envelope_start)
        end = min(end, envelope_end)
        if duration_seconds is not None and math.isfinite(duration_seconds):
            end = min(end, duration_seconds)
        if end <= start:
            continue
        valid_tracks[track_index] = {
            "id": track_index,
            "start": start,
            "end": end,
            "speaker_id": speaker_id,
        }

    valid_words_by_id = {word["id"]: word for word in valid_words}
    events: dict[float, list[tuple[int, str, int]]] = {
        envelope_start: [],
        envelope_end: [],
    }
    for track_id, track in valid_tracks.items():
        events.setdefault(track["start"], []).append((1, "track", track_id))
        events.setdefault(track["end"], []).append((-1, "track", track_id))
    for word_id, word in valid_words_by_id.items():
        events.setdefault(word["start"], []).append((1, "word", word_id))
        events.setdefault(word["end"], []).append((-1, "word", word_id))

    times = sorted(events)
    active_tracks: dict[int, dict[str, Any]] = {}
    active_words: dict[int, dict[str, Any]] = {}
    spans: list[dict[str, Any]] = []
    for point, next_point in zip(times, times[1:]):
        # End events are applied before start events so all intervals are [start, end).
        for direction, kind, event_id in sorted(events[point], key=lambda event: event[0]):
            bucket = active_tracks if kind == "track" else active_words
            source = valid_tracks if kind == "track" else valid_words_by_id
            if direction < 0:
                bucket.pop(event_id, None)
            else:
                bucket[event_id] = source[event_id]
        if next_point <= point:
            continue

        active_speakers = {item["speaker_id"] for item in active_tracks.values()}
        reason: str | None = None
        speaker_id: str | None = None
        if len(active_speakers) > 1:
            reason = "speaker_overlap"
        elif any(word["uncertain"] for word in active_words.values()):
            reason = "uncertain_word_or_speaker"
        elif any(word["speaker_id"] not in {"speaker_1", "speaker_2"} for word in active_words.values()):
            reason = "speaker_unassigned"
        elif not active_speakers:
            reason = "no_speaker_track"
        elif next(iter(active_speakers)) not in {"speaker_1", "speaker_2"}:
            reason = "speaker_unassigned"
        else:
            speaker_id = next(iter(active_speakers))
            if any(word["speaker_id"] != speaker_id for word in active_words.values()):
                reason = "speaker_alignment_mismatch"

        state = "unknown" if reason else "safe"
        if spans and spans[-1]["end"] == point and spans[-1]["state"] == state \
                and spans[-1].get("speaker_id") == speaker_id and spans[-1].get("reason") == reason:
            spans[-1]["end"] = next_point
        else:
            spans.append({
                "start": point,
                "end": next_point,
                "state": state,
                "speaker_id": speaker_id,
                "reason": reason,
            })

    windows: list[dict[str, Any]] = []
    for span in spans:
        span_duration = span["end"] - span["start"]
        if span_duration <= 0:
            continue
        piece_count = max(1, math.ceil(span_duration / _EMOTION_MAX_WINDOW_SECONDS))
        piece_duration = span_duration / piece_count
        for piece_index in range(piece_count):
            start = span["start"] + piece_duration * piece_index
            end = span["end"] if piece_index == piece_count - 1 else span["start"] + piece_duration * (piece_index + 1)
            safe = span["state"] == "safe"
            reason = span.get("reason")
            if safe and end - start < _EMOTION_MIN_WINDOW_SECONDS:
                status, label, reason = "unknown", "unknown", "window_too_short"
            elif safe:
                status, label = "pending", None
            else:
                status, label = "unknown", "unknown"
            windows.append({
                "start": float(start),
                "end": float(end),
                "speaker_id": span.get("speaker_id"),
                "analysis_safe": safe,
                "status": status,
                "label": label,
                "model_score": None,
                "reason": reason,
                "_word_texts": [],
            })

    # Count each fully safe ASR word once. A midpoint assigns words crossing a
    # maximum-window boundary to exactly one adjacent window.
    for word in sorted(valid_words, key=lambda item: (item["start"], item["end"], item["id"])):
        speaker_id = word["speaker_id"]
        if word["uncertain"] or speaker_id not in {"speaker_1", "speaker_2"}:
            continue
        cursor = word["start"]
        covered = True
        for window in windows:
            if window["end"] <= cursor:
                continue
            if window["start"] > cursor or not window["analysis_safe"] or window["speaker_id"] != speaker_id:
                covered = False
                break
            cursor = min(word["end"], window["end"])
            if cursor >= word["end"]:
                break
        if not covered or cursor < word["end"]:
            continue
        midpoint = (word["start"] + word["end"]) / 2.0
        owner = next((
            window for window in windows
            if window["analysis_safe"] and window["speaker_id"] == speaker_id
            and window["start"] <= midpoint < window["end"]
        ), None)
        if owner is not None:
            owner["_word_texts"].append(word["text"])

    return windows


def _apply_emotion_prediction(window: dict[str, Any], prediction: Mapping[str, Any]) -> None:
    labels = [str(label) for label in prediction.get("labels", [])]
    scores = [float(score) for score in prediction.get("scores", [])]
    if not labels or len(labels) != len(scores) or any(not math.isfinite(score) for score in scores):
        raise RuntimeError("情绪模型返回的 labels/scores 结构无效。")
    top_index = max(range(len(scores)), key=scores.__getitem__)
    window.update({
        "status": "ok",
        "label": labels[top_index],
        "model_score": scores[top_index],
        "scores": scores,
        "labels": labels,
        "model": _public_model_metadata(prediction.get("model", {})),
        "reason": None,
    })


def _public_model_metadata(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    allowed = {
        "id",
        "model_id",
        "revision",
        "model_revision",
        "weights_sha256",
        "device",
        "runtime_version",
    }
    return {str(key): item for key, item in value.items() if key in allowed and isinstance(item, (str, int, float, bool, type(None)))}

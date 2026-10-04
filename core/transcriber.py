from __future__ import annotations

import os
import gc
import time
from pathlib import Path
from threading import Event
from typing import Any, Callable

from config import ASR_MODEL_DIR, OPENAI_WHISPER_MODEL_DIR, WHISPER_MODEL
from utils.audio_utils import ProcessingCancelled
from utils.logger import log_event


def _enable_offline_mode() -> None:
    """Keep all model libraries from trying network access during inference."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _model_files_present(path: Path) -> bool:
    required = (
        "model.bin",
        "config.json",
        "tokenizer.json",
        "vocabulary.json",
        "preprocessor_config.json",
    )
    return (
        path.is_dir()
        and all((path / name).is_file() for name in required)
        and (path / "model.bin").stat().st_size > 0
    )


def _local_whisper_weights(path: Path) -> Path | None:
    if path.is_file() and path.suffix.lower() == ".pt":
        return path
    if path.is_dir():
        candidates = sorted(path.glob("*.pt"))
        return candidates[0] if candidates else None
    return None


class Transcriber:
    """Lazy, local-only Chinese ASR with word timestamps when faster-whisper is available."""

    def __init__(
        self,
        device: str = "cuda",
        compute_type: str | None = None,
        beam_size: int = 5,
        model_path: str | os.PathLike[str] = ASR_MODEL_DIR,
        fallback_path: str | os.PathLike[str] = OPENAI_WHISPER_MODEL_DIR,
        inference_mode: str = "accurate",
        batch_size: int = 2,
        cpu_threads: int = 8,
    ) -> None:
        self.device = device
        self.compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self.beam_size = max(1, int(beam_size))
        self.requested_inference_mode = inference_mode
        self.requested_batch_size = min(4, max(2, int(batch_size)))
        self.cpu_threads = min(8, max(1, int(cpu_threads)), max(1, os.cpu_count() or 1))
        self.model_path = Path(model_path).expanduser().resolve()
        self.fallback_path = Path(fallback_path).expanduser().resolve()
        self._model: Any = None
        self._batched_model: Any = None
        self._backend: str | None = None
        self.effective_inference_mode = "accurate"
        self.effective_batch_size = 0
        self.fallback_path_taken: str | None = None
        self.model_load_seconds = 0.0
        self.inference_seconds = 0.0

    @property
    def is_ready(self) -> bool:
        return _model_files_present(self.model_path) or _local_whisper_weights(self.fallback_path) is not None

    @property
    def backend(self) -> str | None:
        return self._backend

    def load(self) -> None:
        if self._model is not None:
            return
        _enable_offline_mode()

        if _model_files_present(self.model_path):
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("缺少 faster-whisper 运行依赖，请按部署说明安装锁定依赖。") from exc
            try:
                load_started = time.perf_counter()
                self._model = WhisperModel(
                    str(self.model_path),
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    num_workers=1,
                )
                self.model_load_seconds = time.perf_counter() - load_started
            except Exception as exc:
                if self.device == "cuda":
                    raise RuntimeError(
                        "本地 large-v3 模型无法在当前 GPU 配置中加载；可在设置中选择 CPU，"
                        "或按部署说明检查 CUDA/PyTorch/CTranslate2 版本。"
                    ) from exc
                raise RuntimeError("本地 large-v3 模型加载失败，请检查模型文件完整性。") from exc
            self._backend = "faster-whisper"
            if self.requested_inference_mode == "gpu_batch" and self.device == "cuda":
                try:
                    from faster_whisper import BatchedInferencePipeline
                    self._batched_model = BatchedInferencePipeline(self._model)
                    self.effective_inference_mode = "gpu_batch_vad"
                    self.effective_batch_size = self.requested_batch_size
                except (ImportError, AttributeError, TypeError):
                    self.fallback_path_taken = "batch_api_unavailable"
                    self.effective_inference_mode = "accurate_batch_api_fallback"
            elif self.requested_inference_mode == "gpu_batch":
                self.fallback_path_taken = "cuda_unavailable"
                self.effective_inference_mode = "accurate_cuda_fallback"
            return

        fallback = _local_whisper_weights(self.fallback_path)
        if fallback is not None:
            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError("存在本地 Whisper 权重，但缺少 openai-whisper 回退依赖。") from exc
            try:
                load_started = time.perf_counter()
                self._model = whisper.load_model(str(fallback), device=self.device)
                self.model_load_seconds = time.perf_counter() - load_started
            except Exception as exc:
                raise RuntimeError("本地 Whisper 回退权重加载失败。") from exc
            self._backend = "openai-whisper-local-fallback"
            if self.requested_inference_mode == "gpu_batch":
                self.fallback_path_taken = "batched_backend_unavailable"
                self.effective_inference_mode = "accurate_backend_fallback"
            return

        raise FileNotFoundError(
            f"未找到本地 {WHISPER_MODEL} 模型。请先运行离线模型准备脚本；推理不会自动联网下载模型。"
        )

    def transcribe(
        self,
        audio_path: str | os.PathLike[str],
        cancel_event: Event | None = None,
        progress: Callable[[str], None] | None = None,
        progress_seconds: Callable[[float], None] | None = None,
        diagnostic_task_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        audio = Path(audio_path).expanduser().resolve()
        if not audio.is_file():
            raise FileNotFoundError("规范化音频文件不存在。")
        if cancel_event and cancel_event.is_set():
            raise ProcessingCancelled("已在语音识别前取消。")
        self.load()
        if progress:
            progress("正在识别中文语音…")

        if self._backend == "faster-whisper":
            if self._batched_model is not None:
                self.fallback_path_taken = None
                for attempt_index, batch_size in enumerate((self.requested_batch_size, 1)):
                    oom = False
                    try:
                        words = self._transcribe_batched(
                            audio, batch_size, cancel_event, progress_seconds
                        )
                    except ProcessingCancelled:
                        raise
                    except Exception as exc:
                        if not self._is_cuda_oom(exc):
                            raise
                        if cancel_event and cancel_event.is_set():
                            raise ProcessingCancelled("已在语音识别分段边界取消。") from exc
                        oom = True

                    if not oom:
                        self.effective_batch_size = batch_size
                        self.effective_inference_mode = (
                            "gpu_batch_vad"
                            if attempt_index == 0
                            else "gpu_batch_vad_batch1_fallback"
                        )
                        if attempt_index > 0:
                            self.fallback_path_taken = "batch_oom_batch1"
                        return words

                    # Leave the except block before collecting again so its traceback
                    # cannot keep the failed generator's CUDA locals alive.
                    next_fallback_code = (
                        "batch_oom_batch1" if attempt_index == 0 else "batch_oom_serial"
                    )
                    log_event(
                        "retry",
                        task_seq=diagnostic_task_seq,
                        fallback_code=next_fallback_code,
                        attempt=attempt_index + 1,
                    )
                    self._clear_cuda_cache()

                words = self._transcribe_serial(audio, cancel_event, progress_seconds)
                self.effective_batch_size = 0
                self.effective_inference_mode = "accurate_oom_serial_fallback"
                self.fallback_path_taken = "batch_oom_serial"
                return words
            if self.fallback_path_taken is None:
                self.effective_inference_mode = "accurate"
            return self._transcribe_serial(audio, cancel_event, progress_seconds)

        started = time.perf_counter()
        try:
            result = self._model.transcribe(
                str(audio),
                language="zh",
                task="transcribe",
                fp16=self.device == "cuda",
                verbose=False,
            )
        finally:
            self.inference_seconds += time.perf_counter() - started
        if cancel_event and cancel_event.is_set():
            raise ProcessingCancelled("已在语音识别完成边界取消。")
        return [
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": str(segment.get("text", "")),
                "uncertain": True,
            }
            for segment in result.get("segments", [])
            if str(segment.get("text", "")).strip()
        ]

    def _transcribe_serial(
        self,
        audio: Path,
        cancel_event: Event | None,
        progress_seconds: Callable[[float], None] | None,
    ) -> list[dict[str, Any]]:
        started = time.perf_counter()
        try:
            raw_segments, _ = self._model.transcribe(
                str(audio),
                language="zh",
                task="transcribe",
                beam_size=self.beam_size,
                word_timestamps=True,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            return self._collect_words(raw_segments, cancel_event, progress_seconds)
        finally:
            self.inference_seconds += time.perf_counter() - started

    def _transcribe_batched(
        self,
        audio: Path,
        batch_size: int,
        cancel_event: Event | None,
        progress_seconds: Callable[[float], None] | None,
    ) -> list[dict[str, Any]]:
        if cancel_event and cancel_event.is_set():
            raise ProcessingCancelled("已在语音识别前取消。")
        started = time.perf_counter()
        raw_segments: Any | None = None
        self._reset_batched_state()
        try:
            raw_segments, _ = self._batched_model.transcribe(
                str(audio),
                language="zh",
                task="transcribe",
                beam_size=self.beam_size,
                word_timestamps=True,
                without_timestamps=False,
                vad_filter=True,
                condition_on_previous_text=False,
                batch_size=batch_size,
            )
            return self._collect_words(raw_segments, cancel_event, progress_seconds)
        finally:
            if raw_segments is not None:
                close = getattr(raw_segments, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            self._reset_batched_state()
            self.inference_seconds += time.perf_counter() - started

    def _reset_batched_state(self) -> None:
        if self._batched_model is not None and hasattr(self._batched_model, "last_speech_timestamp"):
            self._batched_model.last_speech_timestamp = 0.0

    @staticmethod
    def _collect_words(
        raw_segments: Any,
        cancel_event: Event | None,
        progress_seconds: Callable[[float], None] | None,
    ) -> list[dict[str, Any]]:
        words: list[dict[str, Any]] = []
        for segment in raw_segments:
            if cancel_event and cancel_event.is_set():
                raise ProcessingCancelled("已在语音识别分段边界取消。")
            segment_words = getattr(segment, "words", None) or []
            if segment_words:
                for word in segment_words:
                    start = float(word.start)
                    end = float(word.end)
                    text = str(word.word)
                    if end >= start and text.strip():
                        probability = getattr(word, "probability", None)
                        words.append({
                            "start": start,
                            "end": end,
                            "text": text,
                            "uncertain": probability is not None and float(probability) < 0.35,
                        })
            else:
                text = str(getattr(segment, "text", ""))
                if text.strip():
                    words.append({
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": text,
                        "uncertain": True,
                    })
            if progress_seconds:
                progress_seconds(float(segment.end))
        return sorted(words, key=lambda item: (item["start"], item["end"]))

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        message = f"{type(exc).__name__}: {exc}".lower()
        return any(marker in message for marker in (
            "out of memory",
            "cuda_error_out_of_memory",
            "cudnn_status_alloc_failed",
            "cuda error 2",
        ))

    @staticmethod
    def _clear_cuda_cache() -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

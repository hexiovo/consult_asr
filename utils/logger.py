"""Privacy-bounded, local diagnostic events for the desktop application.

Only events and values declared below can reach disk. This module deliberately
does not capture stdout, stderr, Python warnings, exception messages, or paths.
"""

from __future__ import annotations

import errno
import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import LOG_DIR


LOG_FILE_NAME = "consult-asr-events.log"
MAX_BYTES = 2 * 1024 * 1024
BACKUP_COUNT = 3
LOCK_TIMEOUT_SECONDS = 0.1
_MAX_RECORD_BYTES = 16 * 1024

_STAGES = frozenset(
    {
        "normalize",
        "asr_model_load",
        "asr",
        "diarization_model_load",
        "diarization",
        "acoustics",
        "emotion",
        "cleanup",
        "export",
        "queue",
        "application",
    }
)
_FALLBACK_CODES = frozenset(
    {
        "batch_api_unavailable",
        "cuda_unavailable",
        "batched_backend_unavailable",
        "batch_oom_batch1",
        "batch_oom_serial",
    }
)
_INFERENCE_MODES = frozenset(
    {
        "accurate",
        "gpu_batch",
        "gpu_batch_vad",
        "accurate_cuda_fallback",
        "accurate_batch_api_fallback",
        "accurate_backend_fallback",
        "gpu_batch_vad_batch1_fallback",
        "accurate_oom_serial_fallback",
    }
)
_COMPUTE_TYPES = frozenset(
    {"auto", "float16", "bfloat16", "float32", "int8", "int8_float16", "int8_bfloat16"}
)
_ERROR_CATEGORIES = frozenset(
    {"file_missing", "permission", "timeout", "memory", "invalid_value", "runtime", "type", "os_error", "unknown"}
)
_ERRNO_CATEGORIES = frozenset(
    {"missing", "permission", "disk_full", "busy", "timeout", "io_error", "unknown"}
)
_OUTCOMES = frozenset(
    {"ready", "no_speech", "cancelled", "failed", "exported", "export_failed", "success"}
)
_TONE_STATUSES = frozenset({"complete", "failed", "partial", "disabled", "no_speech", "unknown"})

# Each event has an explicit field schema. Unknown fields and invalid values are
# omitted, so call sites cannot accidentally serialize arbitrary user content.
_EVENT_FIELDS: dict[str, dict[str, Any]] = {
    "app_start": {},
    "app_stop": {},
    "processing_start": {},
    "stage_complete": {"stage": _STAGES, "elapsed_ms": "duration"},
    "inference_config": {
        "device": frozenset({"cpu", "cuda"}),
        "compute_type": _COMPUTE_TYPES,
        "beam_size": "small_int",
        "requested_mode": _INFERENCE_MODES,
        "effective_mode": _INFERENCE_MODES,
        "requested_batch_size": "batch_size",
        "effective_batch_size": "batch_size",
        "cpu_threads": "small_int",
    },
    "fallback": {
        "fallback_code": _FALLBACK_CODES,
        "effective_mode": _INFERENCE_MODES,
        "effective_batch_size": "batch_size",
    },
    "retry": {"fallback_code": _FALLBACK_CODES, "attempt": "small_int"},
    "cancel": {"stage": _STAGES},
    "terminal": {
        "outcome": _OUTCOMES,
        "stage": _STAGES,
        "error_category": _ERROR_CATEGORIES,
        "errno_category": _ERRNO_CATEGORIES,
        "tone_status": _TONE_STATUSES,
    },
    "tone_result": {
        "tone_status": _TONE_STATUSES,
        "error_category": _ERROR_CATEGORIES,
        "errno_category": _ERRNO_CATEGORIES,
    },
    "export": {
        "outcome": frozenset({"success", "failed"}),
        "error_category": _ERROR_CATEGORIES,
        "errno_category": _ERRNO_CATEGORIES,
    },
    "session_restore": {"record_count": "count"},
}

_STATE_LOCK = threading.RLock()
_HANDLER: _EventFileHandler | None = None
_RUN_ID: str | None = None
_TASK_SEQUENCE = 0


def _valid_integer(value: Any, minimum: int, maximum: int) -> bool:
    return type(value) is int and minimum <= value <= maximum


def _sanitize_fields(event: str, fields: dict[str, Any]) -> dict[str, int | str]:
    schema = _EVENT_FIELDS[event]
    clean: dict[str, int | str] = {}
    for name, value in fields.items():
        validator = schema.get(name)
        if validator is None:
            continue
        if isinstance(validator, frozenset):
            if type(value) is str and value in validator:
                clean[name] = value
        elif validator == "duration":
            if _valid_integer(value, 0, 86_400_000):
                clean[name] = value
        elif validator == "small_int":
            if _valid_integer(value, 0, 64):
                clean[name] = value
        elif validator == "batch_size":
            if _valid_integer(value, 0, 4):
                clean[name] = value
        elif validator == "count":
            if _valid_integer(value, 0, 1_000_000):
                clean[name] = value
    return clean


def _errno_category(exc: BaseException) -> str:
    if not isinstance(exc, OSError):
        return "unknown"
    code = exc.errno
    if code == errno.ENOENT:
        return "missing"
    if code in {errno.EACCES, errno.EPERM}:
        return "permission"
    if code == errno.ENOSPC:
        return "disk_full"
    if code == errno.EBUSY:
        return "busy"
    if code == errno.ETIMEDOUT:
        return "timeout"
    if code == errno.EIO:
        return "io_error"
    return "unknown"


def _error_category(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return "file_missing"
    if isinstance(exc, PermissionError):
        return "permission"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, MemoryError):
        return "memory"
    if isinstance(exc, ValueError):
        return "invalid_value"
    if isinstance(exc, RuntimeError):
        return "runtime"
    if isinstance(exc, TypeError):
        return "type"
    if isinstance(exc, OSError):
        return "os_error"
    return "unknown"


def exception_fields(exc: BaseException) -> dict[str, str]:
    """Return only fixed error categories; never inspect exception text."""
    try:
        return {
            "error_category": _error_category(exc),
            "errno_category": _errno_category(exc),
        }
    except Exception:
        return {"error_category": "unknown", "errno_category": "unknown"}


class _EventFileHandler:
    """One-record-at-a-time writer coordinated across threads and processes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_name(path.name + ".lock")
        self._thread_lock = threading.RLock()
        self._closed = False

    def close(self) -> None:
        with self._thread_lock:
            self._closed = True

    def emit(self, record: dict[str, Any]) -> bool:
        try:
            payload = (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
            if len(payload) > _MAX_RECORD_BYTES:
                return False
            with self._thread_lock:
                if self._closed:
                    return False
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self._locked_file():
                    self._rotate_if_needed(len(payload))
                    # Opening and closing for every event prevents a process from
                    # retaining a handle to a file another process has rotated.
                    descriptor = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
                    start_size: int | None = None
                    try:
                        start_size = os.fstat(descriptor).st_size
                        offset = 0
                        while offset < len(payload):
                            written = os.write(descriptor, payload[offset:])
                            if written <= 0:
                                raise OSError("event write made no progress")
                            offset += written
                    except Exception:
                        if start_size is not None:
                            try:
                                os.ftruncate(descriptor, start_size)
                            except OSError:
                                pass
                        raise
                    finally:
                        os.close(descriptor)
            return True
        except Exception:
            # Logging is best-effort: never let diagnostics interrupt processing.
            return False

    def _locked_file(self):
        return _ProcessFileLock(self.lock_path, LOCK_TIMEOUT_SECONDS)

    def _backup_path(self, index: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{index}")

    def _rotate_if_needed(self, incoming_size: int) -> None:
        try:
            current_size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if current_size + incoming_size <= MAX_BYTES:
            return
        for index in range(BACKUP_COUNT, 0, -1):
            destination = self._backup_path(index)
            source = self.path if index == 1 else self._backup_path(index - 1)
            try:
                destination.unlink(missing_ok=True)
                if source.exists():
                    os.replace(source, destination)
            except OSError:
                raise


class _ProcessFileLock:
    def __init__(self, path: Path, timeout: float) -> None:
        self.path = path
        self.timeout = timeout
        self._stream = None
        self._locked = False

    def __enter__(self) -> _ProcessFileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            self._stream = os.fdopen(descriptor, "r+b", buffering=0)
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        try:
            if os.fstat(descriptor).st_size == 0:
                self._stream.seek(0)
                self._stream.write(b"\0")
            deadline = time.monotonic() + self.timeout
            while True:
                try:
                    self._try_lock()
                    self._locked = True
                    return self
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("bounded log lock timeout")
                    time.sleep(0.005)
        except Exception:
            stream = self._stream
            self._stream = None
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
            raise

    def _try_lock(self) -> None:
        assert self._stream is not None
        self._stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def __exit__(self, exc_type, exc, traceback) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            if self._locked:
                if os.name == "nt":
                    import msvcrt

                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            stream.close()


def _close_handler(handler: _EventFileHandler | None) -> None:
    if handler is not None:
        try:
            handler.close()
        except Exception:
            pass


def _disable_logging() -> _EventFileHandler | None:
    global _HANDLER, _RUN_ID, _TASK_SEQUENCE
    try:
        with _STATE_LOCK:
            old_handler = _HANDLER
            _HANDLER = None
            _RUN_ID = None
            _TASK_SEQUENCE = 0
            return old_handler
    except Exception:
        # A damaged lock must still not let diagnostics prevent application startup.
        old_handler = _HANDLER
        _HANDLER = None
        _RUN_ID = None
        _TASK_SEQUENCE = 0
        return old_handler


def init_logging(log_dir: os.PathLike[str] | str = LOG_DIR) -> bool:
    """Enable best-effort local events; any initialization failure disables logging."""
    global _HANDLER, _RUN_ID, _TASK_SEQUENCE
    new_handler: _EventFileHandler | None = None
    old_handler: _EventFileHandler | None = None
    try:
        path = Path(log_dir) / LOG_FILE_NAME
        new_handler = _EventFileHandler(path)
        run_id = secrets.token_hex(12)
        with _STATE_LOCK:
            old_handler = _HANDLER
            _HANDLER = new_handler
            _RUN_ID = run_id
            _TASK_SEQUENCE = 0
    except Exception:
        old_handler = _disable_logging()
        _close_handler(new_handler)
        _close_handler(old_handler)
        return False
    _close_handler(old_handler)
    log_event("app_start")
    return True


def shutdown_logging() -> None:
    """Write a final event when possible, then disable and close the handler."""
    global _HANDLER, _RUN_ID, _TASK_SEQUENCE
    try:
        log_event("app_stop")
    except Exception:
        pass
    handler = _disable_logging()
    _close_handler(handler)


def is_logging_enabled() -> bool:
    """Return whether event logging is active without propagating logger faults."""
    try:
        with _STATE_LOCK:
            return _HANDLER is not None and _RUN_ID is not None
    except Exception:
        return False


def begin_task() -> int | None:
    """Return this process's anonymous single-file processing-task sequence."""
    global _TASK_SEQUENCE
    try:
        with _STATE_LOCK:
            if _HANDLER is None or _RUN_ID is None:
                return None
            _TASK_SEQUENCE += 1
            return _TASK_SEQUENCE
    except Exception:
        return None


def begin_batch() -> int | None:
    """Backward-compatible alias; the identifier denotes one processing task."""
    return begin_task()


def log_event(event: str, *, task_seq: int | None = None, **fields: Any) -> bool:
    """Write a schema-checked event. Uninitialized or failed logging is a no-op."""
    if type(event) is not str or event not in _EVENT_FIELDS:
        return False
    try:
        with _STATE_LOCK:
            handler = _HANDLER
            run_id = _RUN_ID
    except Exception:
        return False
    if handler is None or run_id is None:
        return False
    record: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "event": event,
    }
    if _valid_integer(task_seq, 1, 1_000_000):
        record["task_seq"] = task_seq
    record.update(_sanitize_fields(event, fields))
    try:
        return bool(handler.emit(record))
    except Exception:
        return False


def log_exception(
    event: str,
    exc: BaseException,
    *,
    task_seq: int | None = None,
    **fields: Any,
) -> bool:
    """Add fixed exception/errno categories without serializing exception text."""
    fields.update(exception_fields(exc))
    return log_event(event, task_seq=task_seq, **fields)

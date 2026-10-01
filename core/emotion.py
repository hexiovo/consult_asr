"""Isolated, local CPU inference for the emotion2vec speech model.

The parent process talks to a long-lived worker over newline-delimited JSON.
This keeps the optional FunASR runtime out of the application's import path and
lets cancellation terminate only the emotion worker.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


MODEL_ID = "iic/emotion2vec_plus_base"
MODEL_REVISION = "58354f1e5b6c118cf8b175b5b1955d6b07df42c2"
WEIGHTS_SHA256 = "60710b5aae1dbe69bdac8920028fb05882d4314fd09031922b4b61ee9e7aadbd"
MANIFEST_NAME = "emotion2vec.manifest.json"
RUNTIME_MANIFEST_NAME = "emotion-runtime.manifest.json"
LABELS = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
)

EXPECTED_FILES = {
    "model.pt": {
        "size": 1_118_245_678,
        "sha256": WEIGHTS_SHA256,
    },
    "config.yaml": {
        "size": 3_170,
        "sha256": "50674ec187406b4f1208925594ceaf1f28f16a22333b5f2ac5b3d0cbffdd670e",
    },
    "configuration.json": {
        "size": 343,
        "sha256": "8b6a745213025c7d4565f9f074cfef1b5cd5ef76e38e2a8f7f8a3db271e735b2",
    },
    "tokens.txt": {
        "size": 120,
        "sha256": "654625c18ef0f0583441a3b811c20904f2cf30febb716f32e14810b8e6bb715c",
    },
}

_MAX_WINDOW_SECONDS = 8.0
_MIN_WINDOW_SECONDS = 0.5
_START_TIMEOUT_SECONDS = 240.0
_INFERENCE_TIMEOUT_SECONDS = 120.0
_MAX_RESPONSE_BYTES = 1_000_000
_STOP = object()


class EmotionRuntimeError(RuntimeError):
    """The local emotion runtime or model could not complete inference."""

    def __init__(self, message: str, error_code: str = "runtime_error") -> None:
        super().__init__(message)
        self.error_code = error_code


class EmotionCancelled(EmotionRuntimeError):
    """Inference was cancelled and the isolated worker was stopped."""

    def __init__(self, message: str = "emotion inference was cancelled") -> None:
        super().__init__(message, error_code="cancelled")


def _base_model_metadata() -> dict[str, Any]:
    return {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weights_sha256": WEIGHTS_SHA256,
        "device": "cpu",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_emotion_assets(model_dir: os.PathLike[str] | str, runtime_dir: os.PathLike[str] | str) -> dict[str, Any]:
    """Check locally prepared files without loading a model or using a network.

    The 1.1 GB weight file is checked by size and the hash recorded in the
    preparation manifest. The preparation helper computes its SHA-256 before
    writing that manifest; this fast readiness check deliberately does not
    re-read the whole file on every application startup.
    """

    model_path = Path(model_dir).expanduser().resolve()
    runtime_path = Path(runtime_dir).expanduser().resolve()
    result: dict[str, Any] = {
        "ready": False,
        "reason": "model assets are not prepared",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weights_sha256": WEIGHTS_SHA256,
        "device": "cpu",
    }
    try:
        manifest_path = model_path / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
            result["reason"] = "model preparation manifest does not match the pinned model revision"
            return result
        files = manifest.get("files")
        if not isinstance(files, dict):
            result["reason"] = "model preparation manifest has no file inventory"
            return result
        for name, expected in EXPECTED_FILES.items():
            entry = files.get(name)
            path = model_path / name
            if not isinstance(entry, dict) or not path.is_file():
                result["reason"] = f"required model file is missing: {name}"
                return result
            if (
                entry.get("size") != expected["size"]
                or entry.get("sha256") != expected["sha256"]
                or path.stat().st_size != expected["size"]
            ):
                result["reason"] = f"required model file is incomplete or does not match the pinned manifest: {name}"
                return result

        lock_path = Path(__file__).resolve().parents[1] / "requirements-tone-lock.txt"
        runtime_manifest_path = runtime_path / RUNTIME_MANIFEST_NAME
        runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
        lock_sha = _sha256_file(lock_path)
        if runtime_manifest.get("requirements_sha256") != lock_sha:
            result["reason"] = "isolated runtime does not match requirements-tone-lock.txt"
            return result
        expected_packages = runtime_manifest.get("packages")
        if not isinstance(expected_packages, dict) or not expected_packages:
            result["reason"] = "isolated runtime manifest has no package inventory"
            return result
        installed = _installed_versions(runtime_path)
        for name, version in expected_packages.items():
            if installed.get(name.lower()) != version:
                result["reason"] = f"isolated runtime package is missing or has the wrong version: {name}"
                return result

        result.update(
            {
                "ready": True,
                "reason": "pinned local model files and isolated runtime are present",
                "model_dir": str(model_path),
                "runtime_dir": str(runtime_path),
                "runtime_packages": expected_packages,
            }
        )
        return result
    except FileNotFoundError as exc:
        result["reason"] = f"required local preparation manifest is missing: {Path(exc.filename).name}"
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        result["reason"] = f"local preparation metadata could not be read: {type(exc).__name__}"
    return result


def _installed_versions(runtime_dir: Path) -> dict[str, str]:
    """Read wheel metadata only; this does not import any installed package."""

    versions: dict[str, str] = {}
    for metadata in runtime_dir.glob("*.dist-info/METADATA"):
        name = None
        version = None
        try:
            with metadata.open("r", encoding="utf-8", errors="replace") as source:
                for line in source:
                    if line.startswith("Name: "):
                        name = line[6:].strip().lower().replace("_", "-")
                    elif line.startswith("Version: "):
                        version = line[9:].strip()
                    if name and version:
                        break
        except OSError:
            continue
        if name and version:
            versions[name] = version
    return versions


class EmotionRecognizer:
    """A reusable, isolated local CPU emotion model worker."""

    def __init__(
        self,
        model_dir: os.PathLike[str] | str,
        runtime_dir: os.PathLike[str] | str,
        python_executable: os.PathLike[str] | str = sys.executable,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.runtime_dir = Path(runtime_dir).expanduser().resolve()
        self.python_executable = str(Path(python_executable).expanduser().resolve())
        self.worker_script = Path(__file__).resolve().parents[1] / "tools" / "emotion_worker.py"
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[Any] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._closed = False
        self._model_metadata: dict[str, Any] = _base_model_metadata()

        readiness = check_emotion_assets(self.model_dir, self.runtime_dir)
        if not readiness["ready"]:
            raise EmotionRuntimeError(str(readiness["reason"]))
        if not self.worker_script.is_file():
            raise EmotionRuntimeError("emotion worker script is missing")
        if not Path(self.python_executable).is_file():
            raise EmotionRuntimeError("configured Python executable is missing")

    @property
    def model_metadata(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._model_metadata)

    def recognize(
        self,
        audio_path: os.PathLike[str] | str,
        start: float,
        end: float,
        cancel_event: Any = None,
    ) -> dict[str, Any]:
        path = Path(audio_path).expanduser().resolve()
        start_value = self._finite_time(start, "start")
        end_value = self._finite_time(end, "end")
        duration = end_value - start_value
        if not path.is_file():
            raise EmotionRuntimeError("emotion input audio file is missing")
        if start_value < 0 or duration <= 0:
            raise EmotionRuntimeError("emotion window must have increasing non-negative bounds")
        if duration < _MIN_WINDOW_SECONDS:
            raise EmotionRuntimeError(
                "emotion window is too short to classify; no model label was guessed",
                error_code="short_audio",
            )
        if duration > _MAX_WINDOW_SECONDS:
            raise EmotionRuntimeError(f"emotion window exceeds the {_MAX_WINDOW_SECONDS:g}-second model limit")
        if self._is_cancelled(cancel_event):
            self._stop_worker(kill=True)
            raise EmotionCancelled("emotion inference was cancelled")

        request = {
            "cmd": "predict",
            "path": str(path),
            "start": start_value,
            "end": end_value,
        }
        with self._lock:
            if self._closed:
                raise EmotionRuntimeError("emotion recognizer is closed")
            try:
                self._ensure_worker(cancel_event)
                self._send_request(request)
                response = self._wait_response(cancel_event, _INFERENCE_TIMEOUT_SECONDS)
                return self._validate_result(response)
            except EmotionCancelled:
                self._stop_worker(kill=True)
                raise
            except EmotionRuntimeError as exc:
                if exc.error_code != "short_audio":
                    self._stop_worker(kill=True)
                raise
            except Exception as exc:
                details = self._stderr_summary()
                self._stop_worker(kill=True)
                suffix = f"; worker output: {details}" if details else ""
                raise EmotionRuntimeError(f"emotion worker failed: {type(exc).__name__}{suffix}") from exc

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            if process is None:
                return
            try:
                if process.poll() is None and process.stdin is not None:
                    process.stdin.write('{"cmd":"shutdown"}\n')
                    process.stdin.flush()
                process.wait(timeout=5.0)
            except (OSError, subprocess.TimeoutExpired):
                self._terminate_process(process)
            finally:
                self._release_process_handles(process)

    def __enter__(self) -> "EmotionRecognizer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def _finite_time(value: float, name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise EmotionRuntimeError(f"emotion {name} bound must be numeric") from exc
        if not math.isfinite(number):
            raise EmotionRuntimeError(f"emotion {name} bound must be finite")
        return number

    @staticmethod
    def _is_cancelled(cancel_event: Any) -> bool:
        return bool(cancel_event is not None and callable(getattr(cancel_event, "is_set", None)) and cancel_event.is_set())

    def _ensure_worker(self, cancel_event: Any) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if self._process is not None:
            self._stop_worker(kill=True)

        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "-1",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "NUMEXPR_NUM_THREADS": "2",
                "MODELSCOPE_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        command = [
            self.python_executable,
            "-X",
            "utf8",
            "-B",
            str(self.worker_script),
            "--model-dir",
            str(self.model_dir),
            "--runtime-dir",
            str(self.runtime_dir),
        ]
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                command,
                cwd=str(self.worker_script.parents[1]),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
            )
        except OSError as exc:
            raise EmotionRuntimeError(f"could not start isolated emotion worker: {exc}") from exc

        self._process = process
        self._responses = queue.Queue()
        self._stderr_tail.clear()
        self._stdout_thread = threading.Thread(target=self._read_stdout, args=(process, self._responses), daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, args=(process,), daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        response = self._wait_response(cancel_event, _START_TIMEOUT_SECONDS)
        if not response.get("ok") or response.get("event") != "ready":
            reason = response.get("error") or "worker did not report a ready state"
            raise EmotionRuntimeError(str(reason))
        model = response.get("model")
        if not isinstance(model, dict) or model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
            raise EmotionRuntimeError("worker reported unexpected model metadata")
        if model.get("weights_sha256") != WEIGHTS_SHA256 or model.get("device") != "cpu":
            raise EmotionRuntimeError("worker reported unexpected model weights or device")
        self._model_metadata = dict(model)

    def _send_request(self, request: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            raise EmotionRuntimeError("emotion worker is not running")
        process.stdin.write(json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n")
        process.stdin.flush()

    def _wait_response(self, cancel_event: Any, timeout: float) -> dict[str, Any]:
        responses = self._responses
        if responses is None:
            raise EmotionRuntimeError("emotion worker response reader is unavailable")
        deadline = time.monotonic() + timeout
        while True:
            if self._is_cancelled(cancel_event):
                self._stop_worker(kill=True)
                raise EmotionCancelled("emotion inference was cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EmotionRuntimeError(f"emotion worker timed out after {timeout:g} seconds")
            try:
                line = responses.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            if line is _STOP:
                details = self._stderr_summary()
                suffix = f": {details}" if details else ""
                raise EmotionRuntimeError(f"emotion worker exited before responding{suffix}")
            if not isinstance(line, str) or len(line.encode("utf-8", errors="replace")) > _MAX_RESPONSE_BYTES:
                raise EmotionRuntimeError("emotion worker returned an invalid or oversized response")
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EmotionRuntimeError("emotion worker returned malformed JSON") from exc
            if not isinstance(response, dict):
                raise EmotionRuntimeError("emotion worker returned a non-object response")
            if response.get("ok") is False:
                message = response.get("error") or "unspecified worker error"
                error_code = response.get("error_code") or "worker_error"
                raise EmotionRuntimeError(str(message), error_code=str(error_code))
            return response

    def _validate_result(self, response: dict[str, Any]) -> dict[str, Any]:
        labels = response.get("labels")
        scores = response.get("scores")
        model = response.get("model")
        if not isinstance(labels, list) or not isinstance(scores, list) or len(labels) != len(LABELS) or len(scores) != len(LABELS):
            raise EmotionRuntimeError("model did not return the official nine emotion classes")
        if tuple(labels) != LABELS:
            raise EmotionRuntimeError("model returned labels in an unexpected order")
        finite_scores: list[float] = []
        for value in scores:
            try:
                score = float(value)
            except (TypeError, ValueError) as exc:
                raise EmotionRuntimeError("model returned a non-numeric score") from exc
            if not math.isfinite(score):
                raise EmotionRuntimeError("model returned a non-finite score")
            finite_scores.append(score)
        if not isinstance(model, dict):
            raise EmotionRuntimeError("model metadata is missing from the worker response")
        expected = self._model_metadata
        for key in ("id", "revision", "weights_sha256", "device"):
            if model.get(key) != expected.get(key):
                raise EmotionRuntimeError(f"worker response model metadata mismatch: {key}")
        return {"labels": list(labels), "scores": finite_scores, "model": dict(model)}

    @staticmethod
    def _read_stdout(process: subprocess.Popen[str], responses: queue.Queue[Any]) -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                responses.put(line.rstrip("\r\n"))
        except (OSError, ValueError):
            pass
        finally:
            responses.put(_STOP)

    def _read_stderr(self, process: subprocess.Popen[str]) -> None:
        try:
            assert process.stderr is not None
            for line in process.stderr:
                clipped = line.rstrip("\r\n")[-1000:]
                if clipped:
                    self._stderr_tail.append(clipped)
        except (OSError, ValueError):
            pass

    def _stderr_summary(self) -> str:
        return " | ".join(self._stderr_tail)

    def _stop_worker(self, *, kill: bool) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        if process.poll() is None:
            if not kill and process.stdin is not None:
                try:
                    process.stdin.write('{"cmd":"shutdown"}\n')
                    process.stdin.flush()
                except OSError:
                    kill = True
            try:
                process.wait(timeout=2.0 if kill else 5.0)
            except subprocess.TimeoutExpired:
                self._terminate_process(process)
        self._release_process_handles(process)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                    process.wait(timeout=2.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass

    def _release_process_handles(self, process: subprocess.Popen[str]) -> None:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=1.0)
        self._stdout_thread = None
        self._stderr_thread = None
        self._responses = None

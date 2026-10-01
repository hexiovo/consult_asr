"""Offline CPU worker for the pinned emotion2vec model.

The parent process owns lifecycle and cancellation. This child owns model
imports, the resident checkpoint, and inference over one local audio window at
a time. Stdout is reserved for newline-delimited JSON responses.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any


MODEL_ID = "iic/emotion2vec_plus_base"
MODEL_REVISION = "58354f1e5b6c118cf8b175b5b1955d6b07df42c2"
WEIGHTS_SHA256 = "60710b5aae1dbe69bdac8920028fb05882d4314fd09031922b4b61ee9e7aadbd"
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
MAX_WINDOW_SECONDS = 8.0
MIN_WINDOW_SECONDS = 0.5


class WorkerError(RuntimeError):
    def __init__(self, message: str, error_code: str = "inference_error") -> None:
        super().__init__(message)
        self.error_code = error_code


def _protocol_writer():
    """Keep a private protocol fd, then redirect all normal stdout to stderr."""

    protocol_fd = os.dup(sys.stdout.fileno())
    protocol = os.fdopen(protocol_fd, "w", encoding="utf-8", buffering=1)
    try:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    except OSError:
        pass
    sys.stdout = sys.stderr
    return protocol


def _emit(protocol, message: dict[str, Any]) -> None:
    protocol.write(json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n")
    protocol.flush()


def _disable_network() -> dict[str, Any]:
    """Make accidental network access fail closed inside this child only."""

    attempts = {"count": 0}

    def blocked(*args, **kwargs):
        attempts["count"] += 1
        raise RuntimeError("network access is disabled in the emotion worker")

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.socket.sendto = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    return attempts


def _manifest(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "emotion2vec.manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("model_id") != MODEL_ID or data.get("revision") != MODEL_REVISION:
        raise RuntimeError("local model manifest does not match the pinned emotion2vec revision")
    files = data.get("files", {})
    expected = {
        "model.pt": (1_118_245_678, WEIGHTS_SHA256),
        "config.yaml": (3_170, "50674ec187406b4f1208925594ceaf1f28f16a22333b5f2ac5b3d0cbffdd670e"),
        "configuration.json": (343, "8b6a745213025c7d4565f9f074cfef1b5cd5ef76e38e2a8f7f8a3db271e735b2"),
        "tokens.txt": (120, "654625c18ef0f0583441a3b811c20904f2cf30febb716f32e14810b8e6bb715c"),
    }
    for name, (size, digest) in expected.items():
        entry = files.get(name, {})
        asset = model_dir / name
        if not asset.is_file() or asset.stat().st_size != size:
            raise RuntimeError(f"required model asset is absent or incomplete: {name}")
        if entry.get("size") != size or entry.get("sha256") != digest:
            raise RuntimeError(f"local model manifest has an unexpected hash: {name}")
    return data


def _unwrap_checkpoint(checkpoint: Any) -> dict[str, Any]:
    state = checkpoint
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    if not isinstance(state, dict):
        raise RuntimeError("emotion2vec checkpoint has no state dictionary")
    return state


def _audit_state_dict(checkpoint: Any, model, scope_map: Any) -> dict[str, Any]:
    """Fail before loading if any model tensor would remain randomly initialized."""

    source = _unwrap_checkpoint(checkpoint)
    target = model.state_dict()
    scopes = list(scope_map or []) + ["module.", "None"]
    probes: dict[str, Any] = {}
    missing: list[str] = []
    mismatched: list[str] = []
    matched = 0

    for target_key, target_tensor in target.items():
        source_key = target_key
        for index in range(0, len(scopes), 2):
            src_prefix = str(scopes[index])
            dst_prefix = str(scopes[index + 1]) if index + 1 < len(scopes) else "None"
            src_prefix = "" if src_prefix.lower() == "none" else src_prefix
            dst_prefix = "" if dst_prefix.lower() == "none" else dst_prefix
            candidate = None
            if dst_prefix == "" and (src_prefix + target_key) in source:
                candidate = src_prefix + target_key
            elif target_key.startswith(dst_prefix):
                replacement = target_key.replace(dst_prefix, src_prefix, 1)
                if replacement in source:
                    candidate = replacement
            if candidate is not None:
                source_key = candidate

        source_tensor = source.get(source_key)
        if source_tensor is None:
            missing.append(target_key)
            continue
        if not hasattr(source_tensor, "shape") or tuple(source_tensor.shape) != tuple(target_tensor.shape):
            mismatched.append(target_key)
            continue
        matched += 1
        if target_key in {"proj.weight", "proj.bias"} or target_key.startswith("blocks.0."):
            probes[target_key] = (source_tensor, source_key)

    if missing or mismatched:
        examples = (missing[:3] + mismatched[:3])
        raise RuntimeError(
            "emotion2vec checkpoint does not fully match the model; "
            f"missing={len(missing)}, shape_mismatches={len(mismatched)}, examples={examples}"
        )
    if not getattr(model, "proj", None):
        raise RuntimeError("emotion2vec classification head is absent")
    if int(model.proj.out_features) != len(LABELS):
        raise RuntimeError("emotion2vec classification head does not have nine outputs")
    if not {"proj.weight", "proj.bias"}.issubset(probes):
        raise RuntimeError("classification head tensors were not found in the checkpoint")
    if not any(key.startswith("blocks.0.") for key in probes):
        raise RuntimeError("encoder tensors were not found in the checkpoint")
    return {
        "source_state": source,
        "probes": probes,
        "matched_tensor_count": matched,
        "model_tensor_count": len(target),
    }


def _load_model(model_dir: Path, runtime_dir: Path, network_attempts: dict[str, int]):
    # These are set before importing FunASR/Torch to avoid CUDA initialization
    # and model-hub auto-discovery in this process.
    for name, value in {
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
    }.items():
        os.environ[name] = value

    sys.path.insert(0, str(runtime_dir))
    import torch

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(0)

    import funasr.auto.auto_model as auto_model_module
    from funasr import AutoModel

    original_loader = auto_model_module.load_pretrained_model
    audit_result: dict[str, Any] = {}
    model_file = (model_dir / "model.pt").resolve()
    original_torch_load = torch.load

    def audited_loader(**kwargs):
        weight_path = Path(kwargs["path"]).resolve()
        model = kwargs["model"]
        if weight_path != model_file:
            kwargs["ignore_init_mismatch"] = False
            return original_loader(**kwargs)

        # FunASR 1.1.5's emotion2vec inference path goes through
        # Emotion2vec.inference -> extract_features -> forward and never reads
        # AudioEncoder.decoder. The pinned published checkpoint omits only
        # that pretraining decoder. Drop this unused module before loading so
        # every remaining inference tensor must come from the checkpoint.
        audio_encoder = model.modality_encoders["AUDIO"]
        decoder = getattr(audio_encoder, "decoder", None)
        unused_decoder_keys = []
        if decoder is not None:
            unused_decoder_keys = [
                "modality_encoders.AUDIO.decoder." + key
                for key in decoder.state_dict().keys()
            ]
            audio_encoder.decoder = None

        audit_holder: dict[str, Any] = {}

        def capture_checkpoint(source_path, *args, **load_kwargs):
            checkpoint = original_torch_load(source_path, *args, **load_kwargs)
            if Path(source_path).resolve() == model_file:
                audit_holder.update(_audit_state_dict(checkpoint, model, kwargs.get("scope_map")))
            return checkpoint

        torch.load = capture_checkpoint
        kwargs["ignore_init_mismatch"] = False
        try:
            result = original_loader(**kwargs)
        finally:
            torch.load = original_torch_load

        if not audit_holder:
            raise RuntimeError("FunASR did not load the pinned checkpoint through the audited path")
        probes = audit_holder.pop("probes")
        source_state = audit_holder.pop("source_state")
        loaded_state = model.state_dict()
        for target_key, (source_tensor, source_key) in probes.items():
            if not torch.equal(loaded_state[target_key].detach().cpu(), source_tensor.detach().cpu()):
                raise RuntimeError(f"checkpoint tensor was not loaded into the model: {target_key}")
        if not {"proj.weight", "proj.bias"}.issubset(probes):
            raise RuntimeError("classification head was not verified against the checkpoint")
        if not any(key.startswith("blocks.0.") for key in probes):
            raise RuntimeError("encoder was not verified against the checkpoint")
        audit_result.update(
            {
                "strict_shape_match": True,
                "matched_model_tensors": audit_holder["matched_tensor_count"],
                "model_tensors": audit_holder["model_tensor_count"],
                "classifier_tensors_from_checkpoint": True,
                "encoder_tensor_probe": True,
                "classifier_keys": ["proj.weight", "proj.bias"],
                "encoder_probe_keys": [key for key in probes if key.startswith("blocks.0.")][:2],
                "checkpoint_state_keys": len(source_state),
                "inference_unused_training_decoder_removed": decoder is not None,
                "inference_unused_training_decoder_tensor_count": len(unused_decoder_keys),
                "inference_unused_training_decoder_keys": unused_decoder_keys,
            }
        )
        del source_state
        return result

    auto_model_module.load_pretrained_model = audited_loader
    try:
        started = time.perf_counter()
        recognizer = AutoModel(
            model=str(model_dir),
            device="cpu",
            ncpu=2,
            batch_size=1,
            check_latest=False,
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
            log_level="ERROR",
            ignore_init_mismatch=False,
            trust_remote_code=False,
        )
        model_load_seconds = time.perf_counter() - started
    finally:
        auto_model_module.load_pretrained_model = original_loader

    if not audit_result.get("strict_shape_match"):
        raise RuntimeError("pinned checkpoint did not pass strict tensor compatibility checks")
    if network_attempts["count"]:
        raise RuntimeError("model runtime attempted network access during offline startup")

    model = recognizer.model
    if not getattr(model, "proj", None) or int(model.proj.out_features) != len(LABELS):
        raise RuntimeError("loaded model has no verified nine-class classifier")

    labels_path = model_dir / "tokens.txt"
    raw_labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    normalized = [_normalize_label(label) for label in raw_labels]
    if tuple(normalized) != LABELS:
        raise RuntimeError("local token list does not match the official nine-label order")

    metadata = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weights_sha256": WEIGHTS_SHA256,
        "device": "cpu",
        "funasr_version": "1.1.5",
        "load_seconds": round(model_load_seconds, 3),
        "checkpoint_audit": audit_result,
        "network_guard_active": True,
        "network_attempts": network_attempts["count"],
        "embeddings_exported": False,
    }
    return recognizer, metadata, torch


def _normalize_label(value: str) -> str:
    label = value.strip()
    if label == "<unk>":
        return "unknown"
    if "/" in label:
        label = label.rsplit("/", 1)[-1].strip()
    return label.lower()


def _read_window(audio_path: str, start: float, end: float):
    import numpy as np
    import soundfile as sf

    path = Path(audio_path)
    if not path.is_file() or path.suffix.lower() not in {".wav", ".flac", ".ogg", ".aiff", ".aif"}:
        raise RuntimeError("emotion input must be an existing local audio file")
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise RuntimeError("emotion window bounds are invalid")
    seconds = end - start
    if seconds < MIN_WINDOW_SECONDS:
        raise WorkerError("emotion window is too short to classify; no model label was guessed", "short_audio")
    if seconds > MAX_WINDOW_SECONDS:
        raise RuntimeError(f"emotion window exceeds the {MAX_WINDOW_SECONDS:g}-second model limit")

    info = sf.info(str(path))
    if info.samplerate != 16000 or info.channels != 1:
        raise RuntimeError("emotion input must be normalized to 16 kHz mono before inference")
    start_sample = int(round(start * info.samplerate))
    end_sample = int(round(end * info.samplerate))
    if start_sample >= info.frames or end_sample > info.frames:
        raise RuntimeError("emotion window falls outside the local audio file")
    audio, sample_rate = sf.read(
        str(path),
        start=start_sample,
        stop=end_sample,
        dtype="float32",
        always_2d=False,
    )
    if sample_rate != 16000 or not isinstance(audio, np.ndarray) or audio.ndim != 1 or audio.size == 0:
        raise RuntimeError("emotion input window could not be decoded as mono 16 kHz audio")
    if not np.isfinite(audio).all():
        raise RuntimeError("emotion input window contains non-finite audio samples")
    return np.ascontiguousarray(audio, dtype=np.float32)


def _predict(recognizer, request: dict[str, Any], torch) -> dict[str, Any]:
    audio = _read_window(str(request["path"]), float(request["start"]), float(request["end"]))
    started = time.perf_counter()
    with torch.inference_mode():
        result = recognizer.generate(
            input=audio,
            fs=16000,
            granularity="utterance",
            extract_embedding=False,
            output_dir=None,
            disable_pbar=True,
        )
    elapsed = time.perf_counter() - started
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
        raise RuntimeError("emotion model returned an unexpected result structure")

    raw_labels = result[0].get("labels")
    raw_scores = result[0].get("scores")
    if not isinstance(raw_labels, list) or not isinstance(raw_scores, list) or len(raw_labels) != len(raw_scores):
        raise RuntimeError("emotion model returned incomplete labels or scores")
    mapped: dict[str, float] = {}
    for raw_label, raw_score in zip(raw_labels, raw_scores):
        label = _normalize_label(str(raw_label))
        score = float(raw_score)
        if label not in LABELS or not math.isfinite(score):
            raise RuntimeError("emotion model returned an unknown label or non-finite score")
        if label in mapped:
            raise RuntimeError("emotion model returned duplicate normalized labels")
        mapped[label] = score
    if set(mapped) != set(LABELS):
        raise RuntimeError("emotion model did not return all official nine classes")
    return {
        "ok": True,
        "labels": list(LABELS),
        "scores": [mapped[label] for label in LABELS],
        "model": dict(_MODEL_METADATA),
        "inference_seconds": round(elapsed, 3),
        "audio_seconds": round(len(audio) / 16000.0, 3),
    }


def main() -> int:
    protocol = _protocol_writer()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--runtime-dir", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    runtime_dir = Path(args.runtime_dir).expanduser().resolve()
    recognizer = None
    torch = None
    global _MODEL_METADATA
    _MODEL_METADATA = {"id": MODEL_ID, "revision": MODEL_REVISION, "weights_sha256": WEIGHTS_SHA256, "device": "cpu"}

    try:
        _manifest(model_dir)
        runtime_manifest = json.loads((runtime_dir / "emotion-runtime.manifest.json").read_text(encoding="utf-8"))
        if runtime_manifest.get("funasr_version") != "1.1.5":
            raise RuntimeError("isolated FunASR runtime version does not match the verified pin")
        network_attempts = _disable_network()
        recognizer, metadata, torch = _load_model(model_dir, runtime_dir, network_attempts)
        _MODEL_METADATA = metadata
        _emit(protocol, {"ok": True, "event": "ready", "model": metadata})
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        _emit(protocol, {"ok": False, "event": "error", "error": f"model startup failed: {type(exc).__name__}: {exc}"})
        return 1

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            command = request.get("cmd")
            if command == "shutdown":
                _emit(protocol, {"ok": True, "event": "closed"})
                break
            if command != "predict":
                raise ValueError("unsupported worker command")
            response = _predict(recognizer, request, torch)
            _emit(protocol, response)
        except BaseException as exc:
            traceback.print_exc(file=sys.stderr)
            _emit(
                protocol,
                {
                    "ok": False,
                    "event": "error",
                    "error_code": getattr(exc, "error_code", "inference_error"),
                    "error": f"inference failed: {type(exc).__name__}: {exc}",
                },
            )
    try:
        if recognizer is not None and getattr(recognizer, "model", None) is not None:
            del recognizer.model
        if torch is not None:
            del torch
    finally:
        protocol.close()
    return 0


_MODEL_METADATA: dict[str, Any] = {}


if __name__ == "__main__":
    raise SystemExit(main())

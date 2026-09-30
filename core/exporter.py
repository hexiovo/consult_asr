from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from docx import Document


_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_stem(value: str) -> str:
    stem = _INVALID_FILENAME.sub("_", value).strip(" .")
    if not stem:
        stem = "咨询记录"
    if stem.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        stem = f"_{stem}"
    return stem[:120]


def _available_stem(directory: Path, requested: str) -> str:
    base = _safe_stem(requested)
    candidate = base
    suffix = 2
    while (directory / f"{candidate}.docx").exists() or (directory / f"{candidate}.json").exists():
        candidate = f"{base} ({suffix})"
        suffix += 1
    return candidate


def _stamp(seconds: float) -> str:
    whole = max(0, int(seconds))
    return f"{whole // 60:02d}:{whole % 60:02d}"


def merge_adjacent_turns(words: list[dict[str, Any]], max_gap_seconds: float = 2.0) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for word in sorted(words, key=lambda item: (float(item["start"]), float(item["end"]))):
        item = {
            "start": float(word["start"]),
            "end": float(word["end"]),
            "text": str(word.get("text", "")),
            "speaker_id": word.get("speaker_id"),
            "uncertain": bool(word.get("uncertain", False)),
        }
        if (
            merged
            and merged[-1]["speaker_id"] == item["speaker_id"]
            and item["start"] - merged[-1]["end"] <= max_gap_seconds
        ):
            merged[-1]["end"] = max(merged[-1]["end"], item["end"])
            merged[-1]["text"] += item["text"]
            merged[-1]["uncertain"] = merged[-1]["uncertain"] or item["uncertain"]
        else:
            merged.append(item)
    return merged


def write_exports(
    *,
    words: list[dict[str, Any]],
    output_dir: str | os.PathLike[str],
    filename: str,
    source_name: str,
    duration_seconds: float | None,
    status: str,
    role_labels: Mapping[str, str] | None = None,
    roles_confirmed: bool = False,
) -> tuple[Path, Path]:
    """Write a Word transcript and a timestamped JSON sidecar without clobbering files."""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stem = _available_stem(directory, filename)
    word_path = directory / f"{stem}.docx"
    json_path = directory / f"{stem}.json"
    temp_word: Path | None = None
    temp_json: Path | None = None
    installed: list[Path] = []

    speaker_names = {"speaker_1": "说话人1", "speaker_2": "说话人2"}
    if role_labels:
        speaker_names.update({str(key): str(value) for key, value in role_labels.items() if value})
    turns = merge_adjacent_turns(words)
    doc = Document()
    doc.add_heading("咨询记录", level=1)
    doc.add_paragraph(f"音频：{Path(source_name).name}")
    if duration_seconds is not None:
        doc.add_paragraph(f"时长：{_stamp(duration_seconds)}")
    if not turns:
        doc.add_paragraph("未检测到语音。")
    for turn in turns:
        speaker_id = turn["speaker_id"]
        speaker = speaker_names.get(speaker_id, "说话人（待确认）")
        marker = " [待核对]" if turn["uncertain"] else ""
        timestamp = f"[{_stamp(turn['start'])}–{_stamp(turn['end'])}] "
        doc.add_paragraph(f"{timestamp}{speaker}：{turn['text']}{marker}")

    json_words = [
        {
            "start": float(word["start"]),
            "end": float(word["end"]),
            "text": str(word.get("text", "")),
            "speaker_id": word.get("speaker_id"),
            "speaker": speaker_names.get(word.get("speaker_id"), "说话人（待确认）"),
            "uncertain": bool(word.get("uncertain", False)),
        }
        for word in sorted(words, key=lambda item: (float(item["start"]), float(item["end"])))
    ]
    json_turns = [
        {
            **turn,
            "speaker": speaker_names.get(turn["speaker_id"], "说话人（待确认）"),
        }
        for turn in turns
    ]
    payload = {
        "schema_version": 1,
        "status": status,
        "source_file": Path(source_name).name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_seconds": duration_seconds,
        "roles_confirmed": bool(roles_confirmed),
        "speakers": speaker_names,
        "word_count": len(json_words),
        "words": json_words,
        "segments": json_turns,
    }

    try:
        with tempfile.NamedTemporaryFile(prefix=".consult-asr-", suffix=".docx", dir=directory, delete=False) as file:
            temp_word = Path(file.name)
        doc.save(temp_word)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".consult-asr-", suffix=".json", dir=directory, delete=False
        ) as file:
            temp_json = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")

        # Outputs are newly chosen names; roll back the first if the second move fails.
        os.replace(temp_word, word_path)
        installed.append(word_path)
        os.replace(temp_json, json_path)
        installed.append(json_path)
        return word_path, json_path
    except Exception:
        for path in installed:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        for path in (temp_word, temp_json):
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

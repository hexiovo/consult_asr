"""Sentence presentation shared by the preview and paired exports."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


_SENTENCE_END = re.compile(r'[。！？!?]+[”’"」』）】]*|(?<!\d)\.(?!\d)[”’"」』）】]*|\n+')
_LABELS_ZH = {
    "angry": "愤怒", "disgusted": "厌恶", "fearful": "害怕",
    "happy": "开心", "neutral": "平静", "other": "其他",
    "sad": "悲伤", "surprised": "惊讶", "unknown": "无法判定",
}


def _merged_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _sentence_emotion(sentence: Mapping[str, Any], windows: list[dict[str, Any]], enabled: bool | None) -> dict[str, Any]:
    unknown = {"status": "unknown", "label": "unknown", "model_score": None}
    if enabled is False:
        return {**unknown, "status": "disabled", "reason": "emotion_disabled"}
    if sentence["uncertain"] or sentence["speaker_id"] not in {"speaker_1", "speaker_2"}:
        return {**unknown, "reason": "uncertain_word_or_speaker"}
    intervals = _merged_intervals(sentence["_intervals"])
    candidates = []
    for window in windows:
        if window.get("speaker_id") != sentence["speaker_id"] or window.get("status") != "ok":
            continue
        if window.get("analysis_safe") is False or window.get("label") in {None, "unknown"}:
            continue
        try:
            start, end = float(window["start"]), float(window["end"])
            score = float(window["model_score"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(value) for value in (start, end, score)) or end <= start:
            continue
        overlap = sum(max(0.0, min(end, right) - max(start, left)) for left, right in intervals)
        if overlap > 0:
            candidates.append((overlap, start, end, score, window))
    # Every spoken word interval must be covered; a reliable neighbouring window
    # must never conceal a failed, overlapping or unassigned part of this sentence.
    covered = _merged_intervals([(item[1], item[2]) for item in candidates])
    for start, end in intervals:
        cursor = start
        for left, right in covered:
            if right <= cursor:
                continue
            if left > cursor + 1e-6:
                break
            cursor = max(cursor, right)
            if cursor >= end - 1e-6:
                break
        if cursor < end - 1e-6:
            return {**unknown, "reason": "incomplete_emotion_coverage"}
    if not candidates:
        return {**unknown, "reason": "no_reliable_emotion_window"}
    # Keep the actual score of the most overlapping window, never invent a
    # sentence probability by averaging uncalibrated model outputs.
    _, start, end, score, window = max(candidates, key=lambda item: (item[0], -item[1]))
    return {"status": "ok", "label": str(window["label"]), "model_score": score,
            "source_window": {"start": start, "end": end},
            "method": "same_speaker_largest_word_overlap_window"}


def build_sentences(words: list[dict[str, Any]], emotion_windows: list[dict[str, Any]] | None = None,
                    tone_enabled: bool | None = None) -> list[dict[str, Any]]:
    """Split at punctuation, speaker changes and gaps over two seconds.

    Fragments inside one ASR word retain that word's original time bounds;
    character-level timestamps are not fabricated.
    """
    sentences: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def finish() -> None:
        nonlocal current
        if current is not None and current["text"].strip():
            current["text"] = current["text"].strip()
            current["emotion"] = _sentence_emotion(current, emotion_windows or [], tone_enabled)
            current.pop("_intervals")
            sentences.append(current)
        current = None

    for word in sorted(words, key=lambda item: (float(item["start"]), float(item["end"]))):
        start, end = float(word["start"]), float(word["end"])
        speaker = word.get("speaker_id")
        text = str(word.get("text", ""))
        if not text.strip():
            continue
        if current is not None and (current["speaker_id"] != speaker or start - current["end"] > 2.0):
            finish()
        offset = 0
        parts = []
        for boundary in _SENTENCE_END.finditer(text):
            parts.append((text[offset:boundary.end()], True))
            offset = boundary.end()
        if offset < len(text):
            parts.append((text[offset:], False))
        for fragment, terminal in parts:
            if current is None:
                current = {"start": start, "end": end, "speaker_id": speaker,
                           "text": "", "uncertain": False, "_intervals": []}
            current["end"] = max(current["end"], end)
            current["text"] += fragment
            current["uncertain"] |= bool(word.get("uncertain", False))
            current["_intervals"].append((start, end))
            if terminal:
                finish()
    finish()
    return sentences


def format_sentence(sentence: Mapping[str, Any]) -> str:
    emotion = sentence["emotion"]
    if emotion["status"] == "ok":
        label = _LABELS_ZH.get(emotion["label"], str(emotion["label"]))
        detail = f"可能情绪：{label}；置信度：{emotion['model_score']:.3f}"
    else:
        detail = "可能情绪：无法判定；置信度：—"
        if emotion["status"] == "disabled":
            detail += "；情绪分析已关闭"
    return f"{sentence['text']}（{detail}）"

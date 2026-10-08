"""Sentence presentation shared by the preview and Word export."""

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


def _sentence_acoustics(sentence: Mapping[str, Any], windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach an actual safe-window measurement, never average window statistics."""
    unknown = {"status": "unknown", "values": {}}
    if sentence["uncertain"] or sentence["speaker_id"] not in {"speaker_1", "speaker_2"}:
        return {**unknown, "reason": "uncertain_word_or_speaker"}
    intervals = _merged_intervals(sentence["_intervals"])
    candidates = []
    for window in windows:
        values = window.get("acoustics")
        if (
            window.get("speaker_id") != sentence["speaker_id"]
            or window.get("analysis_safe") is not True
            or not isinstance(values, Mapping)
            or values.get("analysis_status") == "invalid"
        ):
            continue
        try:
            start, end = float(window["start"]), float(window["end"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        overlap = sum(max(0.0, min(end, right) - max(start, left)) for left, right in intervals)
        if overlap > 0:
            candidates.append((overlap, start, end, values))
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
            return {**unknown, "reason": "incomplete_acoustic_coverage"}
    if not candidates:
        return {**unknown, "reason": "no_reliable_acoustic_window"}
    _, start, end, values = max(candidates, key=lambda item: (item[0], -item[1]))
    return {"status": "ok", "values": dict(values),
            "source_window": {"start": start, "end": end},
            "method": "same_speaker_largest_word_overlap_window"}


def build_sentences(words: list[dict[str, Any]], emotion_windows: list[dict[str, Any]] | None = None,
                    tone_enabled: bool | None = None,
                    analysis_windows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Split at punctuation, speaker changes and gaps over two seconds.

    Fragments inside one ASR word retain that word's original time bounds;
    character-level timestamps are not fabricated.
    """
    return _build_sentences(words, emotion_windows, tone_enabled,
                            analysis_windows=analysis_windows, preserve_intervals=False)


def make_sentence_id(original_sentence_index: int, start: float, end: float) -> str:
    """Return a text-independent ID for one original sentence interval.

    The zero-based index disambiguates equal intervals. ``float.hex`` preserves
    the exact parsed timestamp values while remaining stable across text and
    punctuation corrections.
    """
    if (
        isinstance(original_sentence_index, bool)
        or not isinstance(original_sentence_index, int)
        or original_sentence_index < 0
    ):
        raise ValueError("original sentence index must be a non-negative integer")
    start_value, end_value = float(start), float(end)
    if not math.isfinite(start_value) or not math.isfinite(end_value) or end_value < start_value:
        raise ValueError("sentence ID timestamps must be finite and ordered")
    return f"sentence-{original_sentence_index:06d}-{start_value.hex()}-{end_value.hex()}"


def _sentence_fragments(text: str) -> list[str]:
    """Split corrected display text at the same punctuation as ASR sentences."""
    fragments: list[str] = []
    offset = 0
    for boundary in _SENTENCE_END.finditer(text):
        fragment = text[offset:boundary.end()].strip()
        if fragment:
            fragments.append(fragment)
        offset = boundary.end()
    tail = text[offset:].strip()
    if tail:
        fragments.append(tail)
    return fragments


def revise_sentences(
    words: list[dict[str, Any]],
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    emotion_windows: list[dict[str, Any]] | None = None,
    tone_enabled: bool | None = None,
    analysis_windows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Apply sentence-level edits and return display/export fragments.

    IDs are assigned before overrides from the original sentence's sorted
    index and timestamps. Every fragment produced by edited punctuation keeps
    that parent ``sentence_id`` and the original sentence ``start``/``end``;
    ``fragment_id`` distinguishes the fragments in the current revision.
    Timing is never divided at character boundaries.
    """
    base_sentences = _build_sentences(words, emotion_windows, tone_enabled,
                                     analysis_windows=analysis_windows, preserve_intervals=True)
    windows = emotion_windows or []
    output: list[dict[str, Any]] = []

    for original_index, source in enumerate(base_sentences):
        sentence_id = make_sentence_id(original_index, source["start"], source["end"])
        override = (overrides or {}).get(sentence_id)
        if override is not None and not isinstance(override, Mapping):
            raise TypeError(f"override for {sentence_id} must be a mapping")

        text_value = override.get("text", source["text"]) if override is not None else source["text"]
        text = "" if text_value is None else str(text_value)
        speaker_id = override.get("speaker_id", source["speaker_id"]) if override is not None else source["speaker_id"]
        speaker_changed = speaker_id != source["speaker_id"]

        for fragment_index, fragment in enumerate(_sentence_fragments(text)):
            if speaker_changed:
                emotion = {
                    "status": "unknown",
                    "label": "unknown",
                    "model_score": None,
                    "reason": "speaker_changed_in_override",
                }
            else:
                emotion = _sentence_emotion(
                    {
                        "uncertain": source["uncertain"],
                        "speaker_id": speaker_id,
                        "_intervals": source["_intervals"],
                    },
                    windows,
                    tone_enabled,
                )

            output.append(
                {
                    "sentence_id": sentence_id,
                    "fragment_id": f"{sentence_id}:{fragment_index:04d}",
                    "original_sentence_index": original_index,
                    "fragment_index": fragment_index,
                    "start": source["start"],
                    "end": source["end"],
                    "speaker_id": speaker_id,
                    "text": fragment,
                    "uncertain": source["uncertain"],
                    "emotion": emotion,
                    "acoustics": source["acoustics"],
                    "acoustics_from_original_speaker": speaker_changed,
                }
            )
    return output


def _build_sentences(
    words: list[dict[str, Any]],
    emotion_windows: list[dict[str, Any]] | None,
    tone_enabled: bool | None,
    *,
    analysis_windows: list[dict[str, Any]] | None,
    preserve_intervals: bool,
) -> list[dict[str, Any]]:
    sentences: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def finish() -> None:
        nonlocal current
        if current is not None and current["text"].strip():
            current["text"] = current["text"].strip()
            current["acoustics"] = _sentence_acoustics(current, analysis_windows or [])
            if not preserve_intervals:
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
    acoustic = sentence.get("acoustics", {})
    values = acoustic.get("values", {}) if acoustic.get("status") == "ok" else {}

    def numeric(key: str) -> float | None:
        value = values.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
        return None

    volume = numeric("rms_dbfs")
    pitch = numeric("f0_median_hz")
    rate = numeric("speech_rate_chinese_chars_per_min")
    count = numeric("pause_count")
    duration = numeric("pause_total_seconds")
    detail += "；音量：" + (f"{volume:.1f} dBFS" if volume is not None else "—")
    detail += "；音调/基频：" + (f"{pitch:.1f} Hz" if pitch is not None else "—")
    detail += "；语速：" + (f"{rate:.1f} 字/分钟" if rate is not None else "—")
    detail += "；停顿：" + (
        f"{int(count)} 次/{duration:.2f} 秒" if count is not None and duration is not None else "—"
    )
    contour = values.get("final_contour_candidate")
    contour_label = {"up": "上扬", "down": "下降", "flat": "平稳"}.get(contour, "—")
    delta = numeric("semitone_delta")
    if contour_label != "—" and delta is not None:
        contour_label += f" {delta:+.1f} 半音"
    detail += f"；音高走势：{contour_label}"
    if sentence.get("acoustics_from_original_speaker") and acoustic.get("status") == "ok":
        detail += "；声学数据沿用原音频"
    return f"{sentence['text']}（{detail}）"

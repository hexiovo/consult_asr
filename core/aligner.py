from __future__ import annotations

from collections import defaultdict
from typing import Any


def _annotation_turns(diarization: Any) -> list[tuple[float, float, str]]:
    if diarization is None or not hasattr(diarization, "itertracks"):
        return []
    turns = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start, end = float(segment.start), float(segment.end)
        if end > start:
            turns.append((start, end, str(speaker)))
    return sorted(turns)


def speaker_tracks(diarization: Any) -> list[tuple[float, float, str]]:
    """Return raw diarization intervals with the same stable IDs as align_words."""
    turns = _annotation_turns(diarization)
    speaker_first_seen: dict[str, float] = {}
    for start, _, speaker in turns:
        speaker_first_seen[speaker] = min(start, speaker_first_seen.get(speaker, start))
    speaker_order = sorted(speaker_first_seen, key=lambda name: (speaker_first_seen[name], name))
    speaker_ids = {name: f"speaker_{index + 1}" for index, name in enumerate(speaker_order)}
    return [(start, end, speaker_ids[speaker]) for start, end, speaker in turns]


def align_words(diarization: Any, words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign each ASR word to at most one speaker, retaining ambiguous words."""
    turns = speaker_tracks(diarization)
    speaker_first_seen: dict[str, float] = {}
    for start, _, speaker_id in turns:
        speaker_first_seen[speaker_id] = min(start, speaker_first_seen.get(speaker_id, start))
    speaker_order = sorted(speaker_first_seen, key=lambda speaker_id: (speaker_first_seen[speaker_id], speaker_id))

    aligned: list[dict[str, Any]] = []
    for item in words:
        start, end = float(item["start"]), float(item["end"])
        duration = end - start
        overlaps: dict[str, float] = defaultdict(float)
        for turn_start, turn_end, speaker_id in turns:
            overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
            if overlap:
                overlaps[speaker_id] += overlap

        ordered = sorted(overlaps.items(), key=lambda pair: (-pair[1], speaker_first_seen.get(pair[0], 0), pair[0]))
        speaker_id = None
        uncertain = bool(item.get("uncertain", False))
        if duration > 0 and ordered:
            winner, best_overlap = ordered[0]
            coverage = best_overlap / duration
            next_coverage = ordered[1][1] / duration if len(ordered) > 1 else 0.0
            speaker_id = winner
            uncertain = uncertain or coverage < 0.55 or next_coverage >= 0.25
        elif len(speaker_order) == 1:
            speaker_id = "speaker_1"
            uncertain = True
        else:
            uncertain = True

        aligned.append({
            "start": start,
            "end": end,
            "text": str(item.get("text", "")),
            "speaker_id": speaker_id,
            "uncertain": uncertain,
        })
    return sorted(aligned, key=lambda item: (item["start"], item["end"]))


def align_segments(diarization: Any, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compatibility adapter for older callers."""
    words = []
    for segment in segments:
        items = segment.get("words") or [segment]
        for item in items:
            start = item.get("start", segment.get("start"))
            end = item.get("end", segment.get("end"))
            if start is None or end is None:
                continue
            words.append({
                "start": start,
                "end": end,
                "text": item.get("word", item.get("text", segment.get("text", ""))),
            })
    return align_words(diarization, words)

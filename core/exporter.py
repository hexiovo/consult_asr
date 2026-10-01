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


_EMOTION_LABELS_ZH = {
    "angry": "愤怒",
    "disgusted": "厌恶",
    "fearful": "害怕",
    "happy": "开心",
    "neutral": "平静",
    "other": "其他",
    "sad": "悲伤",
    "surprised": "惊讶",
    "unknown": "未知",
}


def _emotion_label(label: str) -> str:
    return _EMOTION_LABELS_ZH.get(label.casefold(), label or "未知")


def _emotion_reason(reason: Any) -> str:
    return {
        "uncertain_word_or_speaker": "转写置信度或说话人归属不确定",
        "speaker_unassigned": "说话人未可靠归属",
        "speaker_overlap": "存在说话人重叠",
        "no_speaker_track": "当前时间没有说话人轨道覆盖",
        "speaker_alignment_mismatch": "转写词与说话人轨道不一致",
        "window_too_short": "可靠语音窗口过短",
        "emotion_runtime_failed": "情绪模型运行失败",
    }.get(str(reason), "当前窗口无法可靠分析")


def _acoustic_summary(acoustic: Mapping[str, Any]) -> list[str]:
    parts: list[str] = []
    rate = acoustic.get("speech_rate_chinese_chars_per_min")
    if isinstance(rate, (int, float)):
        count = acoustic.get("chinese_character_count")
        duration = acoustic.get("denominator_seconds")
        if isinstance(count, int) and isinstance(duration, (int, float)):
            parts.append(f"语速（ASR估计）{float(rate):.1f} 个中文字/分钟（{count} 字/{float(duration):.2f} 秒）")
        else:
            parts.append(f"语速（ASR估计）{float(rate):.1f} 个中文字/分钟")
    rms = acoustic.get("rms_dbfs")
    if isinstance(rms, (int, float)):
        parts.append(f"音量 RMS {float(rms):.1f} dBFS")
    pause_count, pause_total = acoustic.get("pause_count"), acoustic.get("pause_total_seconds")
    if isinstance(pause_count, int) and isinstance(pause_total, (int, float)):
        parts.append(f"内部停顿 {pause_count} 次/{float(pause_total):.2f} 秒")
    pitch = acoustic.get("f0_median_hz")
    if isinstance(pitch, (int, float)):
        parts.append(f"音高基频中位数 {float(pitch):.1f} Hz")
    contour = acoustic.get("final_contour_candidate")
    if contour in {"up", "down", "flat"}:
        translated = {"up": "上扬", "down": "下降", "flat": "平稳"}[contour]
        delta = acoustic.get("semitone_delta")
        detail = f"（{float(delta):+.1f} 半音）" if isinstance(delta, (int, float)) else ""
        parts.append(f"句末音高走势候选：{translated}{detail}，不代表疑问句判断")
    return parts


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
    segments: list[dict[str, Any]] | None = None,
    emotion_windows: list[dict[str, Any]] | None = None,
    analysis_windows: list[dict[str, Any]] | None = None,
    tone_enabled: bool | None = None,
    tone_status: str | None = None,
    tone_error: str = "",
    tone_model: Mapping[str, Any] | None = None,
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
    turns = merge_adjacent_turns(words) if segments is None else [
        {
            "start": float(turn["start"]),
            "end": float(turn["end"]),
            "text": str(turn.get("text", "")),
            "speaker_id": turn.get("speaker_id"),
            "uncertain": bool(turn.get("uncertain", False)),
            **({"acoustics": turn["acoustics"]} if isinstance(turn.get("acoustics"), dict) else {}),
        }
        for turn in sorted(segments, key=lambda item: (float(item["start"]), float(item["end"])))
    ]
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

    has_extended_analysis = any(
        value is not None
        for value in (segments, emotion_windows, analysis_windows, tone_enabled, tone_status, tone_model)
    )
    if has_extended_analysis:
        doc.add_heading("语音特征与情绪倾向", level=2)
        doc.add_paragraph(
            "声学语速根据可靠窗口内的 ASR 文字估算；情绪标签表示模型给出的语音倾向，不用于诊断。"
            "模型置信分数为未校准的原始输出，不是咨询场景准确率或概率。句末音高走势只描述声学变化，"
            "不据此判定疑问句。"
        )
        feature_windows = analysis_windows if analysis_windows is not None else []
        if feature_windows:
            for window in feature_windows:
                speaker = speaker_names.get(window.get("speaker_id"), "说话人待确认")
                timestamp = f"[{_stamp(float(window['start']))}–{_stamp(float(window['end']))}] "
                acoustic = window.get("acoustics")
                parts = _acoustic_summary(acoustic) if isinstance(acoustic, dict) else []
                if parts:
                    doc.add_paragraph(f"{timestamp}{speaker}：" + "；".join(parts))
                elif not window.get("analysis_safe"):
                    doc.add_paragraph(
                        f"{timestamp}{speaker}：声学特征未知（{_emotion_reason(window.get('reason'))}）"
                    )
        else:
            # Keep compatibility with callers that still provide turn-level acoustics.
            for turn in turns:
                acoustic = turn.get("acoustics")
                if not isinstance(acoustic, dict):
                    continue
                parts = _acoustic_summary(acoustic)
                if parts:
                    speaker = speaker_names.get(turn.get("speaker_id"), "说话人（待确认）")
                    timestamp = f"[{_stamp(turn['start'])}–{_stamp(turn['end'])}] "
                    doc.add_paragraph(f"{timestamp}{speaker}：" + "；".join(parts))
        if tone_enabled is False:
            doc.add_paragraph("语音情绪分析：已关闭。")
        else:
            if tone_status in {"failed", "partial"}:
                status_text = "部分失败" if tone_status == "partial" else "失败"
                doc.add_paragraph(f"语音情绪分析：{status_text}。" + (tone_error.strip() or "模型运行失败。"))
            for window in emotion_windows or []:
                speaker = speaker_names.get(window.get("speaker_id"), "说话人待确认")
                timestamp = f"[{_stamp(float(window['start']))}–{_stamp(float(window['end']))}] "
                label = _emotion_label(str(window.get("label") or "unknown"))
                if window.get("status") == "ok" and isinstance(window.get("model_score"), (int, float)):
                    description = f"情绪倾向：{label}（模型置信分数 {float(window['model_score']):.3f}，未校准）"
                else:
                    description = f"情绪倾向：未知（{_emotion_reason(window.get('reason'))}）"
                doc.add_paragraph(f"{timestamp}{speaker}：{description}")

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
        {**turn, "speaker": speaker_names.get(turn.get("speaker_id"), "说话人（待确认）")}
        for turn in turns
    ]
    payload = {
        "schema_version": 2 if has_extended_analysis else 1,
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
    if has_extended_analysis:
        payload["tone_analysis"] = {
            "enabled": tone_enabled,
            "status": tone_status,
            "error": tone_error or None,
            "model": dict(tone_model or {}),
            "score_interpretation": "raw_uncalibrated_model_score",
            "limitations": "情绪标签仅表示模型给出的语音倾向，不用于诊断；分数不是咨询场景校准概率。",
        }
        payload["emotion_windows"] = [
            {
                **{key: value for key, value in window.items() if not str(key).startswith("_")},
                "speaker": speaker_names.get(window.get("speaker_id"), "说话人（待确认）"),
            }
            for window in (emotion_windows or [])
        ]
        payload["analysis_windows"] = [
            {
                **{key: value for key, value in window.items() if not str(key).startswith("_")},
                "speaker": speaker_names.get(window.get("speaker_id"), "说话人（待确认）"),
            }
            for window in (analysis_windows or [])
        ]

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

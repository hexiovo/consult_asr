"""Measured queue/stage ETA, independent of Qt and model implementations."""

from __future__ import annotations

import math
import statistics
import time
from typing import Callable


STAGE_LABELS = {
    "normalize": "音频准备", "asr": "语音识别", "diarization": "说话人区分",
    "acoustics": "声学分析", "emotion": "情绪分析", "cleanup": "资源释放",
    "asr_model_load": "识别模型加载", "diarization_model_load": "说话人模型加载",
}


def clock_text(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def approximate_time(seconds: float) -> str:
    if seconds < 60:
        return "不到 1 分钟"
    minutes = math.ceil(seconds / 60)
    hours, minutes = divmod(minutes, 60)
    return (f"约 {hours} 小时 {minutes} 分钟" if minutes else f"约 {hours} 小时") if hours else f"约 {minutes} 分钟"


class TimeEstimator:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.started: float | None = None
        self.durations: list[float | None] = []
        self.rates: list[float] = []
        self.completed: set[int] = set()
        self.current: int | None = None
        self.file_started = 0.0
        self.stage: str | None = None
        self.stage_started = 0.0
        self.fraction = 0.0
        self.stage_total: float | None = None
        self.finished_at: float | None = None

    def start(self, count: int) -> None:
        self.__init__(self.clock)
        self.started = self.clock()
        self.durations = [None] * count

    def configure(self, durations: list[float | None], rates: list[float]) -> None:
        self.durations = [value if value is not None and math.isfinite(value) and value > 0 else None
                          for value in durations]
        self.rates = [float(value) for value in rates
                      if isinstance(value, (int, float)) and math.isfinite(value) and value > 0][-5:]

    def progress(self, row: int, stage: str, fraction: float, audio_seconds: float) -> None:
        if self.started is None or not 0 <= row < len(self.durations) or row in self.completed:
            return
        now = self.clock()
        if self.current != row:
            self.current = row
            self.file_started = now
            self.stage = None
        if self.stage != stage:
            self.stage, self.stage_started = stage, now
            self.fraction, self.stage_total = 0.0, None
        if math.isfinite(audio_seconds) and audio_seconds > 0:
            self.durations[row] = audio_seconds
        if not math.isfinite(fraction):
            return
        fraction = min(1.0, max(0.0, fraction))
        if fraction <= self.fraction:
            return
        self.fraction = fraction
        elapsed = now - self.stage_started
        # Collect actual progress before extrapolating. Total remains fixed
        # between observations, so the display counts down instead of growing.
        if fraction >= .02 and elapsed >= 2 and fraction < 1:
            total = elapsed / fraction
            self.stage_total = total if self.stage_total is None else .5 * self.stage_total + .5 * total

    def complete(self, row: int, audio_seconds: float | None, elapsed: float, learn: bool) -> None:
        self.completed.add(row)
        if self.current == row:
            self.current, self.stage, self.stage_total = None, None, None
        if learn and audio_seconds is not None and math.isfinite(audio_seconds) and audio_seconds > 0 and math.isfinite(elapsed) and elapsed > 0:
            rate = elapsed / audio_seconds
            if math.isfinite(rate) and rate > 0:
                self.rates = (self.rates + [rate])[-5:]

    def finish(self) -> None:
        if self.finished_at is None:
            self.finished_at = self.clock()

    def elapsed(self) -> float:
        return max(0.0, (self.finished_at if self.finished_at is not None else self.clock()) - self.started) if self.started is not None else 0.0

    def text(self, cancelling: bool = False) -> str:
        elapsed = clock_text(self.elapsed())
        if self.finished_at is not None:
            return f"总用时 {elapsed} · {'处理已停止' if cancelling else '队列处理结束'}"
        if cancelling:
            return f"已用 {elapsed} · 正在安全停止，暂停剩余时间估计"
        parts = [f"已用 {elapsed}"]
        pending = [duration for index, duration in enumerate(self.durations) if index not in self.completed]
        current_elapsed = self.clock() - self.file_started if self.current is not None else 0.0
        estimate = statistics.median(self.rates) * sum(pending) - current_elapsed if self.rates and pending and all(value is not None for value in pending) else None
        if estimate is not None and estimate > 0:
            parts.append(f"预计队列剩余：{approximate_time(estimate)}（历史耗时估计）")
        else:
            parts.append("预计队列剩余：评估中" if estimate is None else "预计队列剩余：正在重新估计")
        if self.stage:
            remaining = self.stage_total - (self.clock() - self.stage_started) if self.stage_total is not None else None
            detail = approximate_time(remaining) if remaining is not None and remaining > 0 and self.fraction < 1 else "评估中"
            parts.append(f"{STAGE_LABELS.get(self.stage, self.stage)}剩余：{detail}")
        return " · ".join(parts)

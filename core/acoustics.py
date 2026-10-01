"""Small, streaming acoustic summaries for normalized 16 kHz mono audio.

The outputs describe measurable signal properties.  In particular, a phrase-
final pitch rise is only a contour candidate; it is not a question classifier.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.fft import irfft, next_fast_len, rfft
from numpy.lib.stride_tricks import sliding_window_view


_SAMPLE_RATE = 16_000
_BLOCK_FRAMES = 32_000  # 2 seconds of mono float32 audio
_FRAME_HOP = 160  # 10 ms
_ENERGY_WINDOW = 400  # 25 ms
_PITCH_WINDOW = 800  # 50 ms
_PITCH_HOP_MULTIPLIER = 2  # 20 ms pitch-frame spacing
_PAUSE_THRESHOLD_SECONDS = 0.25
_PITCH_MIN_HZ = 60.0
_PITCH_MAX_HZ = 450.0
_PITCH_MIN_CORRELATION = 0.80
_MIN_RELIABLE_PITCH_FRAMES = 5
_MIN_RELIABLE_PITCH_COVERAGE = 0.20
_CONTOUR_TAIL_SECONDS = 0.8
_CONTOUR_MIN_FRAMES_PER_HALF = 3
_CONTOUR_DIRECTION_SEMITONES = 1.0
_PITCH_HISTOGRAM_CENTS = 1
_PITCH_HISTOGRAM_SIZE = int(
    math.ceil(1200.0 * math.log2(_PITCH_MAX_HZ / _PITCH_MIN_HZ))
) + 2


class AcousticsCancelled(InterruptedError):
    """Raised when the caller requests cancellation during analysis."""


def count_chinese_chars(text: str) -> int:
    """Count CJK unified and compatibility ideographs in *text*.

    Latin letters, kana, punctuation, spaces, and digits do not contribute.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    total = 0
    for character in text:
        codepoint = ord(character)
        if (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0x20000 <= codepoint <= 0x2FA1F
            or 0x30000 <= codepoint <= 0x323AF
        ):
            total += 1
    return total


def _check_cancel(cancel_event: Any | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise AcousticsCancelled("Acoustic analysis cancelled")


def _base_result(text_count: int, duration_seconds: float) -> dict[str, Any]:
    rate = None
    if duration_seconds > 0.0 and text_count > 0:
        rate = float(text_count * 60.0 / duration_seconds)

    return {
        "analysis_status": "ok",
        "analysis_reason": None,
        "denominator_seconds": float(duration_seconds),
        "chinese_character_count": int(text_count),
        "speech_rate_chinese_chars_per_min": rate,
        "rms_dbfs": None,
        "rms_reason": None,
        "pause_threshold_seconds": float(_PAUSE_THRESHOLD_SECONDS),
        "pause_energy_threshold_dbfs": None,
        "pause_count": None,
        "pause_total_seconds": None,
        "pause_max_seconds": None,
        "pause_method": "short_time_energy_internal_runs",
        "f0_median_hz": None,
        "f0_coverage": 0.0,
        "f0_valid_frame_count": 0,
        "f0_frame_count": 0,
        "f0_reason": None,
        "final_contour_candidate": "unknown",
        "semitone_delta": None,
        "contour_coverage": 0.0,
        "contour_reason": None,
    }


def _invalid_result(
    text_count: int,
    duration_seconds: float,
    reason: str,
) -> dict[str, Any]:
    result = _base_result(text_count, duration_seconds)
    result["analysis_status"] = "invalid"
    result["analysis_reason"] = reason
    result["rms_reason"] = reason
    result["pause_count"] = None
    result["pause_total_seconds"] = None
    result["pause_max_seconds"] = None
    result["f0_reason"] = reason
    result["contour_reason"] = reason
    return result


def _iter_frames(
    audio_file: sf.SoundFile,
    start_frame: int,
    frame_count: int,
    cancel_event: Any | None,
):
    """Yield bounded batches of 50 ms windows at a 10 ms hop."""

    audio_file.seek(start_frame)
    remaining = frame_count
    read_offset = 0
    next_window_start = 0
    carry = np.empty(0, dtype=np.float32)

    while remaining > 0:
        _check_cancel(cancel_event)
        requested = min(_BLOCK_FRAMES, remaining)
        block = audio_file.read(requested, dtype="float32", always_2d=False)
        if block.size == 0:
            break
        if not np.isfinite(block).all():
            raise FloatingPointError("non-finite audio sample")

        carry_length = int(carry.size)
        if carry_length:
            data = np.concatenate((carry, block))
        else:
            data = block
        data_start = read_offset - carry_length
        local_start = next_window_start - data_start
        available = int(data.size) - local_start

        if available >= _PITCH_WINDOW:
            window_count = 1 + (available - _PITCH_WINDOW) // _FRAME_HOP
            windows = sliding_window_view(data[local_start:], _PITCH_WINDOW)
            frames = windows[::_FRAME_HOP][:window_count]
            yield next_window_start, frames
            next_window_start += int(window_count) * _FRAME_HOP

        next_local_start = next_window_start - data_start
        carry = data[next_local_start:].copy()
        read_offset += int(block.size)
        remaining -= int(block.size)


class _PauseTracker:
    def __init__(self, threshold_dbfs: float) -> None:
        self.threshold_dbfs = threshold_dbfs
        self.count = 0
        self.total_seconds = 0.0
        self.max_seconds = 0.0
        self.run_start: int | None = None
        self.run_end: int | None = None
        self.frame_count = 0

    def update(self, frame_index: int, energy_dbfs: float) -> None:
        is_silent = energy_dbfs <= self.threshold_dbfs
        self.frame_count = frame_index + 1
        if is_silent:
            if self.run_start is None:
                self.run_start = frame_index
            self.run_end = frame_index
            return

        self._finish_run(internal=True)

    def _finish_run(self, internal: bool) -> None:
        if self.run_start is None or self.run_end is None:
            return
        if internal and self.run_start > 0:
            duration = (
                (self.run_end - self.run_start) * _FRAME_HOP + _ENERGY_WINDOW
            ) / _SAMPLE_RATE
            if duration >= _PAUSE_THRESHOLD_SECONDS:
                self.count += 1
                self.total_seconds += duration
                self.max_seconds = max(self.max_seconds, duration)
        self.run_start = None
        self.run_end = None

    def finish(self) -> tuple[int, float, float]:
        # A run still active at EOF touches the trailing edge and is excluded.
        self._finish_run(internal=False)
        return self.count, self.total_seconds, self.max_seconds


def _parabolic_peak(acf: np.ndarray, index: int) -> float:
    if index <= 0 or index >= acf.size - 1:
        return float(index)
    left = float(acf[index - 1])
    center = float(acf[index])
    right = float(acf[index + 1])
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return float(index)
    offset = 0.5 * (left - right) / denominator
    return float(index + max(-0.5, min(0.5, offset)))


def _estimate_f0(frame: np.ndarray) -> float | None:
    samples = frame.astype(np.float64, copy=False)
    samples = samples - float(np.mean(samples))
    window = np.hanning(samples.size)
    windowed = samples * window
    if float(np.dot(windowed, windowed)) <= 1e-10:
        return None

    fft_size = next_fast_len(2 * samples.size)
    spectrum = rfft(windowed, n=fft_size)
    autocorrelation = irfft(spectrum * np.conjugate(spectrum), n=fft_size)[: samples.size]
    zero_lag = float(autocorrelation[0])
    if zero_lag <= 1e-12:
        return None
    autocorrelation = autocorrelation / zero_lag

    min_lag = max(1, int(math.floor(_SAMPLE_RATE / _PITCH_MAX_HZ)))
    max_lag = min(samples.size - 2, int(math.ceil(_SAMPLE_RATE / _PITCH_MIN_HZ)))
    if max_lag <= min_lag:
        return None

    lag_values = autocorrelation[min_lag : max_lag + 1]
    peak_mask = np.zeros(lag_values.size, dtype=bool)
    if lag_values.size >= 3:
        peak_mask[1:-1] = (lag_values[1:-1] >= lag_values[:-2]) & (
            lag_values[1:-1] >= lag_values[2:]
        )
    peak_positions = np.flatnonzero(peak_mask)
    if peak_positions.size == 0:
        peak_positions = np.asarray([int(np.argmax(lag_values))])

    peak_values = lag_values[peak_positions]
    strongest = float(np.max(peak_values))
    if strongest < _PITCH_MIN_CORRELATION:
        return None

    # Resolve near-equal harmonic peaks toward the shorter period.  A spectral
    # subharmonic check below corrects pure/strongly harmonic tones when the
    # fundamental is visibly present.
    near_best = peak_positions[peak_values >= strongest - 0.02]
    lag_index = min_lag + int(np.min(near_best))
    current_peak = float(autocorrelation[lag_index])
    magnitude = np.abs(spectrum)
    frequency_axis = np.fft.rfftfreq(fft_size, d=1.0 / _SAMPLE_RATE)
    band_mask = (frequency_axis >= _PITCH_MIN_HZ) & (frequency_axis <= _PITCH_MAX_HZ)
    band_max = float(np.max(magnitude[band_mask])) if np.any(band_mask) else 0.0

    while lag_index * 2 <= max_lag and band_max > 1e-12:
        subharmonic_index = lag_index * 2
        neighborhood_start = max(0, subharmonic_index - 1)
        neighborhood_end = min(autocorrelation.size, subharmonic_index + 2)
        subharmonic_peak = float(np.max(autocorrelation[neighborhood_start:neighborhood_end]))
        subharmonic_hz = _SAMPLE_RATE / subharmonic_index
        bin_center = int(round(subharmonic_hz * fft_size / _SAMPLE_RATE))
        spectral_start = max(0, bin_center - 1)
        spectral_end = min(magnitude.size, bin_center + 2)
        subharmonic_magnitude = float(np.max(magnitude[spectral_start:spectral_end]))
        if (
            subharmonic_peak >= current_peak - 0.05
            and subharmonic_magnitude >= band_max * 0.03
        ):
            lag_index = subharmonic_index
            current_peak = subharmonic_peak
        else:
            break

    refined_lag = _parabolic_peak(autocorrelation, lag_index)
    frequency = _SAMPLE_RATE / refined_lag
    if not math.isfinite(frequency) or not (_PITCH_MIN_HZ <= frequency <= _PITCH_MAX_HZ):
        return None
    return float(frequency)


def _median_from_histogram(histogram: np.ndarray, count: int) -> float | None:
    if count <= 0:
        return None
    cumulative = np.cumsum(histogram, dtype=np.int64)
    lower_rank = (count - 1) // 2
    upper_rank = count // 2
    lower_bin = int(np.searchsorted(cumulative, lower_rank + 1, side="left"))
    upper_bin = int(np.searchsorted(cumulative, upper_rank + 1, side="left"))
    median_bin = (lower_bin + upper_bin) / 2.0
    return float(_PITCH_MIN_HZ * (2.0 ** (median_bin / 1200.0)))


def _histogram_index(frequency: float) -> int:
    cents = 1200.0 * math.log2(frequency / _PITCH_MIN_HZ)
    index = int(round(cents / _PITCH_HISTOGRAM_CENTS))
    return max(0, min(_PITCH_HISTOGRAM_SIZE - 1, index))


def _contour_summary(
    tail_frames: deque[tuple[float, float | None]],
    segment_duration: float,
) -> tuple[str, float | None, float, str | None]:
    tail_start = max(0.0, segment_duration - _CONTOUR_TAIL_SECONDS)
    tail_end = segment_duration
    midpoint = (tail_start + tail_end) / 2.0
    early: list[float] = []
    late: list[float] = []
    valid_count = 0
    total_count = 0

    for time_seconds, frequency in tail_frames:
        total_count += 1
        if frequency is None:
            continue
        valid_count += 1
        if time_seconds < midpoint:
            early.append(frequency)
        else:
            late.append(frequency)

    coverage = float(valid_count / total_count) if total_count else 0.0
    if (
        len(early) < _CONTOUR_MIN_FRAMES_PER_HALF
        or len(late) < _CONTOUR_MIN_FRAMES_PER_HALF
    ):
        return "unknown", None, coverage, "insufficient_final_voiced_frames"
    if coverage < _MIN_RELIABLE_PITCH_COVERAGE:
        return "unknown", None, coverage, "low_final_voiced_frame_coverage"

    early_median = float(np.median(np.asarray(early, dtype=np.float64)))
    late_median = float(np.median(np.asarray(late, dtype=np.float64)))
    if early_median <= 0.0 or late_median <= 0.0:
        return "unknown", None, coverage, "invalid_final_pitch"
    semitone_delta = float(12.0 * math.log2(late_median / early_median))
    if semitone_delta >= _CONTOUR_DIRECTION_SEMITONES:
        candidate = "up"
    elif semitone_delta <= -_CONTOUR_DIRECTION_SEMITONES:
        candidate = "down"
    else:
        candidate = "flat"
    return candidate, semitone_delta, coverage, None


def analyze_acoustics(
    audio_path: str | Path,
    start: float,
    end: float,
    text: str = "",
    cancel_event: Any | None = None,
) -> dict[str, Any]:
    """Analyze one `[start, end)` segment of a normalized 16 kHz mono WAV.

    Unsupported sample rates/channels and out-of-range times raise
    ``ValueError``. Empty intervals and non-finite samples return the fixed
    schema with ``analysis_status='invalid'``. Cancellation raises
    ``AcousticsCancelled`` (an ``InterruptedError`` subclass) so the caller can
    map it to its processing-cancelled state. File I/O errors are allowed to
    propagate.
    """

    text_count = count_chinese_chars(text)
    try:
        start_seconds = float(start)
        end_seconds = float(end)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("start and end must be finite numbers") from exc
    if not math.isfinite(start_seconds) or not math.isfinite(end_seconds):
        raise ValueError("start and end must be finite numbers")
    if start_seconds < 0.0 or end_seconds < 0.0 or end_seconds < start_seconds:
        raise ValueError("segment times must satisfy 0 <= start <= end")

    _check_cancel(cancel_event)
    with sf.SoundFile(audio_path, mode="r") as audio_file:
        if audio_file.samplerate != _SAMPLE_RATE:
            raise ValueError(
                f"expected normalized {_SAMPLE_RATE} Hz audio, got {audio_file.samplerate} Hz"
            )
        if audio_file.channels != 1:
            raise ValueError(
                f"expected mono audio, got {audio_file.channels} channels"
            )

        start_frame = int(round(start_seconds * _SAMPLE_RATE))
        end_frame = int(round(end_seconds * _SAMPLE_RATE))
        if end_frame > audio_file.frames or start_frame > audio_file.frames:
            raise ValueError("segment times exceed the audio duration")
        sample_count = end_frame - start_frame
        duration_seconds = float(sample_count / _SAMPLE_RATE)
        if sample_count <= 0:
            return _invalid_result(text_count, duration_seconds, "empty_segment")

        # Pass one computes exact segment RMS and catches NaN/Inf before any
        # acoustic feature is accumulated. Only one two-second block is held.
        audio_file.seek(start_frame)
        remaining = sample_count
        sum_squares = 0.0
        while remaining > 0:
            _check_cancel(cancel_event)
            block = audio_file.read(
                min(_BLOCK_FRAMES, remaining), dtype="float32", always_2d=False
            )
            if block.size == 0:
                break
            if not np.isfinite(block).all():
                return _invalid_result(
                    text_count, duration_seconds, "non_finite_audio"
                )
            block64 = block.astype(np.float64, copy=False)
            sum_squares += float(np.dot(block64, block64))
            remaining -= int(block.size)

        if remaining != 0:
            raise OSError("audio file ended before the requested segment")

        rms = math.sqrt(sum_squares / sample_count)
        result = _base_result(text_count, duration_seconds)
        if rms > 0.0:
            global_rms_dbfs = float(20.0 * math.log10(rms))
            result["rms_dbfs"] = global_rms_dbfs
            silence_threshold_dbfs = max(-50.0, global_rms_dbfs - 35.0)
        else:
            global_rms_dbfs = None
            silence_threshold_dbfs = -50.0
            result["rms_reason"] = "digital_silence"
        result["pause_energy_threshold_dbfs"] = float(silence_threshold_dbfs)

        pause_tracker = _PauseTracker(silence_threshold_dbfs)
        histogram = np.zeros(_PITCH_HISTOGRAM_SIZE, dtype=np.int64)
        f0_valid_count = 0
        f0_frame_count = 0
        tail_frame_limit = int(math.ceil(_CONTOUR_TAIL_SECONDS * _SAMPLE_RATE / _FRAME_HOP))
        tail_frames: deque[tuple[float, float | None]] = deque(maxlen=tail_frame_limit)
        minimum_voiced_dbfs = silence_threshold_dbfs + 3.0

        # Pass two advances through bounded frame batches. Energy uses the
        # first 25 ms of each 50 ms frame at 10 ms intervals; pitch uses every
        # second frame, giving a 20 ms hop.
        try:
            for first_start, frames in _iter_frames(
                audio_file, start_frame, sample_count, cancel_event
            ):
                energy_samples = frames[:, :_ENERGY_WINDOW].astype(
                    np.float64, copy=False
                )
                mean_square = np.mean(energy_samples * energy_samples, axis=1)
                energy_dbfs = np.full(mean_square.shape, -np.inf, dtype=np.float64)
                positive = mean_square > 0.0
                energy_dbfs[positive] = 10.0 * np.log10(mean_square[positive])

                for local_index, frame_energy_dbfs in enumerate(energy_dbfs):
                    frame_index = first_start // _FRAME_HOP + local_index
                    pause_tracker.update(frame_index, float(frame_energy_dbfs))
                    if frame_index % _PITCH_HOP_MULTIPLIER != 0:
                        continue

                    f0_frame_count += 1
                    pitch_time = frame_index * _FRAME_HOP / _SAMPLE_RATE
                    frequency: float | None = None
                    if frame_energy_dbfs >= minimum_voiced_dbfs:
                        frequency = _estimate_f0(frames[local_index])
                    if frequency is not None:
                        f0_valid_count += 1
                        histogram[_histogram_index(frequency)] += 1
                    if pitch_time >= max(
                        0.0, duration_seconds - _CONTOUR_TAIL_SECONDS
                    ):
                        tail_frames.append((pitch_time, frequency))
        except FloatingPointError:
            return _invalid_result(text_count, duration_seconds, "non_finite_audio")

        pause_count, pause_total, pause_max = pause_tracker.finish()
        result["pause_count"] = int(pause_count)
        result["pause_total_seconds"] = float(pause_total)
        result["pause_max_seconds"] = float(pause_max)

        coverage = float(f0_valid_count / f0_frame_count) if f0_frame_count else 0.0
        result["f0_coverage"] = coverage
        result["f0_valid_frame_count"] = int(f0_valid_count)
        result["f0_frame_count"] = int(f0_frame_count)
        if f0_valid_count == 0:
            result["f0_reason"] = "no_reliable_voiced_frames"
        elif f0_valid_count < _MIN_RELIABLE_PITCH_FRAMES:
            result["f0_reason"] = "insufficient_voiced_frames"
        elif coverage < _MIN_RELIABLE_PITCH_COVERAGE:
            result["f0_reason"] = "low_voiced_frame_coverage"
        else:
            result["f0_median_hz"] = _median_from_histogram(
                histogram, f0_valid_count
            )
            result["f0_reason"] = None

        (
            contour_candidate,
            semitone_delta,
            contour_coverage,
            contour_reason,
        ) = _contour_summary(tail_frames, duration_seconds)
        result["final_contour_candidate"] = contour_candidate
        result["semitone_delta"] = semitone_delta
        result["contour_coverage"] = contour_coverage
        result["contour_reason"] = contour_reason

        _check_cancel(cancel_event)
        return result

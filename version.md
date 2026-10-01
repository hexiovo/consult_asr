# Version

Current version: **2.2.2**
Updated: 2026-10-01

## 2.2.2

- Show elapsed time, measured current-stage remaining time and a duration-weighted queue estimate using recent successful runs with the same device, precision, beam size and emotion setting.
- Report calibration/overrun as an estimate in progress, stop forecasting on cancellation, reset on new queues and freeze elapsed time after completion.
- Probe audio metadata in the worker and persist only five anonymous numerical processing-rate samples per setting; do not store recording names, paths or content in timing history.
- Verify ETA arithmetic and overrun handling with a controlled clock, synthetic audio metadata, ASR/pipeline/worker progress hooks, partial-failure/cancellation preservation, and the real offscreen Qt label/timer/layout without loading models or using private recordings.

## 2.2.1

- Split preview and Word transcripts into sentences and append possible emotion plus the original model confidence score in parentheses after each sentence.
- Match only same-speaker reliable emotion windows, require full coverage of spoken word intervals, retain unknown/disabled states, and keep scores uncalibrated rather than inventing sentence probabilities.
- Add structured sentences with annotated text and source-window timestamps to extended JSON exports while preserving existing words, segments, and raw analysis windows.

## 2.2.0

- Add optional local emotion2vec analysis, enabled by default in the GUI, with speaker-attributed windows no longer than six seconds, raw model scores, and explicit unknown/partial-failure states.
- Add Chinese speech rate, RMS level, internal pauses, median pitch, and a final pitch-contour candidate per reliable speaker window; pitch rise is described as an acoustic cue, not a question classifier.
- Use raw diarization tracks with word certainty to mask overlap, uncertain/unassigned words, alignment mismatches, and unsupported gaps locally; count each fully safe ASR word once when it crosses a window boundary, and keep short-window acoustics available when emotion inference is too short.
- Add an isolated CPU-only emotion runtime and pinned local model preparation without changing the existing application dependency lock.
- Extend the preview and paired Word/JSON export with feature summaries, model metadata, and score limitations while retaining schema version 1 for legacy export calls.
- Verify the pinned model with an offline CPU forward pass on synthetic mono audio and through the integrated pipeline/export path; the nine raw scores and model revision, weight hash, and CPU metadata matched expectations. `tools/prepare_models.py --verify` passed for the local ASR, pyannote, model, and runtime assets. These checks validate integration only, not emotion accuracy.
- Verify synthetic overlap, uncertain-word, track-covered and unsupported gaps, speaker-ID masking, six-second boundary counting, subsecond acoustics, offscreen preview, paired exports, and schema versions. The emotion checkbox only disables emotion inference; safe-window acoustics remain enabled. Independent review closed its single P1 window-attribution finding after one focused repair; no real-recording classification accuracy or hourly throughput claim is made.

## 2.1.0

- Refresh the PySide6 window with a two-column recording queue and transcript preview, persistent output-path field, compact status badges, and collapsed advanced recognition settings.
- Show per-file processing states, Chinese speaker labels, and disable result actions when no exportable result is selected.
- Verify the real Qt window at 1160×760 and 920×640 using only synthetic in-memory results and mocked model readiness; no audio, model weights, network, token, or GPU inference was used.

## 2.0.3

- Document the successful offline model-asset verification and local CPU initialization of `SpeakerDiarization` using the project's real loader.
- Record the scope boundary: no audio forward, real recording, GPU model inference, one-hour throughput, or transcription/speaker accuracy test was run; GPU compatibility acceptance remains blocked.
- Clarify that the `manifest` records resolved pyannote revisions, while the preparation source does not pin a fixed pyannote revision.
- Documentation and version record only; application source code was not changed.

## 2.0.2

- Load the local pyannote pipeline from its `config.yaml` and rewrite its segmentation/embedding references to explicit local `pytorch_model.bin` checkpoints.
- Verify that the pipeline YAML has the required mapping/name/params structure and that segmentation and embedding each point to a non-empty absolute checkpoint under the local `pyannote` model tree; malformed or empty YAML is reported as not ready.
- Tag invalid segmentation and embedding checkpoint diagnostics as pyannote configuration errors so the GUI keeps Start disabled for null or relative checkpoint values.
- Keep the hidden Hugging Face token prompt visible in the double-click launcher, preserve completed file results after a queue-level error, and retain the final completion summary after model-state refresh.
- Write rewritten YAML with ASCII escapes so the installed pyannote 3.1.1 loader can read configuration paths under Chinese Windows directories.
- Pin setuptools 80.9.0 in the isolated environment so pyannote.audio 3.1.1 can import its `pkg_resources` dependency.
- Validation uses installed-API mocks plus an empty/invalid pipeline configuration check without loading model weights. Gated pyannote weights remain unavailable until account access is authorized; no real model or GPU inference claim is made.

## 2.0.1

- Added an exact-version dependency snapshot for the validated Python 3.10.21 Windows environment, including the official PyTorch CUDA 12.4 wheel source.
- Tightened local-model completeness checks and prevent processing from starting until the ASR and diarization resources are present and their configured local weights can be found.
- Changed the initial and final queue status text to distinguish model readiness, completed files, failed files, and cancellations.
- Validation limits: the ASR weights are present, but pyannote access returned HTTP 401 and its local configuration is missing. CUDA is available according to runtime metadata, but GPU model loading and inference were not run.

## 2.0.0

- Replaced the Tkinter flow with a single-window PySide6 application for file/folder queues, background processing, progress, safe cancellation, transcript preview, and speaker-role confirmation.
- Added a local-only faster-whisper large-v3 Chinese ASR path with word timestamps, beam-size controls, and explicit missing-model errors.
- Added sequential ASR/pyannote loading, 16 kHz mono FFmpeg normalization, overlap-based speaker alignment, per-file outcomes, and safe paired Word/JSON export.
- Added an offline model preparation and verification tool, a separate dependency environment, and deployment instructions.
- Added a double-click pyannote preparation launcher with hidden, in-memory token input for models that require account access; it never accepts model terms automatically.
- Removed question-count role guessing and obsolete Tkinter post-processing modules.

Validation is tracked separately from the version description. No user recording with a corrected transcript is available, so this version makes no accuracy or one-hour performance claim.

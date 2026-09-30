# Version

Current version: **2.0.3**
Updated: 2026-09-30

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

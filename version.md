# Version

Current version: **2.5.0**
Updated: 2026-10-19

## 2.5.0

- Add a Windows 10/11 x64 deployment package with bundled Python 3.10, locked runtime dependencies, FFmpeg and public ASR/emotion models; target hosts need no preinstalled Python or Conda.
- Add a double-click installer that downloads fixed-version Release assets, resumes partial downloads, validates SHA-256 checksums, reassembles large archives and installs into a new directory without overwriting existing installations.
- Use relative launcher/runtime paths and relative pyannote checkpoint references so the application directory can move between locations. Preserve compatibility with existing valid absolute local checkpoints.
- Keep gated pyannote weights outside the public release; first-time setup prompts for an account-authorized Hugging Face token without saving it. Re-running setup retries model preparation without reinstalling the runtime.
- Add offline runtime/model diagnostics and deployment/rebuild instructions. Preserve the existing source/Conda launch path.
- Validate split archives, SHA-256 rejection, HTTP Range resume, existing-directory protection, model preparation references and a full install with host Python/Conda removed from PATH. Move the installed directory and pass all three EXE probes, synthetic CUDA ASR, offline CPU diarization and isolated CPU emotion inference (nine finite scores); these smoke checks do not measure recognition accuracy.

## 2.4.8

- 2026-10-08 maintenance: inspect generated caches and unused artifacts and prepare a one-time cleanup command. Agent deletion was blocked by the tool policy; cleanup remains pending manual execution. Preserve dependency lockfiles, model weights, runtime packages, launcher build/check sources and the PDF guide; no application version change.
- Fix the one-time cleanup script's CMD parsing by using Windows CRLF line endings; reproduce the LF failure with a harmless header-only probe and verify CRLF parsing without deleting project files.
- Present emotion, confidence, RMS volume, pitch/F0, speech rate, pauses and pitch contour together in the same sentence-ending parentheses in both preview and Word; remove the separate per-window feature listing.
- Associate acoustic values with the original speaker's safe window with the largest spoken-word overlap, require full reliable coverage, preserve source-window metadata and corrected-sentence timing, and show missing values as —.
- Verify matching sentence annotations across the preview, Word writer and export pipeline; cover speaker isolation, unavailable/disabled emotion, unsafe/partial/invalid/nonfinite acoustic data, multiple analysis windows, corrected punctuation and speaker overrides. Rebuild the local EXE and pass all three launcher checks.
- Synchronize the native manifest version with 2.4.8 and remove local IDE settings, Python bytecode and runtime logs from version control.
- Remove hardcoded Hugging Face credentials from early Git history before publishing master; the current configuration already contains no hardcoded token.

## 2.4.7

- Start every new application window with an empty queue, transcript preview and role selection instead of automatically restoring the previous session.
- Preserve exported Word files and update the user guide to reflect the new startup behavior.
- Reproduce the previous automatic restore with an isolated cached transcript; verify empty startup despite existing cache, reset correction/role/export controls, preserved saved files, new audio selection, and repeated blank launches. Rebuild the local EXE and pass all three launcher checks with an explicit empty-startup assertion.

## 2.4.6

- Let users select each Word filename and destination in a save dialog, including saving the current exported result under a new name; cancellation leaves its export state unchanged.
- Export only `.docx`, remove the JSON result action and paired-export requirement, and retain internal session recovery with compatibility for prior saved results.
- Keep source fingerprint validation, corrected transcript/roles/features, and atomic Word replacement; update the local user guide and launcher version.
- Verify custom names, batch Word-only exports, corrections/roles, no-speech documents, old/new session restore, cancellation, save-as, collision protection, failed replacement cleanup, confirmed replacement, and stale-source blocking. Rebuild and replace the local EXE; all three launcher checks pass.

## 2.4.5

- Show an animated native “正在启动，请稍候…” window before the EXE starts Python; close it after the main window is shown, using a per-launch ready event. Source launches also display a lightweight Qt startup window.
- Initialize audio playback devices on the first sentence playback instead of blocking application startup. In isolated offscreen checks with no restored session, main-window construction/display dropped from approximately 6–7.5 seconds to 0.19 seconds; first playback still initializes the devices.
- Give guide/log/status/advanced buttons a pale teal background, dark text, visible borders, and hover/pressed/focus/disabled states; use sage green for secondary actions and update guide-window action colors.
- Add a native startup-handshake check to the validated EXE replacement workflow.
- Verify the source startup handoff, lazy playback/reuse/failure handling, actual playback/stop with a generated silent WAV, guide/button colors and layout, Python syntax, and all three launcher checks; rebuild and replace the local EXE.

## 2.4.4

- Prevent sentence correction controls from collapsing at small window sizes or increased display scaling; scroll the review panel when its full content does not fit.
- Bind native Windows window icons and taskbar relaunch properties to the custom icon and launcher EXE rather than the child Python executable.
- Extend launcher checks to cover layout geometry and Windows window identity, and support validated replacement of an existing EXE.

## 2.4.3

- Add a native Windows EXE launcher for the existing local Python, GPU and model environment, with no console window or duplicated model assets.
- Add a custom multi-resolution jade/gold dialogue icon to the EXE, application windows and Windows taskbar.
- Add a reproducible local build and isolated offline Qt launch check; keep the application directory and validated runtime as required local resources.

## 2.4.2

- Add local diagnostic event logging with a fixed privacy whitelist, anonymous run/task identifiers, and bounded UTF-8 rotation.
- Add a main-window shortcut that opens the local log folder and reports unavailable or empty log states safely.
- Associate per-file processing, retries, stage completion, cancellation, and terminal outcomes with one anonymous task sequence.
- Coordinate concurrent application instances during writes and rotation; logger failures safely drop diagnostics without interrupting transcription or paired export.
- Document the log location, retention limit, and fields excluded from diagnostic records.

## 2.4.1

- Rename the recording selectors to “选择录音” and “选择文件夹” and explain that selecting a new batch replaces the current one.
- Ask before replacing a nonempty queue; cancel, Escape, or closing the dialog preserves the current queue, corrections, roles, session file, and pending editor text. Explicit replacement follows the existing batch-loading flow.
- Update README and USER_GUIDE to describe the in-app protection. The illustrated PDF remains the documented v2.3.2 historical edition.

## 2.4.0

- Add opt-in GPU batched ASR with VAD, batch-size fallback, and a bounded ASR CPU-thread setting; accurate serial inference remains the default. Separate ETA histories by inference mode, batch size, and CPU threads.
- Close failed lazy batch iterators, reset their speech-timestamp state, and begin OOM retries only after the failed exception frame is released. Release the ASR model on failure or cancellation before GPU cache cleanup.
- Compare one 76.76-second synthetic Chinese audio on an RTX 4060 Laptop GPU (8 GB), float16, beam 5, and 8 CPU threads: ASR inference took 5.94 seconds in accurate mode and 4.41 seconds with GPU batch 2 (25.6% less inference time, about 1.35x). Both runs returned 212 word entries and timestamp bounds of 0–75.64 seconds; text matched after removing punctuation and whitespace, while sentence punctuation/boundaries differed.
- Record the CUDA driver's approximately 3.2 GiB free-memory sample after model loading as a point-in-time reading, not peak usage. The benchmark retained only timestamp count and minimum/maximum bounds, not per-word arrays for cross-mode comparison.
- Validate OOM iterator close/state reset, traceback release before retry, failure/cancel model cleanup and temporary-directory removal with targeted mocks; AST parsing and `git diff --check` pass. The illustrated PDF remains the v2.3.2 historical edition; `USER_GUIDE.md` contains the current settings.

## 2.3.2

- Add an illustrated PDF user guide with actual Qt interface captures populated only with synthetic examples and explicitly labeled demonstration states.
- Add a local PDF entry in the existing guide window, retaining searchable text help and readable missing-file/open-failure messages.
- Verify the 16-page PDF with eight screenshots, embedded Chinese text, clickable contents/bookmarks and rendered page inspection; test the local PDF opener without private recordings or model inference.

## 2.3.1

- Add a complete Chinese user guide covering deployment, models, transcription, timing, playback/correction, roles, paired exports, recovery, local files and troubleshooting.
- Add an always-available UI guide button with a nonmodal local reader, chapter navigation and text search; read the same USER_GUIDE.md shipped with the project and show a recoverable message if it is missing.

## 2.3.0

- Add sentence-level transcript and speaker correction with original word/timestamp preservation, sentence-range playback from the source audio, and paired Word/JSON re-export of revisions.
- Persist completed results and pending queue state in a bounded local session file; restore only completed results, resume pending rows with their original queue mapping, and retain prior results when a source fingerprint changes.
- Detect missing or stale export pairs on restore and make completed results available for re-export; speaker overrides keep emotion unknown while acoustic summaries remain attached to the original audio windows.
- Validate restored result/window records and degrade an invalid queue row to pending without discarding other completed rows. Validate GUI session, row/ETA mapping, editor transitions, and asynchronous playback using synthetic data in an offscreen Qt window at 1280×800; no private recording, model inference, or GPU was used.

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

from __future__ import annotations

import os
import json
import math
import copy
from pathlib import Path
from threading import Event

from PySide6.QtCore import QThread, Qt, Signal, QUrl, QTimer, QPoint, QSettings, QStandardPaths, QEventLoop
from PySide6.QtGui import QColor, QDesktopServices, QFont, QFontDatabase, QIcon, QPainter, QPolygon
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QLayout,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.help_dialog import UserGuideDialog
from app.pipeline import ConsultationPipeline, FileAnalysis, _processing_profile
from config import (
    ASR_MODEL_DIR,
    DIARIZATION_MODEL_DIR,
    LOG_DIR,
    OUTPUT_DIR,
    TONE_DEFAULT_ENABLED,
    TONE_MODEL_DIR,
    TONE_RUNTIME_DIR,
)
from core.diarizer import Diarizer
from core.transcriber import Transcriber
from core.transcript import build_sentences, format_sentence, make_sentence_id, revise_sentences
from tools.prepare_models import verify_models
from utils.audio_utils import collect_audio_files, probe_audio_duration
from utils.result_store import AtomicJsonStore, ResultStoreError
from utils.time_estimator import TimeEstimator
from utils.logger import (
    begin_task,
    exception_fields,
    is_logging_enabled,
    log_event,
    LOG_FILE_NAME,
    init_logging,
    shutdown_logging,
)


_SESSION_SCHEMA_VERSION = 1
_RESTORABLE_STATUSES = {"ready", "no_speech", "exported", "export_failed"}
_MAX_SESSION_FILES = 2000
_MAX_SESSION_WORDS = 1_000_000


def _json_value(value, depth: int = 0):
    """Convert supported analysis values to bounded JSON primitives."""
    if depth > 24:
        raise ValueError("会话数据层级过深")
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("会话数据包含无效数值")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, depth + 1) for item in value]
    scalar = getattr(value, "item", None)
    if callable(scalar):
        return _json_value(scalar(), depth + 1)
    raise ValueError("会话数据包含不支持的内容")


def _valid_optional_number(value, *, minimum: float | None = None) -> bool:
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return minimum is None or value >= minimum


def _validate_window_records(windows, label: str) -> None:
    if not isinstance(windows, list) or len(windows) > _MAX_SESSION_WORDS:
        raise ValueError(f"缓存{label}无效")
    for window in windows:
        if not isinstance(window, dict):
            raise ValueError(f"缓存{label}无效")
        start, end = window.get("start"), window.get("end")
        if (
            not _valid_optional_number(start, minimum=0)
            or not _valid_optional_number(end, minimum=0)
            or start is None
            or end is None
            or end <= start
            or window.get("speaker_id") not in {None, "speaker_1", "speaker_2"}
            or not isinstance(window.get("analysis_safe"), bool)
            or window.get("status") not in {"pending", "disabled", "unknown", "ok"}
        ):
            raise ValueError(f"缓存{label}无效")
        if window.get("label") is not None and not isinstance(window.get("label"), str):
            raise ValueError(f"缓存{label}无效")
        if window.get("reason") is not None and not isinstance(window.get("reason"), str):
            raise ValueError(f"缓存{label}无效")
        if not _valid_optional_number(window.get("model_score")):
            raise ValueError(f"缓存{label}数值无效")
        labels = window.get("labels")
        if labels is not None and (
            not isinstance(labels, list) or any(not isinstance(item, str) for item in labels)
        ):
            raise ValueError(f"缓存{label}标签无效")
        scores = window.get("scores")
        if scores is not None and (
            not isinstance(scores, list)
            or any(not _valid_optional_number(item) or item is None for item in scores)
        ):
            raise ValueError(f"缓存{label}分数无效")
        acoustics = window.get("acoustics")
        if acoustics is not None and not isinstance(acoustics, dict):
            raise ValueError("缓存声学窗口无效")
        if isinstance(acoustics, dict):
            contour = acoustics.get("final_contour_candidate")
            if contour is not None and not isinstance(contour, str):
                raise ValueError("缓存声学窗口无效")
            for key in (
                "speech_rate_chinese_chars_per_min",
                "rms_dbfs",
                "pause_threshold_seconds",
                "pause_energy_threshold_dbfs",
                "pause_total_seconds",
                "pause_max_seconds",
                "f0_median_hz",
                "f0_coverage",
                "semitone_delta",
                "contour_coverage",
                "denominator_seconds",
            ):
                if key in acoustics and not _valid_optional_number(acoustics[key]):
                    raise ValueError("缓存声学数值无效")
        # Check every nested value too, including acoustic/model metadata fields
        # that the current preview does not display.
        _json_value(window)


def _current_fingerprint(path: Path) -> dict[str, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


class ProcessingWorker(QThread):
    progress = Signal(int, int, int, str)
    analysis_ready = Signal(int, object)
    timing_configured = Signal(object, str)
    stage_progress = Signal(int, str, float, float)

    def __init__(
        self,
        files,
        input_root,
        output_root,
        device_choice,
        compute_type,
        beam_size,
        tone_enabled=None,
        row_indices=None,
        inference_mode="accurate",
        batch_size=2,
        cpu_threads=8,
    ):
        super().__init__()
        self.files = list(files)
        self.row_indices = list(row_indices) if row_indices is not None else list(range(len(self.files)))
        if len(self.row_indices) != len(self.files):
            raise ValueError("row_indices must map every worker file to an existing queue row")
        self.input_root = Path(input_root).resolve() if input_root else None
        self.output_root = Path(output_root).resolve()
        self.device_choice = device_choice
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.tone_enabled = tone_enabled
        self.inference_mode = inference_mode
        self.batch_size = min(4, max(2, int(batch_size)))
        self.cpu_threads = min(8, max(1, int(cpu_threads)))
        self.cancel_event = Event()

    def cancel(self):
        self.cancel_event.set()

    def _device(self) -> str:
        if self.device_choice == "cpu":
            return "cpu"
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def run(self):
        completed_indices: set[int] = set()
        try:
            device = self._device()
            compute_type = self.compute_type
            if device == "cpu":
                compute_type = "int8" if compute_type == "int8_float16" else "float32"
            expected_mode = (
                "gpu_batch_vad"
                if self.inference_mode == "gpu_batch" and device == "cuda"
                else ("accurate_cuda_fallback" if self.inference_mode == "gpu_batch" else "accurate")
            )
            expected_batch = self.batch_size if expected_mode == "gpu_batch_vad" else 0
            durations = []
            for path in self.files:
                durations.append(None if self.cancel_event.is_set() else probe_audio_duration(path))
            profile = _processing_profile(
                device,
                compute_type,
                self.beam_size,
                self.tone_enabled,
                expected_mode,
                expected_batch,
                self.cpu_threads,
            )
            self.timing_configured.emit(durations, profile)
            pipeline = ConsultationPipeline(
                device=device,
                compute_type=compute_type,
                beam_size=self.beam_size,
                tone_enabled=self.tone_enabled,
                inference_mode=self.inference_mode,
                batch_size=self.batch_size,
                cpu_threads=self.cpu_threads,
            )
            if self.inference_mode == "gpu_batch" and device != "cuda":
                self.progress.emit(0, len(self.files), -1, "GPU不可用，当前使用准确模式继续处理。")
            total = len(self.files)
            for work_index, path in enumerate(self.files):
                row = self.row_indices[work_index]
                source = Path(path).resolve()
                if self.input_root:
                    try:
                        relative = source.relative_to(self.input_root)
                        destination = self.output_root / relative.parent
                    except ValueError:
                        destination = self.output_root
                else:
                    destination = self.output_root

                task_seq = begin_task()
                if self.cancel_event.is_set():
                    log_event("processing_start", task_seq=task_seq)
                    log_event("cancel", task_seq=task_seq, stage="queue")
                    log_event(
                        "terminal",
                        task_seq=task_seq,
                        outcome="cancelled",
                        stage="queue",
                    )
                    analysis = FileAnalysis(
                        source,
                        destination,
                        status="cancelled",
                        error="在开始前取消。",
                        diagnostic_task_seq=task_seq,
                    )
                    self.analysis_ready.emit(row, analysis)
                    completed_indices.add(work_index)
                    continue

                current = work_index + 1
                self.progress.emit(current, total, row, f"[{current}/{total}] 正在处理：{source.name}")
                initial_fingerprint = _current_fingerprint(source)
                analysis = pipeline.analyze_file(
                    source,
                    destination,
                    task_seq=task_seq,
                    cancel_event=self.cancel_event,
                    progress=lambda stage, i=current, t=total, r=row: self.progress.emit(
                        i, t, r, f"[{i}/{t}] {stage}"
                    ),
                    stage_progress=lambda stage, fraction, seconds, queue_row=row:
                        self.stage_progress.emit(queue_row, stage, fraction, seconds),
                )
                final_fingerprint = _current_fingerprint(source)
                if initial_fingerprint is not None:
                    analysis.source_size_bytes = initial_fingerprint["size_bytes"]
                    analysis.source_mtime_ns = initial_fingerprint["mtime_ns"]
                if initial_fingerprint != final_fingerprint:
                    analysis.source_changed = True
                self.analysis_ready.emit(row, analysis)
                completed_indices.add(work_index)
        except Exception as exc:
            # Unexpected queue-level errors are reported as failed items, never as success.
            queue_error_fields = exception_fields(exc)
            queue_error = f"处理队列失败：{type(exc).__name__}: {str(exc)[:500]}"
            del exc
            for work_index, path in enumerate(self.files):
                if work_index in completed_indices:
                    continue
                row = self.row_indices[work_index]
                task_seq = begin_task()
                log_event("processing_start", task_seq=task_seq)
                log_event(
                    "terminal",
                    task_seq=task_seq,
                    outcome="failed",
                    stage="queue",
                    **queue_error_fields,
                )
                try:
                    source = Path(path).resolve()
                except (OSError, RuntimeError, ValueError):
                    source = Path(str(path))
                analysis = FileAnalysis(
                    source,
                    self.output_root,
                    status="failed",
                    error=queue_error,
                    diagnostic_task_seq=task_seq,
                )
                self.analysis_ready.emit(row, analysis)
                completed_indices.add(work_index)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        application = QApplication.instance()
        if application is not None:
            application.setApplicationName("ConsultASR")
            application.setOrganizationName("ConsultASR")
            _configure_application_fonts(application)
        self.setWindowTitle("咨询录音转写")
        self.resize(1160, 760)
        self.setMinimumSize(900, 600)
        self.input_files: list[Path] = []
        self.input_root: Path | None = None
        self.analyses: list[FileAnalysis | None] = []
        self.worker: ProcessingWorker | None = None
        self._guide_dialog: UserGuideDialog | None = None
        self.role_choices: dict[str, str | None] = {}
        self._updating_role = False
        self._close_when_done = False
        self._session_error = ""
        self._session_notice = ""
        self._session_ready_count = 0
        self._session_root = Path(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
            or (Path.home() / ".local" / "share" / "ConsultASR")
        )
        self._session_path = self._session_root / "session.json"
        self._session_store = AtomicJsonStore(self._session_path)
        self._eta_index_by_row: dict[int, int] = {}
        self._active_worker_rows: list[int] = []
        self._stale_results: dict[int, FileAnalysis] = {}
        self._sentence_ids_by_row: dict[int, list[str]] = {}
        self._selected_sentence_id: str | None = None
        self._editor_row: int | None = None
        self._suppress_sentence_events = False
        self._media_player = None
        self._audio_output = None
        self._audio_unavailable_reason = ""
        self._media_source_path: Path | None = None
        self._pending_play_request: tuple[Path, str, int, int, int] | None = None
        self._play_generation = 0
        self._started_play_generation: int | None = None
        self._media_source_fingerprint: dict[str, int] | None = None
        self._time_estimator = TimeEstimator()
        self._timing_profile: str | None = None
        self._timing_settings = QSettings("ConsultASR", "LocalTiming")
        self._timing_timer = QTimer(self)
        self._timing_timer.setInterval(1000)
        self._timing_timer.timeout.connect(self._update_time_estimate)
        self.setStyleSheet(_window_stylesheet())

        central = QWidget()
        central.setObjectName("centralRoot")
        outer = QVBoxLayout(central)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(8)

        header = QHBoxLayout()
        title_stack = QVBoxLayout()
        title_stack.setSpacing(1)
        title = QLabel("咨询转写")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setWeight(QFont.Weight.DemiBold)
        title.setFont(title_font)
        subtitle = QLabel("本地处理 · 中文双人区分 · 录音不会上传")
        subtitle.setObjectName("mutedLabel")
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)
        header.addLayout(title_stack)
        header.addStretch(1)
        local_badge = QLabel("本地运行")
        local_badge.setObjectName("smallBadge")
        language_badge = QLabel("中文 · 双人")
        language_badge.setObjectName("smallBadge")
        header.addWidget(local_badge)
        header.addWidget(language_badge)
        self.guide_button = QPushButton("使用指南")
        self.guide_button.setObjectName("quietButton")
        self.guide_button.setToolTip("查看本地指南：部署、转录、校对、导出、恢复与故障排查。")
        self.guide_button.clicked.connect(self._show_user_guide)
        header.addWidget(self.guide_button)
        self.open_log_dir_button = QPushButton("打开日志文件夹")
        self.open_log_dir_button.setObjectName("quietButton")
        self.open_log_dir_button.setToolTip("打开本地诊断日志目录；日志未生成或不可用时会提示。")
        self.open_log_dir_button.clicked.connect(self._open_log_folder)
        header.addWidget(self.open_log_dir_button)
        outer.addLayout(header)

        readiness = QHBoxLayout()
        readiness.setSpacing(7)
        self.asr_ready_label = QLabel()
        self.diarization_ready_label = QLabel()
        self.tone_ready_label = QLabel()
        readiness.addWidget(self.asr_ready_label)
        readiness.addWidget(self.diarization_ready_label)
        readiness.addWidget(self.tone_ready_label)
        self.tone_enabled_check = QCheckBox("情绪倾向分析")
        self.tone_enabled_check.setChecked(TONE_DEFAULT_ENABLED)
        self.tone_enabled_check.setToolTip(
            "默认使用本地 emotion2vec 模型；取消勾选会关闭情绪倾向分析，但仍会继续转写和安全窗口声学特征分析。"
        )
        self.tone_enabled_check.toggled.connect(lambda _checked: self._refresh_model_state())
        readiness.addWidget(self.tone_enabled_check)
        readiness.addStretch(1)
        refresh_models = QPushButton("刷新状态")
        refresh_models.setObjectName("quietButton")
        refresh_models.clicked.connect(self._refresh_model_state)
        readiness.addWidget(refresh_models)
        outer.addLayout(readiness)

        self.choose_files_button = QPushButton("选择录音")
        self.choose_files_button.setObjectName("secondaryButton")
        self.choose_files_button.setToolTip("选择新录音批次会替换当前批次；有未导出的结果时请先保存 Word。")
        self.choose_files_button.clicked.connect(self._choose_files)
        self.choose_folder_button = QPushButton("选择文件夹")
        self.choose_folder_button.setObjectName("secondaryButton")
        self.choose_folder_button.setToolTip("选择新录音批次会替换当前批次；有未导出的结果时请先保存 Word。")
        self.choose_folder_button.clicked.connect(self._choose_folder)
        self.clear_button = QPushButton("清空队列")
        self.clear_button.setObjectName("quietButton")
        self.clear_button.clicked.connect(self._clear_queue)

        queue = QGroupBox("录音队列")
        queue.setObjectName("surfaceCard")
        queue_layout = QVBoxLayout(queue)
        queue_layout.setContentsMargins(12, 17, 12, 10)
        queue_layout.setSpacing(8)
        queue_header = QHBoxLayout()
        queue_header.addStretch(1)
        self.source_summary = QLabel("0 项")
        queue_header.addWidget(self.source_summary)
        queue_layout.addLayout(queue_header)
        source_row = QHBoxLayout()
        source_row.setSpacing(6)
        source_row.addWidget(self.choose_files_button)
        source_row.addWidget(self.choose_folder_button)
        source_row.addStretch(1)
        source_row.addWidget(self.clear_button)
        queue_layout.addLayout(source_row)

        self.queue_empty = QWidget()
        self.queue_empty.setObjectName("emptyState")
        self.queue_empty.setMinimumHeight(160)
        empty_layout = QVBoxLayout(self.queue_empty)
        empty_layout.setContentsMargins(18, 16, 18, 16)
        empty_layout.addStretch(1)
        empty_title = QLabel("还没有录音")
        empty_title.setObjectName("emptyTitle")
        empty_desc = QLabel("从上方添加音频文件或文件夹。\n选中的录音和处理状态会显示在这里。")
        empty_desc.setObjectName("mutedLabel")
        empty_desc.setWordWrap(True)
        empty_layout.addWidget(empty_title)
        empty_layout.addWidget(empty_desc)
        empty_layout.addStretch(1)
        queue_layout.addWidget(self.queue_empty, 1)

        self.table = QTableWidget(0, 3)
        self.table.setObjectName("queueTable")
        self.table.setHorizontalHeaderLabels(["音频", "状态", "说明"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(38)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnHidden(2, True)
        self.table.itemSelectionChanged.connect(self._show_selected_analysis)
        self.table.setMinimumHeight(160)
        self.table.hide()
        queue_layout.addWidget(self.table, 1)
        queue_note = QLabel("选择录音后，可在右侧查看逐字稿与角色确认。")
        queue_note.setObjectName("mutedLabel")
        queue_note.setWordWrap(True)
        queue_layout.addWidget(queue_note)

        review = QGroupBox("转写预览")
        review.setObjectName("surfaceCard")
        review_layout = QVBoxLayout(review)
        review_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        review_layout.setContentsMargins(12, 17, 12, 10)
        review_layout.setSpacing(8)
        review_header = QHBoxLayout()
        self.selected_file_label = QLabel("尚未选择录音")
        self.selected_file_label.setObjectName("mutedLabel")
        self.selected_file_label.setWordWrap(True)
        review_header.addWidget(self.selected_file_label, 1)
        review_header.addStretch(1)
        self.preview_state = QLabel("等待选择")
        review_header.addWidget(self.preview_state)
        review_layout.addLayout(review_header)

        example_row = QHBoxLayout()
        example_row.setSpacing(7)
        self.speaker1_example = QLabel("说话人1示例：处理后显示该说话人的示例发言。")
        self.speaker2_example = QLabel("说话人2示例：处理后显示该说话人的示例发言。")
        for label in (self.speaker1_example, self.speaker2_example):
            label.setWordWrap(True)
            label.setMinimumWidth(0)
            label.setObjectName("exampleCard")
        example_row.addWidget(self.speaker1_example, 1)
        example_row.addWidget(self.speaker2_example, 1)
        review_layout.addLayout(example_row)

        self.preview = QPlainTextEdit()
        self.preview.setObjectName("transcriptPreview")
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("处理完成后，这里会显示带时间和说话人标记的逐字稿。")
        self.preview.setMinimumHeight(112)
        review_layout.addWidget(self.preview, 1)

        correction_box = QGroupBox("逐句校对与回听")
        correction_box.setObjectName("correctionCard")
        self.correction_box = correction_box
        correction_layout = QVBoxLayout(correction_box)
        correction_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        correction_layout.setContentsMargins(8, 9, 8, 7)
        correction_layout.setSpacing(5)
        sentence_row = QHBoxLayout()
        self.sentence_combo = QComboBox()
        self.sentence_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.sentence_combo.setMinimumContentsLength(12)
        self.sentence_combo.setObjectName("sageCombo")
        self.sentence_combo.currentIndexChanged.connect(self._sentence_selected)
        sentence_row.addWidget(self.sentence_combo, 1)
        self.play_sentence_button = QPushButton("回听句子")
        self.play_sentence_button.setObjectName("secondaryButton")
        self.play_sentence_button.clicked.connect(self._play_selected_sentence)
        sentence_row.addWidget(self.play_sentence_button)
        self.stop_audio_button = QPushButton("停止")
        self.stop_audio_button.setObjectName("quietButton")
        self.stop_audio_button.clicked.connect(self._stop_playback)
        sentence_row.addWidget(self.stop_audio_button)
        correction_layout.addLayout(sentence_row)

        edit_row = QHBoxLayout()
        edit_row.addWidget(QLabel("说话人"))
        self.sentence_speaker_combo = QComboBox()
        self.sentence_speaker_combo.addItem("保持自动识别", None)
        self.sentence_speaker_combo.addItem("说话人 1", "speaker_1")
        self.sentence_speaker_combo.addItem("说话人 2", "speaker_2")
        self.sentence_speaker_combo.currentIndexChanged.connect(self._sentence_edit_changed)
        edit_row.addWidget(self.sentence_speaker_combo, 1)
        self.save_correction_button = QPushButton("保存校对")
        self.save_correction_button.setObjectName("secondaryButton")
        self.save_correction_button.setEnabled(False)
        self.save_correction_button.clicked.connect(self._save_sentence_correction)
        edit_row.addWidget(self.save_correction_button)
        correction_layout.addLayout(edit_row)

        self.sentence_editor = QPlainTextEdit()
        self.sentence_editor.setPlaceholderText("可修订当前句子的文字；原始识别文本和时间戳会保留。")
        self.sentence_editor.setMinimumHeight(64)
        self.sentence_editor.setMaximumHeight(96)
        self.sentence_editor.textChanged.connect(self._sentence_edit_changed)
        correction_layout.addWidget(self.sentence_editor)
        self.sentence_precision_note = QLabel(
            "回听按识别时间范围定位；同一识别词内拆出的句子可能共用时间范围。"
        )
        self.sentence_precision_note.setObjectName("mutedLabel")
        self.sentence_precision_note.setWordWrap(True)
        correction_layout.addWidget(self.sentence_precision_note)
        review_layout.addWidget(correction_box)

        role_row = QHBoxLayout()
        role_row.addWidget(QLabel("角色确认"))
        self.role_combo = _SageRoleComboBox()
        self.role_combo.setObjectName("sageCombo")
        self.role_combo.addItem("未确认：保留说话人1 / 说话人2", None)
        self.role_combo.addItem("确认：说话人1是咨询师", "speaker_1")
        self.role_combo.addItem("确认：说话人2是咨询师", "speaker_2")
        self.role_combo.currentIndexChanged.connect(self._role_changed)
        role_row.addWidget(self.role_combo, 1)
        review_layout.addLayout(role_row)

        result_actions = QHBoxLayout()
        result_actions.addStretch(1)
        self.open_word_button = QPushButton("打开 Word")
        self.open_word_button.setObjectName("secondaryButton")
        self.open_word_button.clicked.connect(lambda: self._open_selected("word"))
        self.save_button = QPushButton("保存 Word")
        self.save_button.setObjectName("primaryButton")
        self.save_button.setToolTip("选择文件名和位置，仅保存 Word；批量结果逐个选择。")
        self.save_button.clicked.connect(self._save_results)
        result_actions.addWidget(self.open_word_button)
        result_actions.addWidget(self.save_button)
        review_layout.addLayout(result_actions)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(7)
        queue.setMinimumWidth(300)
        self.review_scroll = QScrollArea()
        self.review_scroll.setObjectName("reviewScroll")
        self.review_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.review_scroll.setWidgetResizable(True)
        self.review_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.review_scroll.setMinimumWidth(420)
        self.review_scroll.setWidget(review)
        splitter.addWidget(queue)
        splitter.addWidget(self.review_scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        outer.addWidget(splitter, 1)

        output_row = QWidget()
        output_row.setObjectName("outputRow")
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(11, 7, 11, 7)
        output_layout.setSpacing(7)
        output_layout.addWidget(QLabel("保存到"))
        self.output_edit = QLineEdit(str(OUTPUT_DIR))
        self.output_edit.setToolTip("保存时默认打开此文件夹，可选择文件名和位置；仅保存 Word。")
        self.output_edit.editingFinished.connect(self._persist_session)
        output_layout.addWidget(self.output_edit, 1)
        choose_output = QPushButton("更改…")
        choose_output.setObjectName("secondaryButton")
        choose_output.clicked.connect(self._choose_output)
        output_layout.addWidget(choose_output)
        outer.addWidget(output_row)

        self.advanced_toggle = QPushButton("显示高级识别设置")
        self.advanced_toggle.setObjectName("quietButton")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        outer.addWidget(self.advanced_toggle, 0, Qt.AlignmentFlag.AlignLeft)
        self.advanced_panel = QWidget()
        self.advanced_panel.setObjectName("advancedPanel")
        advanced_layout = QGridLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(10, 3, 10, 6)
        advanced_layout.setHorizontalSpacing(8)
        advanced_layout.addWidget(QLabel("设备"), 0, 0)
        self.device_combo = _SageRoleComboBox()
        self.device_combo.setObjectName("sageCombo")
        self.device_combo.addItem("自动（优先使用可用 GPU）", "auto")
        self.device_combo.addItem("CPU", "cpu")
        advanced_layout.addWidget(self.device_combo, 0, 1)
        advanced_layout.addWidget(QLabel("识别方式"), 0, 2)
        self.compute_combo = _SageRoleComboBox()
        self.compute_combo.setObjectName("sageCombo")
        self.compute_combo.addItem("质量优先", "float16")
        self.compute_combo.addItem("速度优先", "int8_float16")
        advanced_layout.addWidget(self.compute_combo, 0, 3)
        advanced_layout.addWidget(QLabel("识别细致度"), 0, 4)
        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(1, 10)
        self.beam_spin.setValue(5)
        advanced_layout.addWidget(self.beam_spin, 0, 5)
        advanced_layout.addWidget(QLabel("推理模式"), 1, 0)
        self.inference_mode_combo = _SageRoleComboBox()
        self.inference_mode_combo.setObjectName("sageCombo")
        self.inference_mode_combo.addItem("准确优先（默认）", "accurate")
        self.inference_mode_combo.addItem("GPU批量加速（VAD分段）", "gpu_batch")
        self.inference_mode_combo.setToolTip(
            "GPU批量模式会使用VAD检测语音并改变分段边界及解码回退行为；保留beam=5和词级时间戳。"
            "显存不足时会从头降低batch，必要时退回原串行识别。准确优先模式保持现有VAD关闭路径。"
        )
        self.inference_mode_combo.currentIndexChanged.connect(self._refresh_performance_controls)
        advanced_layout.addWidget(self.inference_mode_combo, 1, 1, 1, 3)
        advanced_layout.addWidget(QLabel("批量大小"), 1, 4)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(2, 4)
        self.batch_size_spin.setValue(2)
        advanced_layout.addWidget(self.batch_size_spin, 1, 5)
        advanced_layout.addWidget(QLabel("ASR CPU线程"), 2, 0)
        self.cpu_threads_spin = QSpinBox()
        self.cpu_threads_spin.setRange(1, 8)
        self.cpu_threads_spin.setValue(8)
        self.cpu_threads_spin.setToolTip("限制 faster-whisper/CTranslate2 使用的CPU线程数，范围为1到8。")
        advanced_layout.addWidget(self.cpu_threads_spin, 2, 1)
        self.performance_hint = QLabel(
            "准确模式维持现有VAD关闭路径；ASR线程数限制在1到8。"
        )
        self.performance_hint.setObjectName("mutedLabel")
        self.performance_hint.setWordWrap(True)
        advanced_layout.addWidget(self.performance_hint, 2, 2, 1, 4)
        self.device_combo.currentIndexChanged.connect(self._refresh_performance_controls)
        self._refresh_performance_controls()
        self.advanced_panel.setVisible(False)
        outer.addWidget(self.advanced_panel)

        activity_row = QHBoxLayout()
        self.progress_label = QLabel("正在检查本地模型…")
        self.progress_label.setObjectName("mutedLabel")
        activity_row.addWidget(self.progress_label, 1)
        self.counts_label = QLabel("待处理 0  ·  完成 0  ·  失败 0  ·  已取消 0")
        self.counts_label.setObjectName("mutedLabel")
        activity_row.addWidget(self.counts_label)
        outer.addLayout(activity_row)
        self.time_estimate_label = QLabel("已用 00:00 · 预计剩余：开始处理后估计")
        self.time_estimate_label.setObjectName("mutedLabel")
        self.time_estimate_label.setWordWrap(True)
        self.time_estimate_label.setToolTip(
            "整队列估计参考相同设备与识别设置下最近五次成功处理的耗时和音频时长。"
            "首次运行显示当前阶段估计；进度不可见、超时或数据不足时显示评估中。"
            "本地只保存数字耗时比例，不保存录音名称、路径或内容。"
        )
        outer.addWidget(self.time_estimate_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("thinProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        outer.addWidget(self.progress_bar)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self.start_button = QPushButton("开始处理")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self._start)
        self.cancel_button = QPushButton("安全取消")
        self.cancel_button.setObjectName("secondaryButton")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        action_row.addWidget(self.cancel_button)
        action_row.addWidget(self.start_button)
        outer.addLayout(action_row)

        self.setCentralWidget(central)
        _set_status_badge(self.source_summary, "0 项", "muted")
        _set_status_badge(self.preview_state, "等待选择", "muted")
        self._show_selected_analysis()
        self._refresh_model_state()
        self._update_counts()
        # Always start with an empty queue; do not load a previous session.

    def _show_user_guide(self):
        if self._guide_dialog is None:
            self._guide_dialog = UserGuideDialog(self)
        else:
            self._guide_dialog.reload_guide()
        self._guide_dialog.show()
        self._guide_dialog.raise_()
        self._guide_dialog.activateWindow()

    def _reset_time_estimate(self):
        self._timing_timer.stop()
        self._time_estimator = TimeEstimator()
        self._timing_profile = None
        self.time_estimate_label.setText("已用 00:00 · 预计剩余：开始处理后估计")

    def _configure_timing(self, durations, profile):
        self._timing_profile = profile
        try:
            rates = json.loads(self._timing_settings.value(f"rates/{profile}", "[]"))
            if not isinstance(rates, list):
                rates = []
        except (ValueError, TypeError):
            rates = []
        self._time_estimator.configure(durations, rates)
        self._update_time_estimate()

    def _update_time_estimate(self):
        if self._time_estimator.started is None:
            return
        cancelling = bool(self.worker and self.worker.cancel_event.is_set())
        self.time_estimate_label.setText(self._time_estimator.text(cancelling))

    def _refresh_model_state(self, *, preserve_status: bool = False):
        problems = verify_models()
        asr_ready = Transcriber().is_ready and not any(
            problem.startswith("缺少 faster-whisper") for problem in problems
        )
        diarization_ready = Diarizer().is_ready and not any(
            "pyannote" in problem.casefold() for problem in problems
        )
        try:
            from core.emotion import check_emotion_assets

            tone_assets = check_emotion_assets(TONE_MODEL_DIR, TONE_RUNTIME_DIR)
            tone_ready = bool(tone_assets.get("ready"))
            tone_reason = str(tone_assets.get("reason") or "本地 emotion2vec 资源已就绪。")
        except Exception as exc:
            tone_ready = False
            tone_reason = f"情绪模型状态不可用：{type(exc).__name__}: {str(exc)[:300]}"
        tone_enabled = self.tone_enabled_check.isChecked()
        models_ready = asr_ready and diarization_ready and (tone_ready or not tone_enabled)
        self._core_models_ready = asr_ready and diarization_ready
        diarization_text = f"双人区分模型：{'已就绪' if diarization_ready else '未准备'}"
        self.asr_ready_label.setToolTip(str(ASR_MODEL_DIR))
        self.diarization_ready_label.setToolTip(str(DIARIZATION_MODEL_DIR))
        self.tone_ready_label.setToolTip(f"{TONE_MODEL_DIR}\n{tone_reason}")
        self._tone_ready = tone_ready
        _set_status_badge(
            self.asr_ready_label,
            f"large-v3：{'已就绪' if asr_ready else '未准备'}",
            "ready" if asr_ready else "warning",
        )
        _set_status_badge(
            self.diarization_ready_label,
            diarization_text,
            "ready" if diarization_ready else "warning",
        )
        _set_status_badge(
            self.tone_ready_label,
            f"emotion2vec：{'已就绪' if tone_ready else '未准备'}",
            "ready" if tone_ready else "warning",
        )
        if not (self.worker and self.worker.isRunning()):
            self.start_button.setEnabled(models_ready)
            if not preserve_status:
                if models_ready:
                    self.progress_label.setText("本地模型已齐全，可以开始处理。")
                elif tone_enabled and asr_ready and diarization_ready:
                    self.progress_label.setText("情绪模型未准备；运行 python tools/prepare_models.py --tone，或关闭情绪分析。")
                else:
                    self.progress_label.setText("本地模型未齐全；准备完成后请刷新模型状态。")

    def _toggle_advanced(self, checked: bool):
        self.advanced_panel.setVisible(checked)
        self.advanced_toggle.setText("收起高级识别设置" if checked else "显示高级识别设置")

    def _refresh_performance_controls(self, *_):
        batch_selected = self.inference_mode_combo.currentData() == "gpu_batch"
        self.batch_size_spin.setEnabled(batch_selected)
        if batch_selected and self.device_combo.currentData() == "cpu":
            self.performance_hint.setText(
                "当前设备选为CPU，GPU批量模式会回退到准确识别；改为自动后会在CUDA可用时启用。"
            )
        elif batch_selected:
            self.performance_hint.setText(
                "VAD会过滤静音并改变分段/解码回退行为；结果请复核。显存不足时自动降batch或退回准确串行。"
            )
        else:
            self.performance_hint.setText(
                "准确模式维持现有VAD关闭路径；ASR线程数限制在1到8。"
            )

    def _choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择音频文件",
            "",
            "音频文件 (*.wav *.mp3 *.m4a *.flac *.aac *.ogg *.wma *.mp4)",
        )
        if paths:
            self._set_inputs([Path(item).resolve() for item in paths], None)

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择音频文件夹")
        if not folder:
            return
        root = Path(folder).resolve()
        files = collect_audio_files(root)
        if not files:
            QMessageBox.information(self, "没有音频", "所选文件夹中没有支持的音频文件。")
            return
        self._set_inputs(files, root)

    def _confirm_batch_replacement(self) -> bool:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("替换当前批次")
        dialog.setText(
            "新选择会替换当前批次的队列、校对和角色确认，并删除本地恢复记录。"
            "尚未导出的结果可能丢失，请先保存 Word。"
        )
        dialog.setInformativeText("选择“替换当前批次”继续；取消、按 Esc 或关闭此窗口都会保留当前工作。")
        cancel_button = dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        replace_button = dialog.addButton("替换当前批次", QMessageBox.ButtonRole.DestructiveRole)
        dialog.setDefaultButton(cancel_button)
        dialog.setEscapeButton(cancel_button)
        dialog.exec()
        return dialog.clickedButton() == replace_button

    def _set_inputs(self, files, root):
        unique = sorted({Path(item).resolve() for item in files}, key=lambda item: str(item).casefold())
        if not unique:
            return
        if self.input_files and not self._confirm_batch_replacement():
            return
        self._stop_playback(clear_source=True)
        self._delete_session_file()
        self._reset_time_estimate()
        self.input_files = unique
        self.input_root = Path(root).resolve() if root else None
        self.analyses = [None] * len(unique)
        self._stale_results.clear()
        self._session_ready_count = 0
        self.role_choices.clear()
        self.table.setRowCount(len(unique))
        for row, path in enumerate(unique):
            name_item = QTableWidgetItem(path.name)
            name_item.setToolTip(str(path))
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QTableWidgetItem())
            self.table.setItem(row, 2, QTableWidgetItem())
            self._set_queue_status(row, "pending", "待处理")
            self._set_queue_detail(row, "尚未处理")
        self.source_summary.setText(f"{len(unique)} 项")
        self.start_button.setText("开始处理")
        self.queue_empty.setVisible(not unique)
        self.table.setVisible(bool(unique))
        if unique:
            self.table.selectRow(0)
        self.preview.clear()
        self._update_counts()
        self._show_selected_analysis()
        self._persist_session()

    def _clear_queue(self):
        if self.worker and self.worker.isRunning():
            return
        self._stop_playback(clear_source=True)
        self._reset_time_estimate()
        self.input_files = []
        self.input_root = None
        self.analyses = []
        self._stale_results.clear()
        self.role_choices.clear()
        self._sentence_ids_by_row.clear()
        self._delete_session_file()
        self.table.setRowCount(0)
        self.source_summary.setText("0 项")
        self.queue_empty.show()
        self.table.hide()
        self.preview.clear()
        self._update_counts()
        self._show_selected_analysis()

    def _delete_session_file(self):
        try:
            self._session_path.unlink(missing_ok=True)
        except OSError:
            self._session_notice = "本地恢复记录未能清除。"

    def _analysis_to_record(self, result: FileAnalysis | None) -> dict | None:
        if result is None or result.status not in _RESTORABLE_STATUSES:
            return None
        error = str(result.error or "")
        for private_path in (result.input_path, result.output_dir):
            if private_path:
                error = error.replace(str(private_path), "[本地路径]")
        record = {
            "input_path": str(result.input_path),
            "output_dir": str(result.output_dir),
            "status": result.status,
            "words": result.words,
            "segments": result.segments,
            "duration_seconds": result.duration_seconds,
            "elapsed_seconds": result.elapsed_seconds,
            "error": error[:500],
            "word_path": str(result.word_path) if result.word_path else None,
            "json_path": str(result.json_path) if result.json_path else None,
            "roles_confirmed": bool(result.roles_confirmed),
            "counselor_speaker": result.counselor_speaker,
            "emotion_windows": result.emotion_windows,
            "analysis_windows": result.analysis_windows,
            "tone_enabled": result.tone_enabled,
            "tone_model": result.tone_model,
            "tone_status": result.tone_status,
            "tone_error": str(result.tone_error or "")[:500],
            "performance": result.performance,
            "sentence_revisions": result.sentence_revisions,
            "revision": result.revision,
            "exported_revision": result.exported_revision,
            "profile": result.profile,
            "source_size_bytes": result.source_size_bytes,
            "source_mtime_ns": result.source_mtime_ns,
            "source_changed": bool(result.source_changed),
        }
        return _json_value(record)

    def _analysis_from_record(self, record: dict, expected_path: Path) -> FileAnalysis:
        if not isinstance(record, dict):
            raise ValueError("缓存结果结构无效")
        stored_path = record.get("input_path")
        output_text = record.get("output_dir")
        status = record.get("status")
        if (
            not isinstance(stored_path, str)
            or len(stored_path) > 4096
            or not isinstance(output_text, str)
            or len(output_text) > 4096
        ):
            raise ValueError("缓存路径无效")
        if Path(stored_path).expanduser().resolve(strict=False) != expected_path.resolve(strict=False):
            raise ValueError("缓存队列映射无效")
        if status not in _RESTORABLE_STATUSES:
            raise ValueError("缓存状态无效")

        words = record.get("words", [])
        segments = record.get("segments", [])
        if not isinstance(words, list) or len(words) > _MAX_SESSION_WORDS:
            raise ValueError("缓存词级结果无效")
        if not isinstance(segments, list) or len(segments) > _MAX_SESSION_WORDS:
            raise ValueError("缓存片段结果无效")
        for word in words:
            if not isinstance(word, dict):
                raise ValueError("缓存词级结果无效")
            start, end, text = word.get("start"), word.get("end"), word.get("text")
            if not _valid_optional_number(start, minimum=0) or not _valid_optional_number(end, minimum=0):
                raise ValueError("缓存时间戳无效")
            if start is None or end is None or end < start or not isinstance(text, str):
                raise ValueError("缓存词级结果无效")
            if word.get("speaker_id") not in {None, "speaker_1", "speaker_2"}:
                raise ValueError("缓存说话人无效")
            if "uncertain" in word and not isinstance(word["uncertain"], bool):
                raise ValueError("缓存词级置信标记无效")
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError("缓存说话人片段无效")
            start, end = segment.get("start"), segment.get("end")
            if (
                not _valid_optional_number(start, minimum=0)
                or not _valid_optional_number(end, minimum=0)
                or start is None
                or end is None
                or end < start
                or not isinstance(segment.get("text", ""), str)
                or segment.get("speaker_id") not in {None, "speaker_1", "speaker_2"}
            ):
                raise ValueError("缓存说话人片段无效")
        duration = record.get("duration_seconds")
        elapsed = record.get("elapsed_seconds", 0)
        if not _valid_optional_number(duration, minimum=0) or not _valid_optional_number(elapsed, minimum=0):
            raise ValueError("缓存时长无效")

        revisions = record.get("sentence_revisions", {})
        base_sentence_ids = {
            make_sentence_id(index, item["start"], item["end"])
            for index, item in enumerate(build_sentences(words))
        }
        if not isinstance(revisions, dict) or len(revisions) > len(base_sentence_ids):
            raise ValueError("缓存校对记录无效")
        for sentence_key, override in revisions.items():
            if (
                not isinstance(sentence_key, str)
                or len(sentence_key) > 256
                or sentence_key not in base_sentence_ids
                or not isinstance(override, dict)
            ):
                raise ValueError("缓存校对记录无效")
            if "text" in override and (not isinstance(override["text"], str) or len(override["text"]) > 100_000):
                raise ValueError("缓存校对文本无效")
            if override.get("speaker_id") not in {None, "speaker_1", "speaker_2"}:
                raise ValueError("缓存校对说话人无效")

        word_path_text, json_path_text = record.get("word_path"), record.get("json_path")
        if json_path_text is not None and word_path_text is None:
            raise ValueError("缓存导出路径无效")
        if any(
            value is not None and (not isinstance(value, str) or len(value) > 4096)
            for value in (word_path_text, json_path_text)
        ):
            raise ValueError("缓存导出路径无效")
        word_path = Path(word_path_text).expanduser().resolve(strict=False) if word_path_text else None
        json_path = Path(json_path_text).expanduser().resolve(strict=False) if json_path_text else None
        output_dir = Path(output_text).expanduser().resolve(strict=False)
        if word_path is not None:
            if (
                word_path.parent != output_dir
                or word_path.suffix.casefold() != ".docx"
            ):
                raise ValueError("缓存 Word 导出路径无效")

        revision = record.get("revision", 0)
        exported_revision = record.get("exported_revision", -1)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("缓存修订号无效")
        if isinstance(exported_revision, bool) or not isinstance(exported_revision, int) or exported_revision < -1:
            raise ValueError("缓存导出修订号无效")
        source_size = record.get("source_size_bytes")
        source_mtime = record.get("source_mtime_ns")
        if (source_size is None) != (source_mtime is None):
            raise ValueError("缓存源文件指纹无效")
        for fingerprint_value in (source_size, source_mtime):
            if fingerprint_value is not None and (
                isinstance(fingerprint_value, bool)
                or not isinstance(fingerprint_value, int)
                or fingerprint_value < 0
            ):
                raise ValueError("缓存源文件指纹无效")

        result = FileAnalysis(expected_path, output_dir, status)
        result.words = words
        result.segments = segments
        result.duration_seconds = duration
        result.elapsed_seconds = elapsed
        result.error = str(record.get("error", ""))[:500]
        result.word_path = word_path
        result.json_path = json_path
        result.roles_confirmed = record.get("roles_confirmed") is True
        counselor = record.get("counselor_speaker")
        if counselor not in {None, "speaker_1", "speaker_2"}:
            raise ValueError("缓存角色无效")
        result.counselor_speaker = counselor
        emotion_windows = record.get("emotion_windows")
        analysis_windows = record.get("analysis_windows", [])
        if emotion_windows is not None:
            _validate_window_records(emotion_windows, "情绪窗口")
        _validate_window_records(analysis_windows, "声学窗口")
        result.emotion_windows = emotion_windows
        result.analysis_windows = analysis_windows
        tone_enabled = record.get("tone_enabled")
        if tone_enabled not in {None, True, False}:
            raise ValueError("缓存情绪设置无效")
        result.tone_enabled = tone_enabled
        tone_model = record.get("tone_model")
        if tone_model is not None and not isinstance(tone_model, dict):
            raise ValueError("缓存模型信息无效")
        result.tone_model = tone_model
        tone_status = record.get("tone_status")
        if tone_status is not None and not isinstance(tone_status, str):
            raise ValueError("缓存情绪状态无效")
        result.tone_status = tone_status
        result.tone_error = str(record.get("tone_error", ""))[:500]
        performance = record.get("performance", {})
        if not isinstance(performance, dict) or len(performance) > 32:
            raise ValueError("缓存性能记录无效")
        result.performance = performance
        result.sentence_revisions = revisions
        result.revision = revision
        result.exported_revision = exported_revision
        profile = record.get("profile")
        if profile is not None and (not isinstance(profile, str) or len(profile) > 256):
            raise ValueError("缓存处理配置无效")
        result.profile = profile
        result.source_size_bytes = source_size
        result.source_mtime_ns = source_mtime
        source_changed = record.get("source_changed", False)
        if not isinstance(source_changed, bool):
            raise ValueError("缓存源文件状态无效")
        result.source_changed = source_changed
        return result

    def _restore_session(self):
        try:
            payload = self._session_store.read()
        except ResultStoreError:
            self._session_error = "本地恢复记录损坏或过大；旧内容未载入，仍可重新选择音频继续处理。"
            self.progress_label.setText(self._session_error)
            return
        if payload is None:
            return
        try:
            if payload.get("schema_version") != _SESSION_SCHEMA_VERSION:
                raise ValueError("缓存版本不支持")
            files = payload.get("input_files")
            records = payload.get("analyses")
            fingerprints = payload.get("fingerprints")
            if not isinstance(files, list) or not isinstance(records, list) or not isinstance(fingerprints, list):
                raise ValueError("缓存队列结构无效")
            if len(files) > _MAX_SESSION_FILES or len(records) != len(files) or len(fingerprints) != len(files):
                raise ValueError("缓存队列数量无效")
            input_files = []
            for value in files:
                if not isinstance(value, str) or len(value) > 4096:
                    raise ValueError("缓存音频路径无效")
                input_files.append(Path(value).expanduser().resolve(strict=False))
            if len({str(path).casefold() for path in input_files}) != len(input_files):
                raise ValueError("缓存队列存在重复音频")
            input_root_value = payload.get("input_root")
            output_root_value = payload.get("output_root")
            if input_root_value is not None and not isinstance(input_root_value, str):
                raise ValueError("缓存输入目录无效")
            if not isinstance(output_root_value, str) or len(output_root_value) > 4096:
                raise ValueError("缓存输出目录无效")
            input_root = Path(input_root_value).expanduser().resolve(strict=False) if input_root_value else None
            output_root = Path(output_root_value).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            self._session_error = "本地恢复记录格式无效；旧内容未载入，仍可重新选择音频继续处理。"
            self.progress_label.setText(self._session_error)
            return

        analyses = []
        normalized_fingerprints = []
        restored_count = 0
        invalid_rows = set()
        for row, (path, record, fingerprint) in enumerate(zip(input_files, records, fingerprints)):
            valid_fingerprint = fingerprint
            if fingerprint is not None and (
                not isinstance(fingerprint, dict)
                or isinstance(fingerprint.get("size_bytes"), bool)
                or not isinstance(fingerprint.get("size_bytes"), int)
                or fingerprint["size_bytes"] < 0
                or isinstance(fingerprint.get("mtime_ns"), bool)
                or not isinstance(fingerprint.get("mtime_ns"), int)
                or fingerprint["mtime_ns"] < 0
            ):
                valid_fingerprint = None
                invalid_rows.add(row)
            normalized_fingerprints.append(valid_fingerprint)
            try:
                analysis = self._analysis_from_record(record, path) if record is not None else None
            except (OSError, RuntimeError, TypeError, ValueError, KeyError, OverflowError):
                analysis = None
                invalid_rows.add(row)
            if analysis is not None:
                expected_fingerprint = valid_fingerprint
                if analysis.source_size_bytes is not None and analysis.source_mtime_ns is not None:
                    expected_fingerprint = {
                        "size_bytes": analysis.source_size_bytes,
                        "mtime_ns": analysis.source_mtime_ns,
                    }
                current = _current_fingerprint(path)
                if current is None or expected_fingerprint is None or current != expected_fingerprint:
                    analysis.source_changed = True
                if analysis.status == "exported":
                    word_exists = bool(
                        analysis.word_path
                        and analysis.word_path.is_file()
                        and analysis.exported_revision == analysis.revision
                    )
                    if not word_exists:
                        analysis.status = "no_speech" if not analysis.words else "ready"
                        analysis.error = "上次导出文件缺失或版本已变化，请重新保存。"
                restored_count += 1
            analyses.append(analysis)

        try:
            role_choices = payload.get("role_choices", {})
            normalized_roles = {}
            allowed_paths = {str(path) for path in input_files}
            if isinstance(role_choices, dict):
                for path_text, speaker in role_choices.items():
                    if (
                        isinstance(path_text, str)
                        and path_text in allowed_paths
                        and (
                            speaker is None
                            or (isinstance(speaker, str) and speaker in {"speaker_1", "speaker_2"})
                        )
                    ):
                        normalized_roles[path_text] = speaker
            stale_records = payload.get("stale_results", [])
            stale_results = {}
            if isinstance(stale_records, list):
                for entry in stale_records[:len(input_files)]:
                    if not isinstance(entry, dict):
                        continue
                    row = entry.get("row")
                    if (
                        isinstance(row, bool)
                        or not isinstance(row, int)
                        or not 0 <= row < len(input_files)
                        or row in stale_results
                    ):
                        continue
                    try:
                        stale_results[row] = self._analysis_from_record(entry.get("analysis"), input_files[row])
                    except (OSError, RuntimeError, TypeError, ValueError, KeyError, OverflowError):
                        continue
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            normalized_roles = {}
            stale_results = {}

        self.input_files = input_files
        self.input_root = input_root
        self.analyses = analyses
        self.role_choices = normalized_roles
        self._stale_results = stale_results
        self.output_edit.setText(str(output_root))
        self.table.setRowCount(len(input_files))
        changed_count = 0
        for row, path in enumerate(input_files):
            name_item = QTableWidgetItem(path.name)
            name_item.setToolTip(str(path))
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QTableWidgetItem())
            self.table.setItem(row, 2, QTableWidgetItem())
            result = analyses[row]
            fingerprint = normalized_fingerprints[row]
            if result is None:
                current = _current_fingerprint(path)
                changed = current is None or fingerprint is None or current != fingerprint
                if changed:
                    changed_count += 1
                    self._set_queue_status(row, "warning", "源文件待检查")
                    detail = "源音频不存在或已变化；处理时将使用当前文件。"
                else:
                    self._set_queue_status(row, "pending", "待处理")
                    detail = "上次未完成；可继续处理。"
                if row in invalid_rows:
                    detail = "本行恢复记录无效，需重新处理。 " + detail
                self._set_queue_detail(row, detail)
            elif result.source_changed:
                changed_count += 1
                self._set_queue_status(row, "warning", "音频已变化")
                self._set_queue_detail(row, "旧结果与校对已保留；请重新处理当前音频版本。")
            else:
                status_text, tone = _analysis_status(result)
                self._set_queue_status(row, tone, status_text)
                detail = f"已恢复 {len(result.words)} 个时间片段"
                if result.status == "ready" and result.exported_revision != result.revision:
                    detail += " · 可重新导出"
                self._set_queue_detail(row, detail)
        self.source_summary.setText(f"{len(input_files)} 项")
        self.queue_empty.setVisible(not input_files)
        self.table.setVisible(bool(input_files))
        if input_files:
            self.table.selectRow(0)
        pending_count = sum(
            1 for result in analyses
            if result is None or result.status in {"failed", "cancelled"} or result.source_changed
        )
        self._session_ready_count = restored_count
        log_event("session_restore", record_count=restored_count)
        self.progress_label.setText(f"已恢复 {restored_count} 个完成结果，剩余 {pending_count} 个文件。")
        self.start_button.setText("继续处理" if pending_count else "开始处理")
        self._update_counts()
        self._show_selected_analysis()
        if invalid_rows:
            self._session_notice = f"{len(invalid_rows)} 个队列项的恢复记录无效；这些项目已保留为待处理状态。"
        elif changed_count:
            self._session_notice = f"其中 {changed_count} 个源文件已变化或不可用；旧的已完成结果保留在本地会话中。"

    def _persist_session(self):
        if not self.input_files:
            self._delete_session_file()
            return
        analyses = self.analyses
        if len(analyses) != len(self.input_files):
            return
        fingerprints = []
        for row, path in enumerate(self.input_files):
            result = analyses[row]
            if result is not None and result.source_size_bytes is not None and result.source_mtime_ns is not None:
                fingerprints.append({"size_bytes": result.source_size_bytes, "mtime_ns": result.source_mtime_ns})
            else:
                fingerprints.append(_current_fingerprint(path))
        stale_results = [
            {"row": row, "analysis": self._analysis_to_record(result)}
            for row, result in self._stale_results.items()
            if 0 <= row < len(self.input_files) and self._analysis_to_record(result) is not None
        ]
        payload = {
            "schema_version": _SESSION_SCHEMA_VERSION,
            "input_files": [str(path) for path in self.input_files],
            "input_root": str(self.input_root) if self.input_root else None,
            "output_root": str(Path(self.output_edit.text()).expanduser().resolve(strict=False)),
            "fingerprints": fingerprints,
            "analyses": [self._analysis_to_record(result) for result in analyses],
            "role_choices": self.role_choices,
            "stale_results": stale_results,
        }
        try:
            self._session_store.write(_json_value(payload))
        except (ResultStoreError, OSError, TypeError, ValueError):
            self._session_error = "本地恢复记录未能保存；当前处理仍可继续，请及时导出 Word。"
            if not getattr(self, "_session_warning_shown", False):
                self._session_warning_shown = True
                QMessageBox.warning(self, "本地恢复未保存", self._session_error)

    def _choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择保存目录", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _start(self):
        if self.worker and self.worker.isRunning():
            return
        self._refresh_model_state()
        if not self.start_button.isEnabled():
            if (
                getattr(self, "_core_models_ready", False)
                and self.tone_enabled_check.isChecked()
                and not getattr(self, "_tone_ready", False)
            ):
                message = "请运行 python tools/prepare_models.py --tone 准备情绪模型，或取消勾选“情绪倾向分析”。"
            else:
                message = "请先准备 large-v3 与 pyannote 双人区分模型，再刷新模型状态。"
            QMessageBox.information(
                self,
                "本地模型未齐全",
                message,
            )
            return
        if not self.input_files:
            QMessageBox.information(self, "先选择音频", "请选择一个音频文件或文件夹。")
            return
        output_text = self.output_edit.text().strip()
        if not output_text:
            QMessageBox.information(self, "缺少保存目录", "请选择转写结果的保存目录。")
            return
        output_root = Path(output_text).expanduser().resolve()
        if self.input_root and output_root == self.input_root:
            QMessageBox.warning(self, "保存目录冲突", "音频文件夹和保存目录不能相同。")
            return
        if not self.input_root and len(self.input_files) == 1 and output_root == self.input_files[0].parent:
            QMessageBox.warning(self, "保存目录冲突", "请为结果选择与源音频不同的保存目录。")
            return

        pending_rows = [
            row
            for row, result in enumerate(self.analyses)
            if result is None
            or result.status in {"failed", "cancelled"}
            or bool(getattr(result, "source_changed", False))
        ]
        if not pending_rows:
            QMessageBox.information(self, "队列已完成", "没有需要重新处理的文件；如需补存结果，请使用“保存 Word”。")
            return
        for row in pending_rows:
            if self.analyses[row] is not None:
                stale = self.analyses[row]
                if getattr(stale, "source_changed", False):
                    self._stale_results[row] = copy.deepcopy(stale)
                elif stale.status in {"failed", "cancelled"}:
                    self.analyses[row] = None
            self._set_queue_status(row, "pending", "待处理")
            self._set_queue_detail(row, "尚未处理" if self.analyses[row] is None else "使用当前音频版本重新处理；旧结果已保留在本地会话备份")
        device_choice = self.device_combo.currentData() if self.advanced_toggle.isChecked() else "auto"
        compute_type = self.compute_combo.currentData() if self.advanced_toggle.isChecked() else "float16"
        beam_size = self.beam_spin.value() if self.advanced_toggle.isChecked() else 5
        inference_mode = (
            self.inference_mode_combo.currentData()
            if self.advanced_toggle.isChecked() else "accurate"
        )
        batch_size = self.batch_size_spin.value() if self.advanced_toggle.isChecked() else 2
        cpu_threads = self.cpu_threads_spin.value() if self.advanced_toggle.isChecked() else 8
        self.worker = ProcessingWorker(
            [self.input_files[row] for row in pending_rows],
            self.input_root,
            output_root,
            device_choice,
            compute_type,
            beam_size,
            self.tone_enabled_check.isChecked(),
            row_indices=pending_rows,
            inference_mode=inference_mode,
            batch_size=batch_size,
            cpu_threads=cpu_threads,
        )
        self.worker.progress.connect(self._progress)
        self.worker.analysis_ready.connect(self._analysis_ready)
        self.worker.timing_configured.connect(self._configure_timing)
        self.worker.stage_progress.connect(self._stage_timing)
        self.worker.finished.connect(self._finished)
        self.start_button.setEnabled(False)
        self.tone_enabled_check.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.choose_files_button.setEnabled(False)
        self.choose_folder_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("正在准备处理队列…")
        self._timing_profile = None
        self._active_worker_rows = list(pending_rows)
        self._eta_index_by_row = {row: index for index, row in enumerate(pending_rows)}
        self._time_estimator.start(len(pending_rows))
        self._timing_timer.start()
        self._update_time_estimate()
        self._persist_session()
        self.worker.start()

    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.progress_label.setText("已请求取消；将于当前安全处理边界停止。")
            self._update_time_estimate()

    def _progress(self, current, total, row, message):
        self.progress_label.setText(message)
        self.progress_bar.setValue(int((current - 1) / max(total, 1) * 100))
        if 0 <= row < len(self.analyses) and (
            self.analyses[row] is None or row in self._active_worker_rows
        ):
            detail = message.split("] ", 1)[-1]
            self._set_queue_status(row, "processing", "处理中")
            self._set_queue_detail(row, detail)
            if self.table.currentRow() == row:
                _set_status_badge(self.preview_state, "处理中", "processing")
                self.preview.setPlainText(detail)

    def _analysis_ready(self, row: int, result: FileAnalysis):
        if row < 0 or row >= len(self.analyses):
            return
        self.analyses[row] = result
        learn_timing = result.status == "ready" and bool(result.words) and result.tone_status not in {"failed", "partial"}
        if result.profile and result.profile != self._timing_profile:
            self._timing_profile = result.profile
            try:
                rates = json.loads(self._timing_settings.value(f"rates/{result.profile}", "[]"))
                if not isinstance(rates, list):
                    rates = []
            except (ValueError, TypeError):
                rates = []
            self._time_estimator.configure(list(self._time_estimator.durations), rates)
        eta_row = self._eta_index_by_row.get(row)
        if eta_row is not None:
            self._time_estimator.complete(eta_row, result.duration_seconds, result.elapsed_seconds, learn_timing)
        if learn_timing and result.profile:
            self._timing_settings.setValue(f"rates/{result.profile}", json.dumps(self._time_estimator.rates))
            self._timing_settings.sync()
        self._update_time_estimate()
        status_text, tone = _analysis_status(result)
        self._set_queue_status(row, tone, status_text)
        detail = result.error or f"{len(result.words)} 个时间片段 · {result.elapsed_seconds:.1f} 秒"
        if result.tone_status in {"failed", "partial"} and result.tone_error:
            detail += f" · 情绪分析：{result.tone_error[:180]}"
        self._set_queue_detail(row, detail)
        if result.source_changed:
            self._set_queue_status(row, "warning", "音频已变化")
            self._set_queue_detail(row, "处理期间源文件发生变化；此结果保留，请重新处理后再导出。")
        self._update_counts()
        self._persist_session()
        if self.table.currentRow() == row:
            self._show_selected_analysis()

    def _stage_timing(self, row, stage, fraction, audio_seconds):
        eta_row = self._eta_index_by_row.get(row)
        if eta_row is not None:
            self._time_estimator.progress(eta_row, stage, fraction, audio_seconds)
            self._update_time_estimate()

    def _set_queue_status(self, row: int, tone: str, text: str):
        item = self.table.item(row, 1)
        if item is None:
            return
        colors = {
            "pending": ("#6e7c72", "#f0f2ef"),
            "processing": ("#816320", "#f8f0dc"),
            "warning": ("#80601d", "#f7efd9"),
            "ready": ("#38654f", "#e8f1e9"),
            "no_speech": ("#6e7c72", "#f0f2ef"),
            "exported": ("#38654f", "#e8f1e9"),
            "failed": ("#9a433c", "#f8e9e6"),
            "export_failed": ("#9a433c", "#f8e9e6"),
            "cancelled": ("#6e7c72", "#f0f2ef"),
        }
        foreground, background = colors.get(tone, colors["pending"])
        item.setText(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setForeground(QColor(foreground))
        item.setBackground(QColor(background))

    def _set_queue_detail(self, row: int, text: str):
        detail_item = self.table.item(row, 2)
        if detail_item is None:
            return
        detail_item.setText(text)
        path = self.input_files[row]
        for column in (0, 1):
            item = self.table.item(row, column)
            if item is not None:
                item.setToolTip(f"{text}\n{path}")

    def _finished(self):
        self._timing_timer.stop()
        self._time_estimator.finish()
        self._update_time_estimate()
        self.cancel_button.setEnabled(False)
        self.choose_files_button.setEnabled(True)
        self.choose_folder_button.setEnabled(True)
        self.tone_enabled_check.setEnabled(True)
        self.progress_bar.setValue(100)
        cancelled = bool(self.worker and self.worker.cancel_event.is_set())
        self._update_counts()
        done = sum(item is not None and item.status in {"ready", "no_speech", "exported"} for item in self.analyses)
        failed = sum(item is not None and item.status in {"failed", "export_failed"} for item in self.analyses)
        stopped = sum(item is not None and item.status == "cancelled" for item in self.analyses)
        status = "处理已取消" if cancelled else "队列已处理完毕"
        self.progress_label.setText(
            f"{status}：完成 {done} · 失败 {failed} · 已取消 {stopped}。请检查各文件状态。"
        )
        remaining = sum(
            1 for result in self.analyses
            if result is None or result.status in {"failed", "cancelled"} or result.source_changed
        )
        self.start_button.setText("继续处理" if remaining else "开始处理")
        self._persist_session()
        self._refresh_model_state(preserve_status=True)
        if getattr(self, "_close_when_done", False):
            QTimer.singleShot(0, self.close)

    def _update_counts(self):
        done = sum(item is not None and item.status in {"ready", "no_speech", "exported"} for item in self.analyses)
        failed = sum(item is not None and item.status in {"failed", "export_failed"} for item in self.analyses)
        cancelled = sum(item is not None and item.status == "cancelled" for item in self.analyses)
        waiting = max(0, len(self.input_files) - done - failed - cancelled)
        self.counts_label.setText(f"待处理 {waiting}  ·  完成 {done}  ·  失败 {failed}  ·  已取消 {cancelled}")

    def _role_changed(self, _index):
        if self._updating_role:
            return
        row = self.table.currentRow()
        if row >= 0 and row < len(self.analyses):
            path = str(self.input_files[row])
            self.role_choices[path] = self.role_combo.currentData()
            self._persist_session()

    def _base_sentences(self, result: FileAnalysis) -> list[dict]:
        return build_sentences(result.words, result.emotion_windows, result.tone_enabled, result.analysis_windows)

    def _populate_sentence_review(self, row: int, result: FileAnalysis):
        base = self._base_sentences(result)
        previous_id = self._selected_sentence_id if self._editor_row == row else None
        ids = [make_sentence_id(index, item["start"], item["end"]) for index, item in enumerate(base)]
        self._sentence_ids_by_row[row] = ids
        self._suppress_sentence_events = True
        self.sentence_combo.clear()
        for index, (sentence_id, item) in enumerate(zip(ids, base)):
            start = max(0, int(float(item["start"])))
            label = f"[{start // 60:02d}:{start % 60:02d}] {_speaker_label(item.get('speaker_id'))} · {item['text'][:54]}"
            self.sentence_combo.addItem(label, sentence_id)
        selected_index = ids.index(previous_id) if previous_id in ids else (0 if ids else -1)
        if selected_index >= 0:
            self.sentence_combo.setCurrentIndex(selected_index)
        self._editor_row = row
        self._selected_sentence_id = ids[selected_index] if selected_index >= 0 else None
        self._load_sentence_editor(row, self._selected_sentence_id, base)
        enabled = bool(base) and result.status in {"ready", "exported", "export_failed"} and not result.source_changed
        self.sentence_combo.setEnabled(enabled)
        self.sentence_editor.setEnabled(enabled)
        self.sentence_speaker_combo.setEnabled(enabled)
        self.save_correction_button.setEnabled(False)
        self.play_sentence_button.setEnabled(enabled and not self._audio_unavailable_reason)
        self.stop_audio_button.setEnabled(self._media_player is not None)
        if self._audio_unavailable_reason:
            self.sentence_precision_note.setText("当前环境无法启用本地回听；转写、校对和导出仍可使用。")
        elif result.source_changed:
            self.sentence_precision_note.setText("源音频已变化；旧结果与校对已保留，请先重新处理后再回听或导出。")
        else:
            self.sentence_precision_note.setText(
                "编辑器按原始句完整校对；增加标点拆出的片段仍共享原时间范围。"
                "回听按识别时间范围定位，同一识别词内拆出的句子可能共用时间范围。"
            )
        self._suppress_sentence_events = False

    def _load_sentence_editor(self, row: int, sentence_id: str | None, base=None):
        if not (0 <= row < len(self.analyses)) or self.analyses[row] is None or not sentence_id:
            self._suppress_sentence_events = True
            self.sentence_editor.clear()
            self.sentence_speaker_combo.setCurrentIndex(0)
            self._suppress_sentence_events = False
            return
        result = self.analyses[row]
        sentences = base if base is not None else self._base_sentences(result)
        sentence_index = next(
            (index for index, item in enumerate(sentences)
             if make_sentence_id(index, item["start"], item["end"]) == sentence_id),
            None,
        )
        if sentence_index is None:
            return
        original = sentences[sentence_index]
        override = result.sentence_revisions.get(sentence_id, {})
        text = override.get("text", original["text"])
        speaker = override.get("speaker_id")
        self._suppress_sentence_events = True
        self.sentence_editor.setPlainText(str(text))
        combo_index = 0
        if speaker in {"speaker_1", "speaker_2"}:
            combo_index = 1 if speaker == "speaker_1" else 2
        self.sentence_speaker_combo.setCurrentIndex(combo_index)
        self._suppress_sentence_events = False
        self._editor_original_text = str(text)
        self._editor_original_speaker = original.get("speaker_id")

    def _sentence_selected(self, index: int):
        if self._suppress_sentence_events:
            return
        previous_id = self._selected_sentence_id
        row = self._editor_row
        if row is not None and previous_id:
            self._save_sentence_correction(silent=True, row=row, sentence_id=previous_id, refresh=False)
        self._stop_playback()
        row = self.table.currentRow()
        sentence_id = self.sentence_combo.itemData(index) if index >= 0 else None
        self._selected_sentence_id = sentence_id
        self._editor_row = row if row >= 0 else None
        if row >= 0 and row < len(self.analyses) and self.analyses[row] is not None:
            self._load_sentence_editor(row, sentence_id)
        self.save_correction_button.setEnabled(False)

    def _sentence_edit_changed(self, _value=None):
        if self._suppress_sentence_events:
            return
        has_selection = self._editor_row is not None and bool(self._selected_sentence_id)
        if not has_selection:
            self.save_correction_button.setEnabled(False)
            return
        result = self.analyses[self._editor_row]
        if result is None:
            self.save_correction_button.setEnabled(False)
            return
        existing = result.sentence_revisions.get(self._selected_sentence_id, {})
        saved_text = existing.get("text")
        saved_speaker = existing.get("speaker_id")
        current_text = self.sentence_editor.toPlainText()
        current_speaker = self.sentence_speaker_combo.currentData()
        if current_speaker == self._editor_original_speaker:
            current_speaker = None
        self.save_correction_button.setEnabled(
            current_text != (saved_text if saved_text is not None else self._editor_original_text)
            or current_speaker != saved_speaker
        )

    def _save_sentence_correction(
        self,
        *,
        silent: bool = False,
        row: int | None = None,
        sentence_id: str | None = None,
        refresh: bool = True,
    ):
        row = self._editor_row if row is None else row
        sentence_id = self._selected_sentence_id if sentence_id is None else sentence_id
        if row is None or not sentence_id or not 0 <= row < len(self.analyses):
            return
        result = self.analyses[row]
        if result is None or result.status not in {"ready", "exported", "export_failed"}:
            return
        if row == self._editor_row and not self.save_correction_button.isEnabled():
            return
        text = self.sentence_editor.toPlainText().strip()
        speaker = self.sentence_speaker_combo.currentData()
        base = self._base_sentences(result)
        original_index = next(
            (index for index, item in enumerate(base)
             if make_sentence_id(index, item["start"], item["end"]) == sentence_id),
            None,
        )
        if original_index is None:
            return
        original = base[original_index]
        override = {}
        if text != original["text"]:
            override["text"] = text
        if speaker in {"speaker_1", "speaker_2"} and speaker != original.get("speaker_id"):
            override["speaker_id"] = speaker
        if override:
            result.sentence_revisions[sentence_id] = override
        else:
            result.sentence_revisions.pop(sentence_id, None)
        if result.status == "exported":
            result.status = "ready"
        result.revision += 1
        result.error = ""
        self._persist_session()
        if refresh and self.table.currentRow() == row:
            self._show_selected_analysis()
        elif self.save_correction_button is not None:
            self.save_correction_button.setEnabled(False)
        if not silent:
            self.progress_label.setText("已保存校对；原始识别文字和时间戳已保留，保存 Word 可同步导出修订。")

    def _ensure_audio_player(self) -> bool:
        """Open audio devices only when playback is requested, never at startup."""
        if self._media_player is not None:
            return True
        if self._audio_unavailable_reason:
            return False
        self.sentence_precision_note.setText("正在初始化本地回听设备，请稍候…")
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

            self._audio_output = QAudioOutput(self)
            self._media_player = QMediaPlayer(self)
            self._media_player.setAudioOutput(self._audio_output)
            self._media_player.positionChanged.connect(self._media_position_changed)
            self._media_player.mediaStatusChanged.connect(self._media_status_changed)
            self._media_player.errorOccurred.connect(self._media_error)
        except Exception as exc:
            for component in (self._media_player, self._audio_output):
                if component is not None:
                    component.deleteLater()
            self._media_player = None
            self._audio_output = None
            self._audio_unavailable_reason = type(exc).__name__
            self.play_sentence_button.setEnabled(False)
            self.sentence_precision_note.setText("当前环境无法启用本地回听；转写、校对和导出仍可使用。")
            return False
        self.sentence_precision_note.setText("回听按识别时间范围定位；首次使用时才初始化音频设备。")
        return True

    def _play_selected_sentence(self):
        row = self.table.currentRow()
        sentence_id = self.sentence_combo.currentData()
        if row < 0 or row >= len(self.analyses) or self.analyses[row] is None or not sentence_id:
            return
        result = self.analyses[row]
        if result.source_changed:
            QMessageBox.information(self, "源音频已变化", "旧转写仍已保留；请先重新处理当前音频后再回听。")
            return
        path = result.input_path.resolve(strict=False)
        if not path.is_file():
            QMessageBox.information(self, "找不到音频", "当前音频文件不可用；本地转写结果仍已保留。")
            return
        current_fingerprint = _current_fingerprint(path)
        expected_fingerprint = None
        if result.source_size_bytes is not None and result.source_mtime_ns is not None:
            expected_fingerprint = {
                "size_bytes": result.source_size_bytes,
                "mtime_ns": result.source_mtime_ns,
            }
        if current_fingerprint is None or (
            expected_fingerprint is not None and current_fingerprint != expected_fingerprint
        ):
            result.source_changed = True
            self._persist_session()
            self._show_selected_analysis()
            QMessageBox.information(self, "源音频已变化", "旧转写仍已保留；当前文件与处理时的音频版本不同，请先重新处理。")
            return
        base = self._base_sentences(result)
        sentence = next(
            (item for index, item in enumerate(base)
             if make_sentence_id(index, item["start"], item["end"]) == sentence_id),
            None,
        )
        if sentence is None:
            return
        self.play_sentence_button.setEnabled(False)
        try:
            if not self._ensure_audio_player():
                QMessageBox.information(self, "无法回听", "当前环境无法启用本地音频回听；转写仍可继续。")
                return
        finally:
            self.play_sentence_button.setEnabled(not self._audio_unavailable_reason)
        # Event processing during initialization may change the selected recording.
        if self.table.currentRow() != row or self.sentence_combo.currentData() != sentence_id:
            return
        start_ms = max(0, int(round(float(sentence["start"]) * 1000)))
        end_ms = max(start_ms + 120, int(round(float(sentence["end"]) * 1000)))
        if result.duration_seconds is not None:
            end_ms = min(end_ms, int(round(result.duration_seconds * 1000)))
        self._play_generation += 1
        generation = self._play_generation
        self._pending_play_request = (path, sentence_id, start_ms, end_ms, generation)
        source_changed = (
            path != self._media_source_path
            or current_fingerprint != self._media_source_fingerprint
        )
        if source_changed:
            self._media_player.stop()
            self._started_play_generation = None
            self._media_source_path = path
            self._media_source_fingerprint = current_fingerprint
            self._media_player.setSource(QUrl.fromLocalFile(str(path)))
            return
        self._begin_pending_play()

    def _begin_pending_play(self):
        request = self._pending_play_request
        if request is None or self._media_player is None:
            return
        path, sentence_id, start_ms, _end_ms, generation = request
        if self._started_play_generation == generation:
            return
        row = self.table.currentRow()
        if (
            generation != self._play_generation
            or path != self._media_source_path
            or row < 0
            or row >= len(self.input_files)
            or self.input_files[row].resolve(strict=False) != path
            or self.sentence_combo.currentData() != sentence_id
        ):
            self._pending_play_request = None
            return
        try:
            from PySide6.QtMultimedia import QMediaPlayer

            if self._media_player.mediaStatus() not in {
                QMediaPlayer.MediaStatus.LoadedMedia,
                QMediaPlayer.MediaStatus.BufferedMedia,
            }:
                return
        except Exception:
            return
        self._started_play_generation = generation
        self._media_player.setPosition(start_ms)
        self._media_player.play()
        self.stop_audio_button.setEnabled(True)

    def _media_status_changed(self, status):
        if self._media_player is None or self._pending_play_request is None:
            return
        try:
            from PySide6.QtMultimedia import QMediaPlayer

            loaded_statuses = {
                QMediaPlayer.MediaStatus.LoadedMedia,
                QMediaPlayer.MediaStatus.BufferedMedia,
            }
        except Exception:
            return
        request = self._pending_play_request
        if status in loaded_statuses and request[0] == self._media_source_path:
            self._begin_pending_play()

    def _media_position_changed(self, position: int):
        request = self._pending_play_request
        if (
            request is None
            or request[4] != self._play_generation
            or request[4] != self._started_play_generation
            or self._media_player is None
        ):
            return
        try:
            if int(self._media_player.position()) != int(position):
                return
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return
        if position >= request[3]:
            self._stop_playback()

    def _media_error(self, _error, _message=""):
        if self._media_player is None:
            return
        self._pending_play_request = None
        self._started_play_generation = None
        self._play_generation += 1
        self._media_source_path = None
        self._media_source_fingerprint = None
        self.sentence_precision_note.setText("当前系统无法播放此音频格式；转写和校对仍可继续。")

    def _stop_playback(self, *, clear_source: bool = False):
        self._play_generation += 1
        self._pending_play_request = None
        self._started_play_generation = None
        if self._media_player is not None:
            self._media_player.stop()
            if clear_source:
                self._media_player.setSource(QUrl())
        if clear_source:
            self._media_source_path = None
            self._media_source_fingerprint = None

    def _show_selected_analysis(self):
        row = self.table.currentRow()
        previous_editor_row = self._editor_row
        if previous_editor_row is not None and previous_editor_row != row:
            if self.save_correction_button.isEnabled():
                self._save_sentence_correction(
                    silent=True,
                    row=previous_editor_row,
                    sentence_id=self._selected_sentence_id,
                    refresh=False,
                )
            self._stop_playback(clear_source=True)
        self.preview.clear()
        self.speaker1_example.setText("说话人1示例：—")
        self.speaker2_example.setText("说话人2示例：—")
        self.role_combo.setEnabled(False)
        self.save_button.setEnabled(False)
        self.open_word_button.setEnabled(False)
        if row < 0 or row >= len(self.analyses):
            self.selected_file_label.setText("尚未选择录音")
            self._updating_role = True
            self.role_combo.setCurrentIndex(0)
            self._updating_role = False
            self._suppress_sentence_events = True
            self.sentence_combo.clear()
            self.sentence_combo.setEnabled(False)
            self.sentence_editor.clear()
            self.sentence_editor.setEnabled(False)
            self.sentence_speaker_combo.setCurrentIndex(0)
            self.sentence_speaker_combo.setEnabled(False)
            self.save_correction_button.setEnabled(False)
            self.play_sentence_button.setEnabled(False)
            self.stop_audio_button.setEnabled(self._media_player is not None)
            self._suppress_sentence_events = False
            self._editor_row = None
            self._selected_sentence_id = None
            _set_status_badge(self.preview_state, "等待选择", "muted")
            self.preview.setPlainText("从左侧队列选择一条录音；处理完成后，这里会显示逐字稿和角色确认。")
            return
        self.selected_file_label.setText(self.input_files[row].name)
        self.selected_file_label.setToolTip(str(self.input_files[row]))
        result = self.analyses[row]
        self._updating_role = True
        self.role_combo.setCurrentIndex(0)
        self._updating_role = False
        if result is None:
            status_item = self.table.item(row, 1)
            detail_item = self.table.item(row, 2)
            status_text = status_item.text() if status_item is not None else "待处理"
            detail_text = detail_item.text() if detail_item is not None else ""
            tone = "processing" if status_text == "处理中" else "pending"
            _set_status_badge(self.preview_state, status_text, tone)
            self._suppress_sentence_events = True
            self.sentence_combo.clear()
            self.sentence_combo.setEnabled(False)
            self.sentence_editor.clear()
            self.sentence_editor.setEnabled(False)
            self.sentence_speaker_combo.setCurrentIndex(0)
            self.sentence_speaker_combo.setEnabled(False)
            self.save_correction_button.setEnabled(False)
            self.play_sentence_button.setEnabled(False)
            self.stop_audio_button.setEnabled(self._media_player is not None)
            self._suppress_sentence_events = False
            self._editor_row = None
            self._selected_sentence_id = None
            self.preview.setPlainText(detail_text or "该文件尚未处理。")
            return

        status_text, tone = _analysis_status(result)
        if result.source_changed:
            status_text, tone = "音频已变化，旧结果已保留", "warning"
        _set_status_badge(self.preview_state, status_text, tone)

        choice = self.role_choices.get(str(result.input_path))
        for index in range(self.role_combo.count()):
            if self.role_combo.itemData(index) == choice:
                self._updating_role = True
                self.role_combo.setCurrentIndex(index)
                self._updating_role = False
                break
        sample_by_speaker = {}
        for segment in result.segments:
            speaker = segment.get("speaker_id")
            if speaker in {"speaker_1", "speaker_2"} and segment.get("text", "").strip():
                sample_by_speaker.setdefault(speaker, str(segment["text"]).strip())
        self.speaker1_example.setText("说话人1示例：" + sample_by_speaker.get("speaker_1", "暂无记录"))
        self.speaker2_example.setText("说话人2示例：" + sample_by_speaker.get("speaker_2", "暂无记录"))

        if result.status == "failed":
            lines = ["处理失败", result.error]
            if result.words:
                lines.extend(["", "已保留识别文本（说话人区分未完成）："])
                for item in build_sentences(result.words, tone_enabled=result.tone_enabled):
                    start = max(0, int(item.get("start", 0)))
                    speaker = _speaker_label(item.get("speaker_id"))
                    lines.append(f"[{start // 60:02d}:{start % 60:02d}] {speaker}：{format_sentence(item)}")
            self.preview.setPlainText("\n".join(lines))
        elif result.status == "cancelled":
            self.preview.setPlainText("处理已取消\n" + result.error)
        elif result.status == "no_speech":
            self.preview.setPlainText("未检测到语音。仍可保存一份标明无语音的 Word 结果。")
        elif result.words:
            lines = []
            reviewed_sentences = revise_sentences(
                result.words,
                result.sentence_revisions,
                result.emotion_windows,
                result.tone_enabled,
                result.analysis_windows,
            )
            for item in reviewed_sentences:
                start = max(0, int(float(item["start"])))
                timestamp = f"[{start // 60:02d}:{start % 60:02d}]"
                speaker = _speaker_label(item.get("speaker_id"))
                mark = " 〔待核对〕" if item.get("uncertain") else ""
                lines.append(f"{timestamp} {speaker}{mark}：{format_sentence(item)}")
            if result.tone_enabled is not None:
                lines.extend(["", "句后声学数据取对应说话人的重叠分析窗口；缺失时显示—。"])
                if result.tone_enabled is False:
                    lines.append("情绪倾向分析：已关闭。")
                else:
                    if result.tone_status in {"failed", "partial"}:
                        state = "部分失败" if result.tone_status == "partial" else "失败"
                        lines.append(f"情绪倾向分析：{state}。{result.tone_error}")
                    lines.append("句后置信度显示未校准的模型原始分数，不代表准确率或概率。")
            if result.sentence_revisions:
                lines.extend(["", "声学统计依据原音频分段；人工校对不重算统计。"])
            self.preview.setPlainText("\n".join(lines))
        else:
            self.preview.setPlainText("没有可预览的转写内容。")

        self._populate_sentence_review(row, result)
        self.role_combo.setEnabled(
            result.status in {"ready", "exported", "export_failed"} and bool(result.words) and not result.source_changed
        )
        self.save_button.setEnabled(
            result.status in {"ready", "no_speech", "exported", "export_failed"} and not result.source_changed
        )
        self.open_word_button.setEnabled(bool(result.word_path and result.word_path.is_file()))

    def _save_results(self):
        if self.save_correction_button.isEnabled():
            self._save_sentence_correction(silent=True, refresh=False)
        output_text = self.output_edit.text().strip()
        if not output_text:
            QMessageBox.information(self, "缺少保存目录", "请选择转写结果的保存目录。")
            return
        output_root = Path(output_text).expanduser().resolve()
        selected_row = self.table.currentRow()
        eligible = [
            (index, item)
            for index, item in enumerate(self.analyses)
            if item is not None
            and not item.source_changed
            and item.status in {"ready", "no_speech", "exported", "export_failed"}
            and (index == selected_row or self._needs_export(item))
        ]
        if not eligible:
            QMessageBox.information(self, "没有待保存的结果", "请先完成录音处理。")
            return
        pipeline = ConsultationPipeline()
        saved = 0
        skipped = 0
        errors = []
        for index, result in eligible:
            desired_dir = self._output_directory_for_row(index, output_root)
            default_name = result.word_path.name if result.word_path else f"{result.input_path.stem}.docx"
            dialog = QFileDialog(self, f"保存 Word：{result.input_path.name}", str(desired_dir))
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            dialog.setFileMode(QFileDialog.FileMode.AnyFile)
            dialog.setNameFilter("Word 文档 (*.docx)")
            dialog.setDefaultSuffix("docx")
            dialog.selectFile(default_name)
            if not dialog.exec():
                skipped += 1
                continue
            chosen = Path(dialog.selectedFiles()[0]).expanduser()
            # Qt adds the default suffix for names without an extension.
            target_path = chosen if chosen.suffix.lower() == ".docx" else Path(str(chosen) + ".docx")
            target_path = target_path.resolve(strict=False)
            if target_path != chosen.resolve(strict=False) and target_path.exists():
                answer = QMessageBox.question(
                    self, "确认替换", f"{target_path.name} 已存在，是否替换？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    skipped += 1
                    continue
            try:
                if target_path == result.input_path.resolve(strict=False):
                    raise ValueError("保存路径不能与源音频相同。")
                word_path = pipeline.export_file(
                    result,
                    counselor_speaker=self.role_choices.get(str(result.input_path)),
                    target_path=target_path,
                    overwrite=target_path.exists(),
                )
                self._set_queue_status(index, "exported", "已导出")
                self._set_queue_detail(index, f"Word 已保存：{word_path.name}")
                saved += 1
            except Exception as exc:
                error = self._redact_result_error(result, str(exc)[:500])
                result.status = "export_failed"
                result.error = error
                errors.append(f"{result.input_path.name}：{error[:300]}")
                self._set_queue_status(index, "export_failed", "导出失败")
                self._set_queue_detail(index, error)
            self._persist_session()
        self._update_counts()
        self._show_selected_analysis()
        if not saved and not errors:
            return
        message = f"已保存 {saved} 个 Word 结果。"
        if skipped:
            message += f"已跳过 {skipped} 个结果。"
        if errors:
            message += "\n\n导出失败：\n" + "\n".join(errors[:8])
            QMessageBox.warning(self, "导出完成但有失败项", message)
        else:
            QMessageBox.information(self, "保存完成", message)

    def _output_directory_for_row(self, row: int, output_root: Path) -> Path:
        source = self.input_files[row].resolve(strict=False)
        if self.input_root:
            try:
                relative = source.relative_to(self.input_root.resolve(strict=False))
                return output_root / relative.parent
            except ValueError:
                pass
        return output_root

    @staticmethod
    def _redact_result_error(result: FileAnalysis, message: str) -> str:
        sanitized = str(message).strip() or "导出失败。"
        for private_path in (result.input_path, result.output_dir):
            if private_path:
                sanitized = sanitized.replace(str(private_path), "[本地路径]")
        return sanitized[:500]

    def _needs_export(self, result: FileAnalysis, desired_output_dir: Path | None = None) -> bool:
        if result.source_changed:
            return False
        if result.status in {"ready", "no_speech", "export_failed"}:
            return True
        if result.status == "exported":
            requested = self.role_choices.get(str(result.input_path))
            if not result.words:
                requested = None
            files_missing = not (
                result.word_path
                and result.word_path.is_file()
            )
            return (
                requested != result.counselor_speaker
                or result.revision != result.exported_revision
                or files_missing
                or (
                    desired_output_dir is not None
                    and desired_output_dir.resolve(strict=False) != result.output_dir.resolve(strict=False)
                )
            )
        return False

    def _open_selected(self, kind):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.analyses) or self.analyses[row] is None:
            return
        result = self.analyses[row]
        path = result.word_path
        if path is None or not path.is_file():
            QMessageBox.information(self, "文件尚未保存", "请先保存 Word 结果。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_log_folder(self):
        if not is_logging_enabled():
            QMessageBox.information(self, "日志暂不可用", "本次运行的日志尚未初始化。")
            return
        log_dir = Path(LOG_DIR)
        try:
            if not log_dir.exists() or not log_dir.is_dir():
                QMessageBox.information(self, "日志暂不可用", "本地日志文件夹尚未创建或不可用。")
                return
            status_note = ""
            try:
                log_path = log_dir / LOG_FILE_NAME
                if not log_path.exists():
                    status_note = "日志文件尚未生成。"
                elif log_path.stat().st_size == 0:
                    status_note = "日志文件目前为空。"
            except OSError:
                status_note = "日志文件状态暂时无法读取。"
            opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir.resolve())))
        except Exception:
            QMessageBox.warning(self, "无法打开日志文件夹", "系统未能打开本地日志文件夹。")
            return
        if not opened:
            message = "系统未能打开本地日志文件夹。"
            if status_note:
                message += "\n" + status_note
            QMessageBox.warning(self, "无法打开日志文件夹", message)
        elif status_note:
            QMessageBox.information(
                self,
                "日志文件夹",
                f"{status_note}\n系统已接受打开目录请求。",
            )

    def closeEvent(self, event):
        self._stop_playback(clear_source=True)
        if self.worker and self.worker.isRunning():
            self._close_when_done = True
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.progress_label.setText("正在安全停止当前处理；窗口会在处理线程结束后关闭。")
            event.ignore()
            return
        if self.save_correction_button.isEnabled():
            self._save_sentence_correction(silent=True, refresh=False)
        self._stop_playback(clear_source=True)
        self._persist_session()
        event.accept()


def run_app(startup_window=None) -> int:
    try:
        try:
            init_logging()
        except Exception:
            pass
        from utils.windows_shell import set_process_identity

        set_process_identity()
        app = QApplication.instance() or QApplication([])
        _configure_application_icon(app)
        _configure_application_fonts(app)
        window = MainWindow()
        window.setWindowIcon(app.windowIcon())
        window.show()
        _configure_window_identity(window)
        app.processEvents()
        from app.startup import finish_startup

        finish_startup(startup_window)
        return app.exec()
    finally:
        try:
            shutdown_logging()
        except Exception:
            pass


def _configure_application_icon(app: QApplication) -> None:
    icon_path = Path(__file__).resolve().parents[1] / "assets" / "consult_asr.ico"
    icon = QIcon(str(icon_path))
    if not icon.isNull():
        app.setWindowIcon(icon)


def _configure_window_identity(window: QMainWindow) -> dict[str, bool]:
    app = QApplication.instance()
    if os.name != "nt" or app is None or app.platformName() != "windows":
        return {"icons": False, "shell": False}
    from utils.windows_shell import set_window_identity, release_icons

    icon_path = Path(__file__).resolve().parents[1] / "assets" / "consult_asr.ico"
    launcher = os.environ.get("CONSULT_ASR_LAUNCHER_PATH", "")
    result, handles = set_window_identity(int(window.winId()), icon_path, launcher)
    if handles:
        window.destroyed.connect(lambda *_: release_icons(handles))
    return result


def _file_status(status: str) -> tuple[str, str]:
    return {
        "ready": ("识别完成，待导出", "ready"),
        "no_speech": ("未检测到语音", "no_speech"),
        "failed": ("失败", "failed"),
        "export_failed": ("导出失败", "export_failed"),
        "cancelled": ("已取消", "cancelled"),
        "exported": ("已导出", "exported"),
    }.get(status, (status or "未知状态", "pending"))


def _analysis_status(result: FileAnalysis) -> tuple[str, str]:
    status, tone = _file_status(result.status)
    if result.status in {"ready", "exported"}:
        if result.tone_status == "failed":
            return ("转写完成，情绪分析失败" if result.status == "ready" else "已导出，情绪分析失败", "warning")
        if result.tone_status == "partial":
            return ("转写完成，情绪分析部分失败" if result.status == "ready" else "已导出，情绪分析部分失败", "warning")
    return status, tone


def _speaker_label(speaker_id: str | None) -> str:
    return {
        "speaker_1": "说话人1",
        "speaker_2": "说话人2",
    }.get(speaker_id, "说话人待确认" if not speaker_id else speaker_id)


def _set_status_badge(label: QLabel, text: str, tone: str) -> None:
    colors = {
        "ready": ("#38654f", "#e8f1e9", "#d7e6d9"),
        "warning": ("#80601d", "#f7efd9", "#eadcb7"),
        "processing": ("#816320", "#f8f0dc", "#eadcb7"),
        "pending": ("#69776d", "#f0f2ef", "#e3e7e2"),
        "muted": ("#69776d", "#f0f2ef", "#e3e7e2"),
        "failed": ("#98453d", "#f8e9e6", "#efd4cf"),
        "export_failed": ("#98453d", "#f8e9e6", "#efd4cf"),
        "cancelled": ("#69776d", "#f0f2ef", "#e3e7e2"),
        "no_speech": ("#69776d", "#f0f2ef", "#e3e7e2"),
        "exported": ("#38654f", "#e8f1e9", "#d7e6d9"),
    }
    foreground, background, border = colors.get(tone, colors["muted"])
    label.setText(text)
    label.setStyleSheet(
        "QLabel {"
        f"color: {foreground}; background: {background}; border: 1px solid {border};"
        "border-radius: 10px; padding: 3px 8px;"
        "}"
    )


class _SageRoleComboBox(QComboBox):
    """Paint a small local triangle over the Qt combo box drop-down area."""

    def paintEvent(self, event):
        super().paintEvent(event)
        center_x = self.width() - 12
        center_y = self.height() // 2
        triangle = QPolygon(
            [
                QPoint(center_x - 4, center_y - 2),
                QPoint(center_x + 4, center_y - 2),
                QPoint(center_x, center_y + 3),
            ]
        )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#748177"))
        painter.drawPolygon(triangle)


def _window_stylesheet() -> str:
    return """
    QMainWindow, QWidget#centralRoot {
        background: #f5f5f0;
        color: #293a32;
    }
    QLabel { color: #293a32; }
    QLabel#mutedLabel { color: #748177; }
    QLabel#smallBadge {
        color: #65766b;
        background: #eef2ed;
        border: 1px solid #e0e6de;
        border-radius: 10px;
        padding: 4px 8px;
    }
    QLabel#emptyTitle { font-size: 12pt; font-weight: 600; }
    QLabel#exampleCard {
        color: #53655b;
        background: #f4f7f3;
        border: 1px solid #e4eae2;
        border-radius: 9px;
        padding: 8px;
    }
    QGroupBox#surfaceCard {
        background: #fffefa;
        border: 1px solid #e0e6df;
        border-radius: 13px;
        margin-top: 7px;
        font-weight: 600;
    }
    QGroupBox#surfaceCard::title {
        subcontrol-origin: margin;
        left: 13px;
        padding: 0 4px;
        color: #52645a;
    }
    QScrollArea#reviewScroll { background: transparent; border: none; }
    QGroupBox#correctionCard {
        background: #f7f9f6;
        border: 1px solid #dde6dd;
        border-radius: 9px;
        margin-top: 7px;
        font-weight: normal;
    }
    QGroupBox#correctionCard::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: #52645a;
    }
    QWidget#emptyState {
        background: #f7f8f4;
        border: 1px solid #eef0eb;
        border-radius: 10px;
    }
    QWidget#outputRow {
        background: #fffefa;
        border: 1px solid #e0e6df;
        border-radius: 10px;
    }
    QPushButton {
        color: #34483d;
        background: #fffefa;
        border: 1px solid #d8e1d8;
        border-radius: 8px;
        padding: 6px 11px;
        min-height: 19px;
    }
    QPushButton:hover { background: #f1f6f1; border-color: #b9cdbd; }
    QPushButton:pressed { background: #e8f0e8; }
    QPushButton:disabled { color: #96a197; background: #edf0ec; border-color: #e4e8e3; }
    QPushButton#primaryButton {
        color: #ffffff;
        background: #386c55;
        border: 1px solid #386c55;
        font-weight: 600;
    }
    QPushButton#primaryButton:hover { background: #2f614b; border-color: #2f614b; }
    QPushButton#primaryButton:disabled { color: #8c9990; background: #dfe7e0; border-color: #dfe7e0; }
    QPushButton:focus { border-color: #386c55; }
    QPushButton#secondaryButton { background: #edf4ef; border-color: #c5d8cc; color: #305641; }
    QPushButton#secondaryButton:hover { background: #dfebe2; border-color: #9fbea9; }
    QPushButton#secondaryButton:pressed { background: #cbded1; }
    QPushButton#secondaryButton:disabled { color: #96a197; background: #edf0ec; border-color: #e4e8e3; }
    QPushButton#quietButton { background: #e0eeea; border-color: #b2cdc5; color: #24574c; }
    QPushButton#quietButton:hover { background: #cee3dc; border-color: #89b3a6; }
    QPushButton#quietButton:pressed, QPushButton#quietButton:checked { background: #bdd8cd; border-color: #739e8f; }
    QPushButton#quietButton:focus, QPushButton#secondaryButton:focus { border-color: #386c55; }
    QPushButton#quietButton:disabled { color: #96a197; background: #edf0ec; border-color: #e4e8e3; }
    QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTableWidget {
        color: #293a32;
        background: #ffffff;
        border: 1px solid #dce4dc;
        border-radius: 8px;
        selection-background-color: #dcebe0;
        padding: 5px 7px;
    }
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus {
        border-color: #8fae97;
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 24px;
        border: none;
    }
    QComboBox#sageCombo::down-arrow { image: none; width: 0px; height: 0px; }
    QTableWidget { padding: 2px; alternate-background-color: #fbfcf9; }
    QHeaderView::section {
        color: #728177;
        background: #f7f9f6;
        border: none;
        border-bottom: 1px solid #e5eae4;
        padding: 5px 6px;
        font-weight: 600;
    }
    QTableWidget::item { border-bottom: 1px solid #eef1ec; padding: 5px; }
    QTableWidget::item:selected { background: #e8f1e9; color: #293a32; }
    QPlainTextEdit#transcriptPreview { padding: 9px; }
    QSplitter::handle { background: transparent; }
    QProgressBar#thinProgress {
        background: #e9eee8;
        border: none;
        border-radius: 4px;
        max-height: 8px;
        text-visible: false;
    }
    QProgressBar#thinProgress::chunk { background: #7da087; border-radius: 4px; }
    """


def _configure_application_fonts(app: QApplication) -> None:
    """Load an installed Windows CJK font when Qt's offscreen font list is empty."""
    windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    candidates = [
        windows_dir / "Fonts" / "msyh.ttc",
        windows_dir / "Fonts" / "NotoSansSC-VF.ttf",
        windows_dir / "Fonts" / "simhei.ttf",
    ]
    for font_path in candidates:
        if font_path.is_file():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
            if families:
                app.setFont(QFont(families[0], 9))
                return

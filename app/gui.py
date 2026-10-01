from __future__ import annotations

import os
from pathlib import Path
from threading import Event

from PySide6.QtCore import QThread, Qt, Signal, QUrl, QTimer, QPoint
from PySide6.QtGui import QColor, QDesktopServices, QFont, QFontDatabase, QPainter, QPolygon
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.pipeline import ConsultationPipeline, FileAnalysis
from config import (
    ASR_MODEL_DIR,
    DIARIZATION_MODEL_DIR,
    OUTPUT_DIR,
    TONE_DEFAULT_ENABLED,
    TONE_MODEL_DIR,
    TONE_RUNTIME_DIR,
)
from core.diarizer import Diarizer
from core.transcriber import Transcriber
from tools.prepare_models import verify_models
from utils.audio_utils import collect_audio_files


class ProcessingWorker(QThread):
    progress = Signal(int, int, str)
    analysis_ready = Signal(int, object)

    def __init__(self, files, input_root, output_root, device_choice, compute_type, beam_size, tone_enabled=None):
        super().__init__()
        self.files = list(files)
        self.input_root = Path(input_root).resolve() if input_root else None
        self.output_root = Path(output_root).resolve()
        self.device_choice = device_choice
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.tone_enabled = tone_enabled
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
            pipeline = ConsultationPipeline(
                device=device,
                compute_type=compute_type,
                beam_size=self.beam_size,
                tone_enabled=self.tone_enabled,
            )
            total = len(self.files)
            for index, path in enumerate(self.files, 1):
                source = Path(path).resolve()
                if self.input_root:
                    try:
                        relative = source.relative_to(self.input_root)
                        destination = self.output_root / relative.parent
                    except ValueError:
                        destination = self.output_root
                else:
                    destination = self.output_root

                if self.cancel_event.is_set():
                    analysis = FileAnalysis(
                        source,
                        destination,
                        status="cancelled",
                        error="在开始前取消。",
                    )
                    self.analysis_ready.emit(index - 1, analysis)
                    completed_indices.add(index - 1)
                    continue

                self.progress.emit(index, total, f"[{index}/{total}] 正在处理：{source.name}")
                analysis = pipeline.analyze_file(
                    source,
                    destination,
                    cancel_event=self.cancel_event,
                    progress=lambda stage, i=index, t=total: self.progress.emit(
                        i, t, f"[{i}/{t}] {stage}"
                    ),
                )
                self.analysis_ready.emit(index - 1, analysis)
                completed_indices.add(index - 1)
        except Exception as exc:
            # Unexpected queue-level errors are reported as failed items, never as success.
            for index, path in enumerate(self.files):
                if index in completed_indices:
                    continue
                try:
                    source = Path(path).resolve()
                except (OSError, RuntimeError, ValueError):
                    source = Path(str(path))
                analysis = FileAnalysis(
                    source,
                    self.output_root,
                    status="failed",
                    error=f"处理队列失败：{type(exc).__name__}: {str(exc)[:500]}",
                )
                self.analysis_ready.emit(index, analysis)
                completed_indices.add(index)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        application = QApplication.instance()
        if application is not None:
            _configure_application_fonts(application)
        self.setWindowTitle("咨询录音转写")
        self.resize(1160, 760)
        self.setMinimumSize(900, 600)
        self.input_files: list[Path] = []
        self.input_root: Path | None = None
        self.analyses: list[FileAnalysis | None] = []
        self.worker: ProcessingWorker | None = None
        self.role_choices: dict[str, str | None] = {}
        self._updating_role = False
        self._close_when_done = False
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

        self.choose_files_button = QPushButton("添加录音")
        self.choose_files_button.setObjectName("secondaryButton")
        self.choose_files_button.clicked.connect(self._choose_files)
        self.choose_folder_button = QPushButton("添加文件夹")
        self.choose_folder_button.setObjectName("secondaryButton")
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
        self.open_json_button = QPushButton("打开 JSON")
        self.open_json_button.setObjectName("secondaryButton")
        self.open_json_button.clicked.connect(lambda: self._open_selected("json"))
        self.save_button = QPushButton("保存 Word + JSON")
        self.save_button.setObjectName("primaryButton")
        self.save_button.setToolTip("保存尚未导出的文件，以及角色设置发生变化的文件。")
        self.save_button.clicked.connect(self._save_results)
        result_actions.addWidget(self.open_word_button)
        result_actions.addWidget(self.open_json_button)
        result_actions.addWidget(self.save_button)
        review_layout.addLayout(result_actions)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(7)
        queue.setMinimumWidth(300)
        review.setMinimumWidth(420)
        splitter.addWidget(queue)
        splitter.addWidget(review)
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
        self.output_edit.setToolTip("结果保存到此文件夹；Word 与 JSON 会成对保存。")
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

    def _set_inputs(self, files, root):
        unique = sorted({Path(item).resolve() for item in files}, key=lambda item: str(item).casefold())
        self.input_files = unique
        self.input_root = Path(root).resolve() if root else None
        self.analyses = [None] * len(unique)
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
        self.queue_empty.setVisible(not unique)
        self.table.setVisible(bool(unique))
        if unique:
            self.table.selectRow(0)
        self.preview.clear()
        self._update_counts()
        self._show_selected_analysis()

    def _clear_queue(self):
        if self.worker and self.worker.isRunning():
            return
        self.input_files = []
        self.input_root = None
        self.analyses = []
        self.table.setRowCount(0)
        self.source_summary.setText("0 项")
        self.queue_empty.show()
        self.table.hide()
        self.preview.clear()
        self._update_counts()
        self._show_selected_analysis()

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

        self.analyses = [None] * len(self.input_files)
        self.role_choices.clear()
        for row in range(len(self.input_files)):
            self._set_queue_status(row, "pending", "待处理")
            self._set_queue_detail(row, "尚未处理")
        device_choice = self.device_combo.currentData() if self.advanced_toggle.isChecked() else "auto"
        compute_type = self.compute_combo.currentData() if self.advanced_toggle.isChecked() else "float16"
        beam_size = self.beam_spin.value() if self.advanced_toggle.isChecked() else 5
        self.worker = ProcessingWorker(
            self.input_files,
            self.input_root,
            output_root,
            device_choice,
            compute_type,
            beam_size,
            self.tone_enabled_check.isChecked(),
        )
        self.worker.progress.connect(self._progress)
        self.worker.analysis_ready.connect(self._analysis_ready)
        self.worker.finished.connect(self._finished)
        self.start_button.setEnabled(False)
        self.tone_enabled_check.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.choose_files_button.setEnabled(False)
        self.choose_folder_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("正在准备处理队列…")
        self.worker.start()

    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.progress_label.setText("已请求取消；将于当前安全处理边界停止。")

    def _progress(self, current, total, message):
        self.progress_label.setText(message)
        self.progress_bar.setValue(int((current - 1) / max(total, 1) * 100))
        row = current - 1
        if 0 <= row < len(self.analyses) and self.analyses[row] is None:
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
        status_text, tone = _analysis_status(result)
        self._set_queue_status(row, tone, status_text)
        detail = result.error or f"{len(result.words)} 个时间片段 · {result.elapsed_seconds:.1f} 秒"
        if result.tone_status in {"failed", "partial"} and result.tone_error:
            detail += f" · 情绪分析：{result.tone_error[:180]}"
        self._set_queue_detail(row, detail)
        self._update_counts()
        if self.table.currentRow() == row:
            self._show_selected_analysis()

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

    def _show_selected_analysis(self):
        row = self.table.currentRow()
        self.preview.clear()
        self.speaker1_example.setText("说话人1示例：—")
        self.speaker2_example.setText("说话人2示例：—")
        self.role_combo.setEnabled(False)
        self.save_button.setEnabled(False)
        self.open_word_button.setEnabled(False)
        self.open_json_button.setEnabled(False)
        if row < 0 or row >= len(self.analyses):
            self.selected_file_label.setText("尚未选择录音")
            self._updating_role = True
            self.role_combo.setCurrentIndex(0)
            self._updating_role = False
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
            self.preview.setPlainText(detail_text or "该文件尚未处理。")
            return

        status_text, tone = _analysis_status(result)
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
                for item in result.words:
                    start = max(0, int(item.get("start", 0)))
                    speaker = _speaker_label(item.get("speaker_id"))
                    lines.append(f"[{start // 60:02d}:{start % 60:02d}] {speaker}：{item.get('text', '')}")
            self.preview.setPlainText("\n".join(lines))
        elif result.status == "cancelled":
            self.preview.setPlainText("处理已取消\n" + result.error)
        elif result.status == "no_speech":
            self.preview.setPlainText("未检测到语音。仍可保存一份标明无语音的 Word 和 JSON 结果。")
        elif result.words:
            lines = []
            for item in result.segments:
                start = max(0, int(float(item["start"])))
                timestamp = f"[{start // 60:02d}:{start % 60:02d}]"
                speaker = _speaker_label(item.get("speaker_id"))
                mark = " 〔待核对〕" if item.get("uncertain") else ""
                lines.append(f"{timestamp} {speaker}：{item['text']}{mark}")
            if result.tone_enabled is not None:
                lines.extend(["", "语音特征"])
                for window in result.analysis_windows:
                    features = _format_acoustic_features(window.get("acoustics"))
                    start = max(0, int(float(window.get("start", 0))))
                    end = max(0, int(float(window.get("end", 0))))
                    timestamp = f"[{start // 60:02d}:{start % 60:02d}–{end // 60:02d}:{end % 60:02d}]"
                    speaker = _speaker_label(window.get("speaker_id"))
                    if features:
                        lines.append(f"{timestamp} {speaker}：{features}")
                    elif not window.get("analysis_safe"):
                        lines.append(
                            f"{timestamp} {speaker}：声学特征未知（{_emotion_reason_zh(window.get('reason'))}）"
                        )
                if result.tone_enabled is False:
                    lines.append("情绪倾向分析：已关闭。")
                else:
                    if result.tone_status in {"failed", "partial"}:
                        state = "部分失败" if result.tone_status == "partial" else "失败"
                        lines.append(f"情绪倾向分析：{state}。{result.tone_error}")
                    lines.append("情绪趋势（模型置信分数为未校准原始输出）")
                    for window in result.analysis_windows:
                        start = max(0, int(float(window.get("start", 0))))
                        end = max(0, int(float(window.get("end", 0))))
                        speaker = _speaker_label(window.get("speaker_id"))
                        if window.get("status") == "ok":
                            label = _emotion_label_zh(str(window.get("label") or "unknown"))
                            score = window.get("model_score")
                            detail = f"{label} · 模型置信分数 {float(score):.3f}" if isinstance(score, (int, float)) else label
                        else:
                            detail = f"未知（{_emotion_reason_zh(window.get('reason'))}）"
                        lines.append(f"[{start // 60:02d}:{start % 60:02d}–{end // 60:02d}:{end % 60:02d}] {speaker}：{detail}")
            self.preview.setPlainText("\n".join(lines))
        else:
            self.preview.setPlainText("没有可预览的转写内容。")

        self.role_combo.setEnabled(
            result.status in {"ready", "exported", "export_failed"} and bool(result.words)
        )
        self.save_button.setEnabled(result.status in {"ready", "no_speech", "exported", "export_failed"})
        self.open_word_button.setEnabled(bool(result.word_path and result.word_path.is_file()))
        self.open_json_button.setEnabled(bool(result.json_path and result.json_path.is_file()))

    def _save_results(self):
        eligible = [
            (index, item)
            for index, item in enumerate(self.analyses)
            if item is not None and self._needs_export(item)
        ]
        if not eligible:
            QMessageBox.information(self, "没有待保存的结果", "结果已保存，所选角色也没有变更。")
            return
        pipeline = ConsultationPipeline()
        saved = 0
        errors = []
        for index, result in eligible:
            speaker = self.role_choices.get(str(result.input_path))
            try:
                word_path, json_path = pipeline.export_file(result, counselor_speaker=speaker)
                self._set_queue_status(index, "exported", "已导出")
                self._set_queue_detail(index, f"Word + JSON 已保存：{word_path.name}")
                saved += 1
            except Exception as exc:
                result.status = "export_failed"
                result.error = str(exc)[:500]
                errors.append(f"{result.input_path.name}：{result.error[:300]}")
                self._set_queue_status(index, "export_failed", "导出失败")
                self._set_queue_detail(index, result.error)
        self._update_counts()
        self._show_selected_analysis()
        message = f"已保存 {saved} 个 Word + JSON 结果。"
        if errors:
            message += "\n\n导出失败：\n" + "\n".join(errors[:8])
            QMessageBox.warning(self, "导出完成但有失败项", message)
        else:
            QMessageBox.information(self, "保存完成", message)

    def _needs_export(self, result: FileAnalysis) -> bool:
        if result.status in {"ready", "no_speech", "export_failed"}:
            return True
        if result.status == "exported":
            requested = self.role_choices.get(str(result.input_path))
            if not result.words:
                requested = None
            return requested != result.counselor_speaker
        return False

    def _open_selected(self, kind):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.analyses) or self.analyses[row] is None:
            return
        result = self.analyses[row]
        path = result.word_path if kind == "word" else result.json_path
        if path is None or not path.is_file():
            QMessageBox.information(self, "文件尚未保存", "请先保存 Word 和 JSON 结果。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self._close_when_done = True
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.progress_label.setText("正在安全停止当前处理；窗口会在处理线程结束后关闭。")
            event.ignore()
            return
        event.accept()


def run_app() -> int:
    app = QApplication.instance() or QApplication([])
    _configure_application_fonts(app)
    window = MainWindow()
    window.show()
    return app.exec()


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


def _format_acoustic_features(value) -> str:
    if not isinstance(value, dict):
        return ""
    parts = []
    rate = value.get("speech_rate_chinese_chars_per_min")
    if isinstance(rate, (int, float)):
        count = value.get("chinese_character_count")
        duration = value.get("denominator_seconds")
        if isinstance(count, int) and isinstance(duration, (int, float)):
            parts.append(f"语速（ASR估计）{float(rate):.1f} 个中文字/分钟（{count} 字/{float(duration):.2f} 秒）")
        else:
            parts.append(f"语速（ASR估计）{float(rate):.1f} 个中文字/分钟")
    rms = value.get("rms_dbfs")
    if isinstance(rms, (int, float)):
        parts.append(f"音量 RMS {float(rms):.1f} dBFS")
    pause_count = value.get("pause_count")
    pause_seconds = value.get("pause_total_seconds")
    if isinstance(pause_count, int) and isinstance(pause_seconds, (int, float)):
        parts.append(f"内部停顿 {pause_count} 次/{float(pause_seconds):.2f} 秒")
    f0 = value.get("f0_median_hz")
    if isinstance(f0, (int, float)):
        parts.append(f"基频中位数 {float(f0):.1f} Hz")
    contour = value.get("final_contour_candidate")
    if contour in {"up", "down", "flat"}:
        label = {"up": "上扬", "down": "下降", "flat": "平稳"}[contour]
        delta = value.get("semitone_delta")
        suffix = f" {float(delta):+.1f} 半音" if isinstance(delta, (int, float)) else ""
        parts.append(f"句末音高走势候选 {label}{suffix}（不直接判断疑问句）")
    if not parts and value.get("analysis_status") == "invalid":
        return f"声学分析不可用（{value.get('analysis_reason') or '输入无效'}）"
    return " · ".join(parts)


def _emotion_label_zh(label: str) -> str:
    return {
        "angry": "愤怒",
        "disgusted": "厌恶",
        "fearful": "害怕",
        "happy": "开心",
        "neutral": "平静",
        "other": "其他",
        "sad": "悲伤",
        "surprised": "惊讶",
        "unknown": "未知",
    }.get(label.casefold(), label or "未知")


def _emotion_reason_zh(reason) -> str:
    return {
        "uncertain_word_or_speaker": "转写置信度或说话人归属不确定",
        "speaker_unassigned": "说话人未可靠归属",
        "speaker_overlap": "存在说话人重叠",
        "no_speaker_track": "当前时间没有说话人轨道覆盖",
        "speaker_alignment_mismatch": "转写词与说话人轨道不一致",
        "window_too_short": "可靠语音窗口过短",
        "emotion_runtime_failed": "情绪模型运行失败",
    }.get(str(reason), "当前窗口无法可靠分析")


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
    QPushButton#quietButton { background: transparent; border-color: transparent; color: #62756a; }
    QPushButton#quietButton:hover { background: #edf2ed; border-color: #e2e9e1; }
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

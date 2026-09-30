from __future__ import annotations

import os
from pathlib import Path
from threading import Event

from PySide6.QtCore import QThread, Qt, Signal, QUrl, QTimer
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
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
from config import ASR_MODEL_DIR, DIARIZATION_MODEL_DIR, OUTPUT_DIR
from core.diarizer import Diarizer
from core.transcriber import Transcriber
from tools.prepare_models import verify_models
from utils.audio_utils import collect_audio_files


class ProcessingWorker(QThread):
    progress = Signal(int, int, str)
    analysis_ready = Signal(int, object)

    def __init__(self, files, input_root, output_root, device_choice, compute_type, beam_size):
        super().__init__()
        self.files = list(files)
        self.input_root = Path(input_root).resolve() if input_root else None
        self.output_root = Path(output_root).resolve()
        self.device_choice = device_choice
        self.compute_type = compute_type
        self.beam_size = beam_size
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
        self.resize(1120, 820)
        self.setMinimumSize(860, 660)
        self.input_files: list[Path] = []
        self.input_root: Path | None = None
        self.analyses: list[FileAnalysis | None] = []
        self.worker: ProcessingWorker | None = None
        self.role_choices: dict[str, str | None] = {}
        self._updating_role = False
        self._close_when_done = False

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(11)

        title = QLabel("咨询录音转写")
        title_font = QFont()
        title_font.setPointSize(19)
        title_font.setWeight(QFont.Weight.DemiBold)
        title.setFont(title_font)
        subtitle = QLabel("在本机完成中文转写与双人区分。录音不会上传；先看示例发言，再决定是否标注咨询师/来访者。")
        subtitle.setWordWrap(True)
        outer.addWidget(title)
        outer.addWidget(subtitle)

        readiness = QHBoxLayout()
        self.asr_ready_label = QLabel()
        self.diarization_ready_label = QLabel()
        readiness.addWidget(self.asr_ready_label)
        readiness.addSpacing(18)
        readiness.addWidget(self.diarization_ready_label)
        readiness.addStretch(1)
        refresh_models = QPushButton("刷新模型状态")
        refresh_models.clicked.connect(self._refresh_model_state)
        readiness.addWidget(refresh_models)
        outer.addLayout(readiness)

        source_row = QHBoxLayout()
        self.source_summary = QLabel("尚未选择音频")
        self.source_summary.setMinimumWidth(300)
        source_row.addWidget(self.source_summary, 1)
        self.choose_files_button = QPushButton("选择音频文件…")
        self.choose_files_button.clicked.connect(self._choose_files)
        self.choose_folder_button = QPushButton("选择音频文件夹…")
        self.choose_folder_button.clicked.connect(self._choose_folder)
        self.clear_button = QPushButton("清空")
        self.clear_button.clicked.connect(self._clear_queue)
        source_row.addWidget(self.choose_files_button)
        source_row.addWidget(self.choose_folder_button)
        source_row.addWidget(self.clear_button)
        outer.addLayout(source_row)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("保存到"))
        self.output_edit = QLineEdit(str(OUTPUT_DIR))
        output_row.addWidget(self.output_edit, 1)
        choose_output = QPushButton("选择…")
        choose_output.clicked.connect(self._choose_output)
        output_row.addWidget(choose_output)
        outer.addLayout(output_row)

        self.advanced_toggle = QPushButton("显示高级识别设置")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        outer.addWidget(self.advanced_toggle, 0, Qt.AlignmentFlag.AlignLeft)
        self.advanced_panel = QWidget()
        advanced_layout = QGridLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.addWidget(QLabel("设备"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItem("自动（优先使用可用 GPU）", "auto")
        self.device_combo.addItem("CPU", "cpu")
        advanced_layout.addWidget(self.device_combo, 0, 1)
        advanced_layout.addWidget(QLabel("识别方式"), 0, 2)
        self.compute_combo = QComboBox()
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

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["音频", "状态", "说明"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._show_selected_analysis)
        self.table.setMinimumHeight(165)
        outer.addWidget(self.table, 2)

        activity_row = QHBoxLayout()
        self.progress_label = QLabel("正在检查本地模型…")
        activity_row.addWidget(self.progress_label, 1)
        self.counts_label = QLabel("待处理 0  ·  完成 0  ·  失败 0  ·  已取消 0")
        activity_row.addWidget(self.counts_label)
        outer.addLayout(activity_row)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        outer.addWidget(self.progress_bar)

        review = QGroupBox("转写预览与角色确认")
        review_layout = QVBoxLayout(review)
        example_row = QHBoxLayout()
        self.speaker1_example = QLabel("说话人1示例：—")
        self.speaker2_example = QLabel("说话人2示例：—")
        for label in (self.speaker1_example, self.speaker2_example):
            label.setWordWrap(True)
            label.setMinimumWidth(200)
            label.setStyleSheet("padding: 7px; background: #f3f6f5; border-radius: 5px;")
        example_row.addWidget(self.speaker1_example, 1)
        example_row.addWidget(self.speaker2_example, 1)
        review_layout.addLayout(example_row)

        role_row = QHBoxLayout()
        role_row.addWidget(QLabel("所选录音的角色"))
        self.role_combo = QComboBox()
        self.role_combo.addItem("保留说话人1 / 说话人2", None)
        self.role_combo.addItem("确认：说话人1是咨询师", "speaker_1")
        self.role_combo.addItem("确认：说话人2是咨询师", "speaker_2")
        self.role_combo.currentIndexChanged.connect(self._role_changed)
        role_row.addWidget(self.role_combo, 1)
        self.save_button = QPushButton("保存结果（Word + JSON）")
        self.save_button.setToolTip("保存尚未导出的文件，以及角色设置发生变化的文件。")
        self.save_button.clicked.connect(self._save_results)
        role_row.addWidget(self.save_button)
        review_layout.addLayout(role_row)

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("处理完成后，这里会显示逐字稿和需要核对的片段。")
        self.preview.setMinimumHeight(170)
        review_layout.addWidget(self.preview)
        open_row = QHBoxLayout()
        self.open_word_button = QPushButton("打开所选 Word")
        self.open_word_button.clicked.connect(lambda: self._open_selected("word"))
        self.open_json_button = QPushButton("打开所选 JSON")
        self.open_json_button.clicked.connect(lambda: self._open_selected("json"))
        open_row.addStretch(1)
        open_row.addWidget(self.open_word_button)
        open_row.addWidget(self.open_json_button)
        review_layout.addLayout(open_row)
        outer.addWidget(review, 3)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self.start_button = QPushButton("开始处理")
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self._start)
        self.cancel_button = QPushButton("安全取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        action_row.addWidget(self.cancel_button)
        action_row.addWidget(self.start_button)
        outer.addLayout(action_row)

        self.setCentralWidget(central)
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
        models_ready = asr_ready and diarization_ready
        self.asr_ready_label.setText(
            f"large-v3：{'已就绪' if asr_ready else '未准备'}"
        )
        self.diarization_ready_label.setText(
            f"双人区分模型：{'已就绪' if diarization_ready else '未准备'}"
        )
        self.asr_ready_label.setToolTip(str(ASR_MODEL_DIR))
        self.diarization_ready_label.setToolTip(str(DIARIZATION_MODEL_DIR))
        color = "#25633d" if asr_ready and diarization_ready else "#8a5a14"
        self.asr_ready_label.setStyleSheet(f"color: {color};")
        self.diarization_ready_label.setStyleSheet(f"color: {color};")
        if not (self.worker and self.worker.isRunning()):
            self.start_button.setEnabled(models_ready)
            if not preserve_status:
                self.progress_label.setText(
                    "本地模型已齐全，可以开始处理。"
                    if models_ready
                    else "本地模型未齐全；准备完成后请刷新模型状态。"
                )

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
            self.table.setItem(row, 0, QTableWidgetItem(path.name))
            self.table.setItem(row, 1, QTableWidgetItem("待处理"))
            self.table.setItem(row, 2, QTableWidgetItem(str(path.parent)))
        self.source_summary.setText(f"已选择 {len(unique)} 个音频文件")
        if unique:
            self.table.selectRow(0)
        self.preview.clear()
        self._update_counts()

    def _clear_queue(self):
        if self.worker and self.worker.isRunning():
            return
        self.input_files = []
        self.input_root = None
        self.analyses = []
        self.table.setRowCount(0)
        self.source_summary.setText("尚未选择音频")
        self.preview.clear()
        self._update_counts()

    def _choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择保存目录", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _start(self):
        if self.worker and self.worker.isRunning():
            return
        self._refresh_model_state()
        if not self.start_button.isEnabled():
            QMessageBox.information(
                self,
                "本地模型未齐全",
                "请先准备 large-v3 与 pyannote 双人区分模型，再刷新模型状态。",
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
            self.table.item(row, 1).setText("待处理")
            self.table.item(row, 2).setText(str(self.input_files[row].parent))
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
        )
        self.worker.progress.connect(self._progress)
        self.worker.analysis_ready.connect(self._analysis_ready)
        self.worker.finished.connect(self._finished)
        self.start_button.setEnabled(False)
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

    def _analysis_ready(self, row: int, result: FileAnalysis):
        if row < 0 or row >= len(self.analyses):
            return
        self.analyses[row] = result
        status_text = {
            "ready": "识别完成，待导出",
            "no_speech": "未检测到语音",
            "failed": "失败",
            "export_failed": "导出失败",
            "cancelled": "已取消",
            "exported": "已导出",
        }.get(result.status, result.status)
        self.table.item(row, 1).setText(status_text)
        detail = result.error or f"{len(result.words)} 个时间片段 · {result.elapsed_seconds:.1f} 秒"
        self.table.item(row, 2).setText(detail)
        self._update_counts()
        if self.table.currentRow() == row:
            self._show_selected_analysis()

    def _finished(self):
        self.cancel_button.setEnabled(False)
        self.choose_files_button.setEnabled(True)
        self.choose_folder_button.setEnabled(True)
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
            return
        result = self.analyses[row]
        self._updating_role = True
        self.role_combo.setCurrentIndex(0)
        self._updating_role = False
        if result is None:
            self.preview.setPlainText("该文件尚未处理。")
            return

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
                    lines.append(f"[{start // 60:02d}:{start % 60:02d}] {item.get('text', '')}")
            self.preview.setPlainText("\n".join(lines))
        elif result.status == "cancelled":
            self.preview.setPlainText("处理已取消\n" + result.error)
        elif result.status == "no_speech":
            self.preview.setPlainText("未检测到语音。仍可保存一份标明无语音的 Word 和 JSON 结果。")
        elif result.words:
            lines = []
            for item in result.segments:
                start = max(0, int(item["start"]))
                timestamp = f"[{start // 60:02d}:{start % 60:02d}]"
                speaker = item.get("speaker_id") or "说话人待确认"
                mark = " 〔待核对〕" if item.get("uncertain") else ""
                lines.append(f"{timestamp} {speaker}：{item['text']}{mark}")
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
                self.table.item(index, 1).setText("已导出")
                self.table.item(index, 2).setText(f"Word + JSON 已保存：{word_path.name}")
                saved += 1
            except Exception as exc:
                result.status = "export_failed"
                result.error = str(exc)[:500]
                errors.append(f"{result.input_path.name}：{result.error[:300]}")
                self.table.item(index, 1).setText("导出失败")
                self.table.item(index, 2).setText(result.error)
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
